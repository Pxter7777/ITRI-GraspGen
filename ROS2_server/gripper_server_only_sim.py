# execute it with /usr/bin/python3
import time
import rclpy
from rclpy.node import Node
import numpy as np
import argparse
import logging
import os
import sys
from send_traj_socket import send_traj

# 手動加入專案路徑
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from common_utils import network_config
from common_utils.socket_communication import (
    NonBlockingJSONReceiver,
    NonBlockingJSONSender,
)
from common_utils.custom_logger import CustomFormatter

logger = logging.getLogger(__name__)

class SimOnlyController(Node):
    def __init__(self):
        super().__init__("sim_only_controller")
        
        self.csv_receiver = NonBlockingJSONReceiver(port=network_config.MIA_TO_ROS2_PORT)
        self.csv_sender = NonBlockingJSONSender(port=network_config.ROS2_TO_MIA_PORT)
        self.isaacsim_receiver = NonBlockingJSONReceiver(port=network_config.ISAACSIM_TO_ROS2_PORT)
        self.isaacsim_sender = NonBlockingJSONSender(port=network_config.ROS2_TO_ISAACSIM_PORT)
        
        self.data_source = ""
        self._busy = False
        self._timer_to_clear = None
        
        # --- 狀態追蹤：防止奇怪起始位置的關鍵 ---
        self.current_IO_states = [0, 0, 1]  # 預設 Open
        self.current_joints_states = [0.0] * 6 # 初始位置
        # ---------------------------------------
        
        self._capture_timer = self.create_timer(0.05, self._capture_command)
        logger.info("✨ 模擬控制伺服器已啟動（已修正起始位置與 Timer 邏輯）")

    def _capture_command(self):
        if self._busy:
            return

        data = self.csv_receiver.capture_data()
        if data is not None:
            self.data_source = "csv"
        else:
            data = self.isaacsim_receiver.capture_data()
            if data is not None:
                self.data_source = "isaacsim"
            else:
                return

        logger.info(f"收到新指令: {data['type']}")
        self._busy = True
        self._process_data(data)

    def _process_data(self, data):
        command_lines = []
        # 預設等待時間，給予基礎的緩衝
        estimated_wait_time = data.get("wait_time", 0.5)

        if data["type"] == "arm":
            # 1. 取得目標軌跡（轉換為角度）
            target_traj = [np.rad2deg(j).tolist() for j in data["joints_values"]]
            
            # 2. 💡 解決瞬移關鍵：將「目前位置」插入到軌跡的最前面作為銜接點
            # 這樣發送給 Isaac Sim 的軌跡就會從 current -> target[0] -> target[1]...
            full_traj = [self.current_joints_states] + target_traj
            
            # 3. 更新記憶中的「最後位置」，供下一個指令使用
            self.current_joints_states = target_traj[-1]
            
            # 組合 IO 狀態
            command_lines = [j + self.current_IO_states for j in full_traj]
            
            # 根據點數動態計算等待時間，確保 busy 狀態覆蓋整個動作時間
            estimated_wait_time += len(command_lines) * 0.05 

        elif data["type"] == "gripper":
            grip_map = {
                "close": [1, 0, 0],
                "open": [0, 0, 1],
                "half_open": [0, 1, 0],
                "close_tight": [1, 1, 0]
            }
            self.current_IO_states = grip_map.get(data["grip_type"], [0, 0, 1])
            command_lines = [self.current_joints_states + self.current_IO_states]
            estimated_wait_time = 1.0

        self._execute_send_traj(command_lines, estimated_wait_time)

    def _execute_send_traj(self, command_lines, wait_time):
        try:
            if command_lines:
                send_traj(command_lines)
            
            self._timer_to_clear = self.create_timer(wait_time, self._timer_callback)
            
        except Exception as e:
            logger.error(f"發送失敗: {e}")
            self._handle_completion(is_success=False)

    def _timer_callback(self):
        if self._timer_to_clear is not None:
            self._timer_to_clear.cancel()
            self.destroy_timer(self._timer_to_clear)
            self._timer_to_clear = None
        self._handle_completion(is_success=True)

    def _handle_completion(self, is_success=True):
        sender = self.csv_sender if self.data_source == "csv" else self.isaacsim_sender
        sender.send_data({"message": "Success" if is_success else "Fail"})
        self._busy = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, handlers=[handler], force=True)
    rclpy.init()
    node = SimOnlyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()