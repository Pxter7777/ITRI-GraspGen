# one_arm_control_CPP.py
import time
import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript, SetIO  # ★ add
from collections import deque
import json
import os
import argparse


class TMRobotController(Node):
    def __init__(self):
        super().__init__("tm_robot_controller")
        self.script_cli = None
        self.io_cli = None  # ★ add
        self.tcp_queue = deque()
        self._busy = False
        self._min_send_interval = 0.20
        self._last_send_ts = 0.0

        self.no_retry_on_fail = True
        self.clear_queue_on_fail = False
        self.create_timer(0.05, self._process_queue)

    def setup_services(self):
        self.get_logger().info("等待 ROS 2 服務啟動...")
        self.script_cli = self.create_client(SendScript, "send_script")
        while not self.script_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待 send_script 服務...")

        self.io_cli = self.create_client(SetIO, "set_io")  # ★ add
        while not self.io_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待 set_io 服務...")

    def set_io(self, states: list):
        """設定 End_DI0, End_DI1, End_DI2 狀態，例如 [1, 0, 0]"""
        for pin, state in enumerate(states):
            req = SetIO.Request()
            req.module = 1  # End Module 夾爪
            req.type = 1  # Digital Input
            req.pin = pin
            req.state = float(state)
            future = self.io_cli.call_async(req)
            """ while rclpy.ok():
                rclpy.spin_once(self)
                if future.done():
                    msg = f"✅ End_DI{pin} 設定為 {state}" if future.result().ok else f"⚠️ End_DI{pin} 設定失敗"
                    self.get_logger().info(msg)
                    break """

            def _done(fut, pin=pin, state=state):
                try:
                    result = fut.result()
                    if result.ok:
                        self.get_logger().info(f"✅ End_DI{pin} 設定為 {state}")
                    else:
                        self.get_logger().warn(f"⚠️ End_DI{pin} 設定失敗")
                except Exception as e:
                    self.get_logger().error(f"[SetIO 失敗] {e}")
                finally:
                    self._busy = False

            future.add_done_callback(_done)

    # ★ 加上 append_gripper_* 系列，用 "IO:1,0,0" 字串推進 queue
    def append_gripper_states(self, states):
        if not (isinstance(states, (list, tuple)) and len(states) == 3):
            self.get_logger().error("IO 狀態必須為長度 3 的 list，例如 [1,0,0]")
            return
        self.tcp_queue.append(f"IO:{states[0]},{states[1]},{states[2]}")

    def append_gripper_close(self):
        self.append_gripper_states([1, 0, 0])

    def append_gripper_half(self):
        self.append_gripper_states([0, 1, 0])

    def append_gripper_open(self):
        self.append_gripper_states([0, 0, 1])

    def append_tcp(self, tcp_values: list, vel=20, acc=20, coord=80, fine=False):
        if len(tcp_values) != 6:
            self.get_logger().error("TCP 必須 6 個數字")
            return
        fine_str = "true" if fine else "false"
        script = (
            f'PTP("CPP",{tcp_values[0]:.2f}, {tcp_values[1]:.2f}, {tcp_values[2]:.2f}, '
            f"{tcp_values[3]:.2f}, {tcp_values[4]:.2f}, {tcp_values[5]:.2f},"
            f"{vel},{acc},{coord},{fine_str})"
        )
        self.tcp_queue.append(script)

    def _process_queue(self):
        if self._busy or not self.tcp_queue:
            return
        now = time.time()
        if now - self._last_send_ts < self._min_send_interval:
            return

        cmd = self.tcp_queue.popleft()
        self._last_send_ts = now
        self._busy = True

        # ★ 檢查是否是 IO 指令
        if isinstance(cmd, str) and cmd.startswith("IO:"):
            self.get_logger().info(f"執行夾爪指令: {cmd}")
            try:
                _, vals = cmd.split(":")
                a, b, c = map(int, vals.split(","))
                self.set_io([a, b, c])  # ← 呼叫你原版的 set_io()
            except Exception as e:
                self.get_logger().error(f"IO 解析錯誤: {e}")
            # self._busy = False
            return

        # ★ 否則照原流程送動作
        script_to_run = cmd
        self.get_logger().info(f"正在執行佇列中的腳本: {script_to_run}")
        self._send_script_async(script_to_run)

    def _send_script_async(self, script: str):
        if not self.script_cli:
            self.get_logger().error("send_script 客戶端尚未初始化。")
            self._busy = False
            return
        req = SendScript.Request()
        req.id = "auto"
        req.script = script
        future = self.script_cli.call_async(req)

        def _done(_):
            try:
                res = future.result()
                ok = bool(getattr(res, "ok", False))
                if ok:
                    self.get_logger().info("✅ 執行成功")
                else:
                    self.get_logger().warn("⚠️ 執行失敗：跳過該指令")
            except Exception as e:
                self.get_logger().error(f"[SendScript 失敗] {e}")
            finally:
                self._busy = False

        future.add_done_callback(_done)

    def clear_queue(self):
        n = len(self.tcp_queue)
        self.tcp_queue.clear()
        self.get_logger().info(f"已清空佇列，共 {n} 筆")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a scene point cloud after GraspGen inference, for entire scene"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Directory containing JSON files with point cloud data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.input, "HEEE")
    json_file = os.path.join("output", args.input)
    with open(json_file, "rb") as f:
        data = json.load(f)
    print("data", data)
    position = data["position"]

    position = [i * 1000 for i in position]
    print("positiion", position)
    euler_orientation = data["euler_orientation"]
    signal = position + euler_orientation
    print("signal", signal)
    forward = data["forward_vec"]
    first_position = [p - 100 * f for p, f in zip(position, forward, strict=False)]
    second_position = position
    third_position = [p + 65 * f for p, f in zip(position, forward, strict=False)]

    first_signal = first_position + euler_orientation
    second_signal = second_position + euler_orientation
    third_signal = third_position + euler_orientation

    fourth_position = third_position
    fourth_position[2] = third_position[2] + 70
    fourth_orientation = euler_orientation

    fifth_position = [422.4, -67, fourth_position[2]]
    fifth_orientation = [93.4, 1.4, 96.4]
    sixth_position = [451.6, -119.8, fourth_position[2]]
    sixth_orientation = [-105.1, -67.1, -72.3]

    fourth_signal = fourth_position + fourth_orientation
    fifth_signal = fifth_position + fifth_orientation
    sixth_signal = sixth_position + sixth_orientation

    home_position = [312.7, -148.5, 403.9]
    home_orientation = [92.9, 0.0, 90]
    home_signal = home_position + home_orientation
    rclpy.init()
    node = TMRobotController()
    try:
        node.setup_services()

        # node.append_tcp([500.00, 300.00, 100.00, 90.00, 0.00, 90.00])
        # node.append_tcp([523.00, 193.00, 148.00, 101.00, 1.85, -27.8])
        node.append_tcp(first_signal)
        node.append_tcp(second_signal)
        node.append_tcp(third_signal)
        node.append_gripper_close()
        # time.sleep(1)
        node.append_tcp(fourth_signal)
        node.append_tcp(fifth_signal)
        node.append_tcp(sixth_signal)
        # time.sleep(5)
        node.append_tcp(fifth_signal)
        node.append_tcp(fourth_signal)
        # time.sleep(1)
        node.append_gripper_open()
        node.append_tcp(third_signal)
        node.append_tcp(second_signal)
        node.append_tcp(first_signal)
        node.append_tcp(home_signal)
        # node.append_tcp([367.05, -140.73, 258.21, 91.85, 8.72, 73.71])
        # node.append_gripper_open()

        rclpy.spin(node)
    except KeyboardInterrupt:
        print("中斷程式")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
