# execute it with /usr/bin/python3

# one_arm_control_CPP.py

import time
import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript, SetIO
from tm_msgs.msg import FeedbackState
from geometry_msgs.msg import PoseStamped
from collections import deque
import numpy as np
import argparse
import logging
import os
import sys

# Because this script isn't using itri-graspgen venv, we need to manually add the project root to sys.path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
from common_utils import port_config  # noqa: E402
from common_utils.socket_communication import (  # noqa: E402
    NonBlockingJSONReceiver,
    NonBlockingJSONSender,
)
from common_utils.custom_logger import CustomFormatter  # noqa: E402


logger = logging.getLogger(__name__)

def mrad_to_mmdeg(cartesian_pose:list) -> list:
    position = [p*1000 for p in cartesian_pose[:3]]
    euler_orientation_deg = np.rad2deg(cartesian_pose[3:]).tolist()
    return position + euler_orientation_deg

def is_joint_vel_near_zero(joint_vel: list):
    return all(
        abs(v) < 0.001 for v in joint_vel
    )

def is_cartesion_pose_similar(point1: list, point2: list): # use euler angle, not quaternion
    if point1 is None or point2 is None:
        return False
    pos_similar = all(
        abs(p1 - p2) < 20 for p1, p2 in zip(point1[:3], point2[:3], strict=False)
    )
    orient_identical = all(
        abs(o1 - o2) < 4 for o1, o2 in zip(point1[3:], point2[3:], strict=False)
    )
    return pos_similar and orient_identical

def is_cartesion_pose_identical(point1: list, point2: list): # use euler angle, not quaternion
    if point1 is None or point2 is None:
        return False
    pos_identical = all(
        abs(p1 - p2) < 10 for p1, p2 in zip(point1[:3], point2[:3], strict=False)
    )
    orient_identical = all(
        abs(o1 - o2) < 2 for o1, o2 in zip(point1[3:], point2[3:], strict=False)
    )
    return pos_identical and orient_identical


def is_pose_identical(joints1: list, joints2: list):
    if joints1 is None or joints2 is None:
        return False
    pos_identical = all(
        abs(j1 - j2) < 0.02 for j1, j2 in zip(joints1, joints2, strict=False)
    )
    return pos_identical


successs = 0

class TMRobotController(Node):
    def __init__(self):
        super().__init__("tm_robot_controller")
        self.csv_receiver = NonBlockingJSONReceiver(port=port_config.MIA_TO_ROS2)
        self.csv_sender = NonBlockingJSONSender(port=port_config.ROS2_TO_MIA)
        self.isaacsim_receiver = NonBlockingJSONReceiver(
            port=port_config.ISAACSIM_TO_ROS2
        )
        self.isaacsim_sender = NonBlockingJSONSender(port=port_config.ROS2_TO_ISAACSIM)
        self.data_source = ""
        self.script_cli = None
        self.io_cli = None
        self.tcp_queue = deque()
        self._busy = False
        self._min_send_interval = 0.00
        self._last_send_ts = 0.0

        self.no_retry_on_fail = True
        self.clear_queue_on_fail = False
        self._queue_timer = self.create_timer(0.005, self._process_queue)
        self._capture_timer = self.create_timer(0.05, self._capture_command)
        self.gripper_poll_sec = 0.10

        self.states_need_to_wait = []
        self.moving = False
        self.wait_time = 0
        self.current_moving_type = ""
        self.reached_time = float("inf")
        self.goal_gripper = None
        # record the last LAST_JOINTS_REC_NUM joint positions to detect stuck, using queue
        self.num_response_to_send_back = (
            0  # it's enough because we're not using multi-threading
        )

    def _capture_command(self):
        if self.moving:
            return
        # try capture csv first
        csv_data = self.csv_receiver.capture_data()
        if csv_data is not None:
            data = csv_data
            self.data_source = "csv"
        else:
            isaacsim_data = self.isaacsim_receiver.capture_data()
            if isaacsim_data is not None:
                data = isaacsim_data
                self.data_source = "isaacsim"
            else:
                return  # no data
        logger.info("received new data")
        logger.debug(data)
        self.num_response_to_send_back += 1
        logger.debug(f"{self.num_response_to_send_back}")
        self.reached_time = float("inf")
        self.stuck_start_time = float("inf")
        self.moving = True
        self.wait_time = data["wait_time"]
        self.current_moving_type = data["type"]
        if data["type"] == "arm":
            self.goal_joints = data["joints_values"][-1]
            joints_values_degree = [
                np.rad2deg(joints) for joints in data["joints_values"]
            ]
            goal_degree = joints_values_degree.pop()  # remove the goal
            accepted_joints = []
            for joints in joints_values_degree:
                if len(accepted_joints) == 0:
                    accepted_joints.append(joints)
                    continue
                if any(
                    abs(j1 - j2) > 4
                    for j1, j2 in zip(joints, accepted_joints[-1], strict=False)
                ):
                    accepted_joints.append(joints)
            for joints in accepted_joints:
                self.append_jpp(joints)
            self.append_jpp(goal_degree)
        elif data["type"] == "PTP":
            cartesian_poses_mm_degree = data["cartesian_poses"]
            self.goal_cartesian_pose = cartesian_poses_mm_degree.pop()
            accepted_cartesion_poses = []
            for cartesian_pose in cartesian_poses_mm_degree:
                if len(accepted_cartesion_poses)==0:
                    accepted_cartesion_poses.append(cartesian_pose)
                    continue
                if not is_cartesion_pose_similar(cartesian_pose, accepted_cartesion_poses[-1]):
                    accepted_cartesion_poses.append(cartesian_pose)
            for cartesian_pose in accepted_cartesion_poses:
                self.append_ptp(cartesian_pose)
            self.append_ptp(self.goal_cartesian_pose)
        elif data["type"] == "gripper":
            if data["grip_type"] == "close":
                self.goal_gripper = [1, 0, 0]
                self.append_gripper_states([1, 0, 0])
            elif data["grip_type"] == "open":
                self.goal_gripper = [0, 0, 1]
                self.append_gripper_states([0, 0, 1])
            elif data["grip_type"] == "half_open":
                self.goal_gripper = [0, 1, 0]
                self.append_gripper_states([0, 1, 0])
            elif data["grip_type"] == "close_tight":
                self.goal_gripper = [1, 1, 0]
                self.append_gripper_states([1, 1, 0])

    def setup_services(self):
        logger.info("等待 ROS 2 服務啟動...")

        self.script_cli = self.create_client(SendScript, "send_script")
        while not self.script_cli.wait_for_service(timeout_sec=1.0):
            logger.info("等待 send_script 服務...")

        self.io_cli = self.create_client(SetIO, "set_io")
        while not self.io_cli.wait_for_service(timeout_sec=1.0):
            logger.info("等待 set_io 服務...")

        # 初始化 gripper 狀態追蹤
        self.target_ee_output = None  # 要等待的目標狀態
        self.waiting_for_gripper = False  # 是否等待中

        # 訂閱 feedback_states
        self.create_subscription(
            FeedbackState, "feedback_states", self.feedback_callback, 10
        )
        logger.info("✅ 已訂閱 feedback_states")

    def feedback_callback(self, msg: FeedbackState) -> None:
        if not self.moving:
            return
        current_time = time.time()
        # reach detection
        if self.reached_time > current_time: # hasn't reached yet
            if self.current_moving_type == "arm" and is_pose_identical(msg.joint_pos, self.goal_joints): # Need to change this type name to JPP if possible.
                self.reached_time = current_time
            elif self.current_moving_type == "PTP" and is_cartesion_pose_identical(mrad_to_mmdeg(msg.tool_pose), self.goal_cartesian_pose):
                self.reached_time = current_time
            elif self.current_moving_type == "gripper" and list(msg.ee_digital_output)[:3] == self.goal_gripper:
                self.reached_time = current_time
        
        # stuck detection
        if (
            is_joint_vel_near_zero(list(msg.joint_vel))
            and self.reached_time > current_time
        ): # stuck detected
            if self.stuck_start_time > current_time: # it's a new stuck
                logger.info("new stuck detected, start timing stuck.")
                self.stuck_start_time = current_time
        else:
            self.stuck_start_time = float("inf")
        
        # handle success reach
        if current_time - self.reached_time >= self.wait_time:
            self._handle_success()
        # handle stuck failure
        if current_time - self.stuck_start_time >= 3:
            logger.debug(f"{current_time}, {self.stuck_start_time}")
            logger.error("Stuck detected.")
            self._handle_failure()


    def _start_gripper_wait_timer(self):
        # 建立 Timer，並在執行 callback 時自行取消
        self._wait_timer = self.create_timer(1.0, self._gripper_wait_done)

    def _gripper_wait_done(self):
        logger.info("✅ 夾爪動作等待完成")
        self._busy = False

        if hasattr(self, "_wait_timer"):
            self._wait_timer.cancel()
            del self._wait_timer

    def _start_arm_wait_timer(self, time):
        # 建立 Timer，並在執行 callback 時自行取消
        self._wait_timer_arm = self.create_timer(time, self._arm_wait_done)

    def _arm_wait_done(self):
        logger.info("✅ 夾爪動作等待完成")
        self._busy = False

        if hasattr(self, "_wait_timer_arm"):
            self._wait_timer_arm.cancel()
            del self._wait_timer_arm

    def set_io(self, states: list):
        """設定 End_DO0, End_DO1, End_DO2 狀態，例如 [1, 0, 0]"""
        for pin, state in enumerate(states):
            req = SetIO.Request()
            req.module = 1  # End Module 夾爪
            req.type = 1  # Digital Output
            req.pin = pin
            req.state = float(state)

            future = self.io_cli.call_async(req)

            def _done(fut, pin=pin):
                try:
                    result = fut.result()
                    if result.ok:
                        logger.info(f"✅ End_DO{pin} 設定成功，等待 feedback 確認")
                        # 只設定一次 target 狀態即可
                        if pin == 2:  # 最後一個 pin 設定完成時
                            self.target_ee_output = states
                            self.waiting_for_gripper = True
                    else:
                        logger.warn(f"⚠️ End_DO{pin} 設定失敗，略過等待")
                        self._busy = False
                except Exception as e:
                    logger.error(f"[SetIO 失敗] {e}")
                    self._busy = False

            future.add_done_callback(_done)

    def append_gripper_states(self, states):
        if not (isinstance(states, (list, tuple)) and len(states) == 3):
            logger.error("IO 狀態必須為長度 3 的 list，例如 [1,0,0]")
            return
        self.tcp_queue.append(
            {"script": f"IO:{states[0]},{states[1]},{states[2]}", "wait_time": 0.0}
        )

    def append_gripper_close(self):
        self.append_gripper_states([1, 0, 0])

    def append_gripper_half(self):
        self.append_gripper_states([0, 1, 0])

    def append_gripper_open(self):
        self.append_gripper_states([0, 0, 1])

    def append_tcp(
        self, tcp_values: list, vel=20, acc=20, coord=80, fine=False, wait_time=0.0
    ):
        if len(tcp_values) != 6:
            logger.error("TCP 必須 6 個數字")
            return
        fine_str = "true" if fine else "false"
        script = (
            f'PTP("CPP",{tcp_values[0]:.2f}, {tcp_values[1]:.2f}, {tcp_values[2]:.2f}, '
            f"{tcp_values[3]:.2f}, {tcp_values[4]:.2f}, {tcp_values[5]:.2f},"
            f"{vel},{acc},{coord},{fine_str})"
        )
        self.tcp_queue.append({"script": script, "wait_time": wait_time})
        if wait_time > 0:
            self.states_need_to_wait.append(
                {"position": tcp_values, "time_to_wait": wait_time}
            )
    def append_ptp(
        self, ptp_values: list, vel=20, acc=20, coord=80, fine=False, wait_time=0.0
    ):
        if len(ptp_values) != 6:
            logger.error("TCP 必須 6 個數字")
            return
        fine_str = "true" if fine else "false"
        script = (
            f'PTP("CPP",{ptp_values[0]:.2f}, {ptp_values[1]:.2f}, {ptp_values[2]:.2f}, '
            f"{ptp_values[3]:.2f}, {ptp_values[4]:.2f}, {ptp_values[5]:.2f},"
            f"{vel},{acc},{coord},{fine_str})"
        )
        self.tcp_queue.append({"script": script, "wait_time": wait_time})
        if wait_time > 0:
            self.states_need_to_wait.append(
                {"position": ptp_values, "time_to_wait": wait_time}
            )

    def append_jpp(
        self, joint_values: list, vel=20, acc=20, coord=80, fine=False, wait_time=0.0
    ):
        if len(joint_values) != 6:
            logger.error("TCP 必須 6 個數字")
            return
        # fine_str = "true" if fine else "false"
        script = (
            f'PTP("JPP",{joint_values[0]:.2f}, {joint_values[1]:.2f}, {joint_values[2]:.2f}, '
            f"{joint_values[3]:.2f}, {joint_values[4]:.2f}, {joint_values[5]:.2f},"
            f"40,20,100,true)"
        )
        self.tcp_queue.append({"script": script, "wait_time": wait_time})
        if wait_time > 0:
            self.states_need_to_wait.append(
                {"position": joint_values, "time_to_wait": wait_time}
            )

    def _process_queue(self):
        if self._busy:
            return
        if not self.tcp_queue:
            return
        now = time.time()
        if now - self._last_send_ts < self._min_send_interval:
            return
        item = self.tcp_queue.popleft()
        cmd, wait_time = item["script"], item["wait_time"]
        self._last_send_ts = now
        # self._busy = True

        # IO 指令
        if isinstance(cmd, str) and cmd.startswith("IO:"):
            logger.info(f"執行夾爪指令: {cmd}")
            try:
                _, vals = cmd.split(":")
                a, b, c = map(int, vals.split(","))
                self.set_io([a, b, c])
            except Exception as e:
                logger.error(f"IO 解析錯誤: {e}")
                self._busy = False
            return

        # 動作指令
        script_to_run = cmd
        logger.info(f"正在執行佇列中的腳本: {script_to_run} wait_time={wait_time}")
        self._send_script_async(script_to_run, wait_time)

    def _send_script_async(self, script: str, wait_time):
        if not self.script_cli:
            logger.error("send_script 客戶端尚未初始化。")
            self._busy = False
            return
        req = SendScript.Request()
        req.id = "auto"
        req.script = script
        logger.info("ready to call async")
        future = self.script_cli.call_async(req)
        logger.info("called async")

        def _done(_):
            try:
                res = future.result()
                ok = bool(getattr(res, "ok", False))
                if ok:
                    logger.info("✅ successed")
                else:
                    logger.error("⚠️ failed to send script.")
                    self._handle_failure()
            except Exception as e:
                logger.error(f"[SendScript 失敗] {e}")
            finally:
                if wait_time == 0:
                    self._busy = False

        future.add_done_callback(_done)

    def clear_queue(self):
        n = len(self.tcp_queue)
        self.tcp_queue.clear()
        logger.info(f"已清空佇列，共 {n} 筆")

    def _handle_failure(self):
        if self.num_response_to_send_back == 0:
            return
        self.num_response_to_send_back -= 1
        logger.error("clearing queue.")
        self.clear_queue()
        logger.error(f"Acknowledging failure. {self.num_response_to_send_back}")
        sender = self.csv_sender if self.data_source == "csv" else self.isaacsim_sender
        sender.send_data({"message": "Fail"})
        self.moving = False

    def _handle_success(self):
        if self.num_response_to_send_back == 0:
            return
        self.num_response_to_send_back -= 1
        logger.info(
            f"Acknowledging movement completion. {self.num_response_to_send_back}"
        )
        sender = self.csv_sender if self.data_source == "csv" else self.isaacsim_sender
        sender.send_data({"message": "Success"})
        global successs
        successs += 1
        logger.debug(f"Success count: {successs}")
        self.moving = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a scene point cloud after IsaacSim inference, for entire scene"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Directory containing JSON files with point cloud data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show debug info",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
    else:
        logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)

    rclpy.init()
    node = TMRobotController()

    try:
        node.setup_services()

        # node.append_tcp([500.00, 300.00, 100.00, 90.00, 0.00, 90.00])
        # node.append_tcp([523.00, 193.00, 148.00, 101.00, 1.85, -27.8])
        # for move in data:
        #     if move["type"] == "move_arm":
        #         node.append_tcp(move["goal"], wait_time=move["wait_time"])
        #     elif move["type"] == "gripper" and move["goal"] == "grab":
        #         node.append_gripper_close()
        #     elif move["type"] == "gripper" and move["goal"] == "release":
        #         node.append_gripper_open()
        # node.append_tcp([367.05, -140.73, 258.21, 91.85, 8.72, 73.71])
        # node.append_gripper_open()

        rclpy.spin(node)
    except KeyboardInterrupt:
        print("中斷程式")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.csv_sender.disconnect()
        node.isaacsim_sender.disconnect()
        node.isaacsim_receiver.disconnect()
        node.csv_receiver.disconnect()


if __name__ == "__main__":
    main()
