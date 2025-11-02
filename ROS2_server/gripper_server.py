# execute it with /usr/bin/python3

# one_arm_control_CPP.py

import time
import math
import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript, SetIO
from tm_msgs.msg import FeedbackState
from geometry_msgs.msg import PoseStamped
from collections import deque
import json
import os
import argparse
import logging
from socket_communication import NonBlockingJSONReceiver

import numpy as np

logger = logging.getLogger(__name__)


def quat_to_euler_zyx_deg(qx, qy, qz, qw):
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def normalize_quat(x, y, z, w):
        n = math.sqrt(x * x + y * y + z * z + w * w)
        return (0.0, 0.0, 0.0, 1.0) if n == 0 else (x / n, y / n, z / n, w / n)

    qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
    # yaw (Z)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    # pitch (Y)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = _clamp(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)
    # roll (X)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)  # rx, ry, rz


def is_two_point_identical(point1: list, point2: list):
    pos_identical = all(
        abs(p1 - p2) < 10 for p1, p2 in zip(point1[:3], point2[:3], strict=False)
    )
    orient_identical = all(
        abs(o1 - o2) < 2 for o1, o2 in zip(point1[3:], point2[3:], strict=False)
    )
    return pos_identical and orient_identical

def is_pose_identical(joints1: list, joints2: list):
    pos_identical = all(
        abs(j1 - j2) < 0.1 for j1, j2 in zip(joints1, joints2, strict=False)
    )
    return pos_identical

class TMRobotController(Node):
    def __init__(self):
        super().__init__("tm_robot_controller")
        self.receiver = NonBlockingJSONReceiver(port=9876)
        self.script_cli = None
        self.io_cli = None
        self.tcp_queue = deque()
        self._busy = False
        self._min_send_interval = 0.20
        self._last_send_ts = 0.0

        self.no_retry_on_fail = True
        self.clear_queue_on_fail = False
        self._queue_timer = self.create_timer(0.05, self._process_queue)
        self._capture_timer = self.create_timer(0.05, self._capture_command)
        self.gripper_poll_sec = 0.10

        self.states_need_to_wait = []
        self.moving = False
        self.wait_time = 0
        self.reached_time = 0

    def _capture_command(self):
        if self.moving:
            return

        data = self.receiver.capture_data()
        if data is None:
            return
        print(data)
        self.moving = True
        self.wait_time = data["wait_time"]
        if data["type"] == "arm":
            joints_values = data["joints_values"]
            self.goal_joints = joints_values[-1]
            for joints in joints_values:
                joints = np.rad2deg(joints)
                self.append_jpp(joints)

    def setup_services(self):
        self.get_logger().info("等待 ROS 2 服務啟動...")

        self.script_cli = self.create_client(SendScript, "send_script")
        while not self.script_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待 send_script 服務...")

        self.io_cli = self.create_client(SetIO, "set_io")
        while not self.io_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("等待 set_io 服務...")

        # 初始化 gripper 狀態追蹤
        self.ee_digital_output = [0, 0, 1, 0]  # 初始狀態
        self.target_ee_output = None  # 要等待的目標狀態
        self.waiting_for_gripper = False  # 是否等待中

        # 訂閱 feedback_states
        self.create_subscription(
            FeedbackState, "feedback_states", self.feedback_callback, 10
        )
        self.get_logger().info("✅ 已訂閱 feedback_states")
        self.sub = self.create_subscription(PoseStamped, "/tool_pose", self.cb, 10)
        self.get_logger().info(" subscribe /tool_pose")

    def feedback_callback(self, msg: FeedbackState):
        current_time = time.time()
        self.ee_digital_output = list(msg.ee_digital_output)

        if self.waiting_for_gripper and self.target_ee_output is not None:
            if self.ee_digital_output[:3] == self.target_ee_output:
                self.get_logger().info(
                    f"🔄 夾爪狀態達成: {self.ee_digital_output}，開始等待 6 秒"
                )
                self.waiting_for_gripper = False
                self.target_ee_output = None
                self._start_gripper_wait_timer()
        if self.moving:
            if is_pose_identical(msg.joint_pos, self.goal_joints) and self.reached_time > current_time:
                self.reached_time = current_time
            if current_time - self.reached_time > self.wait_time:
                self.reached_time = float('inf')
                self.moving = False
        
    def cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        # 2)  transform to CPP（x y z rx ry rz）
        x_mm, y_mm, z_mm = p.x * 1000.0, p.y * 1000.0, p.z * 1000.0
        rx, ry, rz = quat_to_euler_zyx_deg(q.x, q.y, q.z, q.w)

        for state in self.states_need_to_wait:
            if is_two_point_identical(
                [x_mm, y_mm, z_mm, rx, ry, rz], state["position"]
            ):
                self.get_logger().info(
                    f"🔄 夾爪狀態達成: {self.ee_digital_output}，開始等待 87 秒"
                )
                self._start_arm_wait_timer(state["time_to_wait"])
                self.states_need_to_wait.remove(state)

    def _start_gripper_wait_timer(self):
        # 建立 Timer，並在執行 callback 時自行取消
        self._wait_timer = self.create_timer(1.0, self._gripper_wait_done)

    def _gripper_wait_done(self):
        self.get_logger().info("✅ 夾爪動作等待完成")
        self._busy = False

        if hasattr(self, "_wait_timer"):
            self._wait_timer.cancel()
            del self._wait_timer

    def _start_arm_wait_timer(self, time):
        # 建立 Timer，並在執行 callback 時自行取消
        self._wait_timer_arm = self.create_timer(time, self._arm_wait_done)

    def _arm_wait_done(self):
        self.get_logger().info("✅ 夾爪動作等待完成")
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
                        self.get_logger().info(
                            f"✅ End_DO{pin} 設定成功，等待 feedback 確認"
                        )
                        # 只設定一次 target 狀態即可
                        if pin == 2:  # 最後一個 pin 設定完成時
                            self.target_ee_output = states
                            self.waiting_for_gripper = True
                    else:
                        self.get_logger().warn(f"⚠️ End_DO{pin} 設定失敗，略過等待")
                        self._busy = False
                except Exception as e:
                    self.get_logger().error(f"[SetIO 失敗] {e}")
                    self._busy = False

            future.add_done_callback(_done)

    def append_gripper_states(self, states):
        if not (isinstance(states, (list, tuple)) and len(states) == 3):
            self.get_logger().error("IO 狀態必須為長度 3 的 list，例如 [1,0,0]")
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
            self.get_logger().error("TCP 必須 6 個數字")
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
    def append_jpp(
        self, joint_values: list, vel=20, acc=20, coord=80, fine=False, wait_time=0.0
    ):
        if len(joint_values) != 6:
            self.get_logger().error("TCP 必須 6 個數字")
            return
        fine_str = "true" if fine else "false"
        script = f'PTP("JPP",{joint_values[0]:.2f}, {joint_values[1]:.2f}, {joint_values[2]:.2f}, ' \
                 f'{joint_values[3]:.2f}, {joint_values[4]:.2f}, {joint_values[5]:.2f},' \
                 f'20,20,200,true)'
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
        self._busy = True

        # IO 指令
        if isinstance(cmd, str) and cmd.startswith("IO:"):
            self.get_logger().info(f"執行夾爪指令: {cmd}")
            try:
                _, vals = cmd.split(":")
                a, b, c = map(int, vals.split(","))
                self.set_io([a, b, c])
            except Exception as e:
                self.get_logger().error(f"IO 解析錯誤: {e}")
                self._busy = False
            return

        # 動作指令
        script_to_run = cmd
        self.get_logger().info(
            f"正在執行佇列中的腳本: {script_to_run} wait_time={wait_time}"
        )
        self._send_script_async(script_to_run, wait_time)

    def _send_script_async(self, script: str, wait_time):
        if not self.script_cli:
            self.get_logger().error("send_script 客戶端尚未初始化。")
            self._busy = False
            return
        req = SendScript.Request()
        req.id = "auto"
        req.script = script
        self.get_logger().info("ready to call async")
        future = self.script_cli.call_async(req)
        self.get_logger().info("called async")

        def _done(_):
            try:
                res = future.result()
                ok = bool(getattr(res, "ok", False))
                if ok:
                    self.get_logger().info("✅ successed")
                else:
                    self.get_logger().warn("⚠️ 執行失敗：跳過該指令")
            except Exception as e:
                self.get_logger().error(f"[SendScript 失敗] {e}")
            finally:
                if wait_time == 0:
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
        node.receiver.disconnect()


if __name__ == "__main__":
    main()
