from common_utils.socket_communication import (
    NonBlockingJSONSender,
    BlockingJSONReceiver,
)
from common_utils import port_config
import time

def scenario2():
    sender = NonBlockingJSONSender(port=port_config.GRASPGEN_TO_ISAACSIM)
    receiever = BlockingJSONReceiver(port=port_config.ISAACSIM_TO_GRASPGEN)
    joints_goal1 = [1.37296326,  0.38553859,  1.05554023,  2.76803983, -1.48792809, 3.09947786]
    joints_goal2 = [1.37296326,  0.58553859,  1.05554023,  2.76803983, -1.48792809, 3.09947786]
    full_act1 = {"moves": [{"type": "arm", "joints_goal": joints_goal1, "wait_time": 0.0}], "obstacles":[]}
    full_act2 = {"moves": [{"type": "arm", "joints_goal": joints_goal2, "wait_time": 0.0}], "obstacles":[]}
    sender.send_data([full_act1])
    time.sleep(3)
    sender.send_data([full_act2])
    time.sleep(3)
    sender.send_data([full_act1])
    time.sleep(3)
def scenario1():
    sender = NonBlockingJSONSender(port=port_config.GRASPGEN_TO_ISAACSIM)
    receiever = BlockingJSONReceiver(port=port_config.ISAACSIM_TO_GRASPGEN)
    joints_goal1 = [1.37296326,  0.38553859,  1.05554023,  2.76803983, -1.48792809, 3.09947786]
    joints_goal2 = [1.37296326,  0.58553859,  1.05554023,  2.76803983, -1.48792809, 3.09947786]
    full_act1 = {"moves": [{"type": "arm", "joints_goal": joints_goal1, "wait_time": 0.0}], "obstacles":[]}
    full_act2 = {"moves": [{"type": "arm", "joints_goal": joints_goal2, "wait_time": 0.0}], "obstacles":[]}
    sender.send_data([full_act1])
    time.sleep(1)
    sender.send_data([full_act2])
    time.sleep(1)
    sender.send_data([full_act1])
    time.sleep(1)

"""
Currently, scenario1 can work normally, however, I can see scenario2 is buggy, and that's because the bad logic currently in sync_with_ROS2.py.
If ROS2 is complete, and there is no moves in queue, the robot will reset its last_joint_states back to default, which caused this weird result.
This usually wouldn't happen, but theoritically and practically can happen sometimes.
"""

if __name__ == "__main__":
    scenario2()