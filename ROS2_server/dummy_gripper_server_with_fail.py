# Run this python script when no real TM5S robot available.
"""
Original gripper_server's behavior:
    gripper_server accept signals from isaac sim script, run it, and responds {"message": "Success"} back to isaacsim script when the actual robot reaches the goal. (if wait_time is not 0.0, it will delay the response.)
This script aims to imitate that behavior, only that all signal will only process for a fixed time, and also delay the wait_time, before the response {"message": "Success"}
"""

import time
import logging
import os
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
from common_utils.socket_communication import (  # noqa: E402
    NonBlockingJSONReceiver,
    NonBlockingJSONSender,
)
from common_utils import port_config  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    force=True,
)
sender = NonBlockingJSONSender(port=port_config.ROS2_TO_ISAACSIM)
# if it says [socket_communication][ERROR] Connection failed. Is the bridge.py script running on localhost:9877? when started, it's normal and can be ignored.
receiver = NonBlockingJSONReceiver(port=port_config.ISAACSIM_TO_ROS2)
i = 0
while True:
    time.sleep(0.1)
    data = receiver.capture_data()
    if data is None:
        continue
    logger.info("Received data!")
    try:
        if data["type"] == "arm":
            time.sleep(2)
        time.sleep(data["wait_time"])
    except KeyError as e:
        logger.exception(
            f"{e}, key error, please make sure the signal format is correct."
        )
    i += 1
    if i % 5 == 0:
        sender.send_data({"message": "Fail"})
    else:
        sender.send_data({"message": "Success"})
