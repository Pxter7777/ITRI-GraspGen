# send one single full act like scripts/workflow_with_isaacsim.py does

from common_utils.socket_communication import (
    NonBlockingJSONSender,
    BlockingJSONReceiver,
)
import json
import os
import logging

logger = logging.getLogger(__name__)

sender = NonBlockingJSONSender(port=9878)
receiever = BlockingJSONReceiver(port=9879)
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_file_dir)
    filepath = os.path.join(
        project_root_dir, "data_for_test", "fullact", "fullact_20251201_210042.json"
    )
    with open(filepath, "rb") as f:
        data1 = json.load(f)

    filepath = os.path.join(
        project_root_dir, "data_for_test", "fullact", "fullact_20251201_210050.json"
    )
    with open(filepath, "rb") as f:
        data2 = json.load(f)

    filepath = os.path.join(
        project_root_dir, "data_for_test", "fullact", "fullact20251201_210052.json"
    )
    with open(filepath, "rb") as f:
        data3 = json.load(f)

    datas = [data1, data2, data3]

    for data in datas:
        sender.send_data(data)
        message = receiever.capture_data()
        print(message)
finally:
    sender.disconnect()
    receiever.disconnect()
