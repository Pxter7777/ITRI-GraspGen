# send one single full act like scripts/workflow_with_isaacsim.py does
# This is deprecated due to I already removed data_for_test


import json
import logging
from pathlib import Path
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    BlockingJSONReceiver,
)

PROJECT_ROOT_DIR = PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)

sender = NonBlockingJSONSender(port=9878)
receiever = BlockingJSONReceiver(port=9879)
try:
    filepath = (
        PROJECT_ROOT_DIR / "data_for_test" / "fullact" / "fullact_20251201_210042.json"
    )
    with open(filepath, "rb") as f:
        data1 = json.load(f)

    filepath = (
        PROJECT_ROOT_DIR / "data_for_test" / "fullact" / "fullact_20251201_210050.json"
    )
    with open(filepath, "rb") as f:
        data2 = json.load(f)

    filepath = (
        PROJECT_ROOT_DIR / "data_for_test" / "fullact" / "fullact20251201_210052.json"
    )
    with open(filepath, "rb") as f:
        data3 = json.load(f)

    data4 = ["EOF"]

    datas = [data1, data2, data3, data4]

    for data in datas:
        sender.send_data(data)
        message = receiever.capture_data()
        print(message)
finally:
    sender.disconnect()
    receiever.disconnect()
