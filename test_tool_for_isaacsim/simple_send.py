# send one single full act like scripts/workflow_with_isaacsim.py does

from common_utils.socket_communication import NonBlockingJSONSender, NonBlockingJSONReceiver, BlockingJSONReceiver
import json
import os
sender = NonBlockingJSONSender(port=9878)
receiever = BlockingJSONReceiver(port=9879)
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_file_dir)
    filepath = os.path.join(project_root_dir, "data_for_test", "fullact", "green_cup_to_pan.json")
    with open(filepath, "rb") as f:
        data = json.load(f)
    sender.send_data(data)
finally:
    sender.disconnect()
    receiever.disconnect()