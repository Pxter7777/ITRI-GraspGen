from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)

sender = NonBlockingJSONSender(port=9876)
receiver = NonBlockingJSONReceiver(port=9877)
data1 = {
    "type": "arm",
    "wait_time": 2.0,
    "joints_values": [
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.0],
        [0.0, -0.3, 2.4, 1.1, -1.4, 3.1],
        [0.0, -0.3, 2.4, 1.2, -1.3, 3.2],
        [0.0, -0.3, 2.4, 1.3, -1.2, 3.3],
        [0.0, -0.3, 2.4, 1.4, -1.1, 3.4],
        [0.0, -0.3, 2.4, 1.5, -1.0, 3.5],
    ],
}
data2 = {
    "type": "arm",
    "wait_time": 0.0,
    "joints_values": [
        [0.0, -0.3, 2.4, 1.5, -1.0, 3.5],
        [0.0, -0.3, 2.4, 1.4, -1.1, 3.4],
        [0.0, -0.3, 2.4, 1.3, -1.2, 3.3],
        [0.0, -0.3, 2.4, 1.2, -1.3, 3.2],
        [0.0, -0.3, 2.4, 1.1, -1.4, 3.1],
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.0],
    ],
}
data3 = {"type": "gripper", "wait_time": 2.0, "grip_type": "close"}
data4 = {"type": "gripper", "wait_time": 2.0, "grip_type": "open"}
datalist = [data1, data2, data3, data4, data1, data2]
wait_for_response = False
while True:
    if len(datalist) == 0:
        break
    if wait_for_response:
        response = receiver.capture_data()
        if response is not None and response["message"] == "Success":
            print("Success! Yeah!")
            wait_for_response = False
        continue
    data = datalist.pop(0)
    sender.send_data(data)
    wait_for_response = True
