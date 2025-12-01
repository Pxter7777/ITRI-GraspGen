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
    "wait_time": 2.0,
    "joints_values": [
        [0.0, -0.3, 2.4, 1.5, -1.0, 3.5],
        [0.0, -0.3, 2.4, 1.4, -1.1, 3.4],
        [0.0, -0.3, 2.4, 1.3, -1.2, 3.3],
        [0.0, -0.3, 2.4, 1.2, -1.3, 3.2],
        [0.0, -0.3, 2.4, 1.1, -1.4, 3.1],
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.0],
    ],
}

datalist = [data1, data1, data1]
wait_for_response = False
success_count = 0
while True:
    if len(datalist) == 0:
        break
    if wait_for_response:
        response = receiver.capture_data()
        if response is None:
            continue
        # Message received
        if response["message"] == "Success":
            print("Success! Yeah!")
            print("Success count: ", success_count) # if this pops up at the moment we started the second command, it's wrong.
            wait_for_response = False
        elif response["message"] == "Fail":
            print("Failure detected. Stopping further commands.")
            break

    data = datalist.pop(0)
    sender.send_data(data)
    wait_for_response = True
