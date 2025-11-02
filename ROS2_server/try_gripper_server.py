from common_utils.socket_communication import NonBlockingJSONSender
import time
sender = NonBlockingJSONSender(port=9876)

data1 = {"type": "arm", "wait_time": 2.0,
    "joints_values": [
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.0],
        [0.0, -0.3, 2.4, 1.1, -1.4, 3.1],
        [0.0, -0.3, 2.4, 1.2, -1.3, 3.2],
        [0.0, -0.3, 2.4, 1.3, -1.2, 3.3],
        [0.0, -0.3, 2.4, 1.4, -1.1, 3.4],
        [0.0, -0.3, 2.4, 1.5, -1.0, 3.5],
    ]
}
data2 = {"type": "arm", "wait_time": 0.0,
    "joints_values": [
        [0.0, -0.3, 2.4, 1.5, -1.0, 3.5],
        [0.0, -0.3, 2.4, 1.4, -1.1, 3.4],
        [0.0, -0.3, 2.4, 1.3, -1.2, 3.3],
        [0.0, -0.3, 2.4, 1.2, -1.3, 3.2],
        [0.0, -0.3, 2.4, 1.1, -1.4, 3.1],
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.0],
    ]
}

sender.send_data(data1)
sender.send_data(data2)