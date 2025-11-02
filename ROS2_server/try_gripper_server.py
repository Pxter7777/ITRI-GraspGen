from common_utils.socket_communication import NonBlockingJSONSender

receiver = NonBlockingJSONSender(port=9876)

data = {"type": "arm", "wait_time": 0.0,
    "joints_values": [
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.1],
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.2],
        [0.0, -0.3, 2.4, 1.0, -1.5, 3.3],
    ]
}

receiver.send_data(data)