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


def linear_interpolate(start_pose, end_pose, num_steps):
    interpolated_poses = []
    if num_steps <= 1:
        return [start_pose]
    for i in range(num_steps):
        step_pose = []
        for j in range(len(start_pose)):
            val = start_pose[j] + (i / (num_steps - 1)) * (end_pose[j] - start_pose[j])
            step_pose.append(val)
        interpolated_poses.append(step_pose)
    return interpolated_poses


start_pose_1 = data1["joints_values"][0]
end_pose_1 = data1["joints_values"][-1]
interpolated_joints_1 = linear_interpolate(start_pose_1, end_pose_1, 100)
data5 = {
    "type": "arm",
    "wait_time": 5.0,
    "joints_values": interpolated_joints_1,
}

start_pose_2 = data2["joints_values"][0]
end_pose_2 = data2["joints_values"][-1]
interpolated_joints_2 = linear_interpolate(start_pose_2, end_pose_2, 100)
data6 = {
    "type": "arm",
    "wait_time": 5.0,
    "joints_values": interpolated_joints_2,
}
data3 = {"type": "gripper", "wait_time": 2.0, "grip_type": "close"}
data4 = {"type": "gripper", "wait_time": 2.0, "grip_type": "open"}
datalist = [data5, data6, data5, data6]
# datalist = [data1, data2, data1, data2]
wait_for_response = False
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
            wait_for_response = False
        elif response["message"] == "Fail":
            print("Failure detected. Stopping further commands.")
            break

    data = datalist.pop(0)
    sender.send_data(data)
    wait_for_response = True
