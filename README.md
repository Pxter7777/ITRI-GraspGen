# ITRI-GraspGen
[Installation Guide](./Installation_Guide.md)

## Abstract
This project aims to deploy and optimize the process of open-loop grasping

Open-loop means the whole process will divide down into parts, for example:
- 2D images to 3D pointcloud
- Grasp Estimating (Generate a grasp pose)
- Motion Planning (Find a way to approach a pose or joint state, by moving the joints)

## Execution Steps
Since each part needs different python environments(GraspGen, IsaacSim, ROS2), currently they are started as separated process, and communicate using my custom module common_utils/socket_communication.py

0. Run tm_driver
    ```bash
    ros2 run tm_driver tm_driver robot_ip:=192.168.1.10
    ```
    - Only run this if we're running on actual robot arm.
1. Run ROS_server
    - If we're running on actual robot arm.
        ```bash
        cd ~/ITRI-GraspGen/ROS2_server && \
        /usr/bin/python3 gripper_server.py
        ```
    - If not, run this dummy server instead:
        ```bash
        cd ~/ITRI-GraspGen/ROS2_server && \
        uv run dummy_gripper_server.py
        ```
2. Run Isaac Sim + cuRobo
    ```bash
    cd ~/ITRI-GraspGen/isaac-sim2real && \
    omni_python sync_with_ROS2.py
    ```
3. Run GraspGen script
    - If camera attached.
        ```bash
        cd ~/ITRI-GraspGen && \
        uv run scripts/workflow_with_isaacsim.py  --no-confirm
        ```
    - If not, use the images we captured.
        ```bash
        cd ~/ITRI-GraspGen && \
        uv run scripts/workflow_with_isaacsim.py  --no-confirm --use-png demo6
        ```

## General Development Reminder
- Do not push anything onto main branch. main branch only allows GitHub pull requests' merge and squash commit.
- Run these before pushing, make sure ruff doesn't complain.
```
uv run ruff check --fix
uv run ruff format
```

## Reminder for AI Agents
- IMPORTANT! Do not run any .py scripts. For now, each script is heavy and messy, I prefer to test things and terminate those process by myself.
- Third_party/ contains submodules from other repo. Do not try to modify them.
- This project currently is messy and contains many needless scripts, I know, but don't try to reconstruct, just try to focus on the urgent tasks for now.


## Credits
Full license details for these dependencies are available in the `NOTICE.md` file.

*   **GraspGen:** [https://github.com/NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
*   **Segment Anything Model 2 (SAM2):** [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
*   **FoundationStereo:** [https://github.com/NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo)
*   **GroundingDINO:** [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

