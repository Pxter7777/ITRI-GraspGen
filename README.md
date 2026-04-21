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
## Calibration
1. generate a pointcloud
    - `uv run main_zed.py`
    - draw a rectangle around the cup
    - hit keyboard space
    - wait till terminal is moving again
    - hit esc to exit
2. manual transform
    - `uv run manual_PC_transform.py`
    - load the pointcloud scene we just created
    - transform it.
    - when done, hit **Save Transform Config (JSON)**
    - ctrl+c to exit the process
3. copy and paste
    - config is saved as **~/ITRI-GraspGen/transform_config/scene_\<datetime\>_transform_config.json**
    - copy the content, and paste into **~/ITRI-GraspGen/PointCloud_Generation/transform_config/sim2.json**


## General Development Reminder
- Do not push anything onto main branch. main branch only allows GitHub pull requests' merge and squash commit.
- Run these before pushing, make sure ruff doesn't complain.
```
uv run ruff check --fix
uv run ruff format
```
- Make good use of logger.
    - For top level scripts, use this to get prefered color and format
        ```python
        from common_utils.custom_logger import CustomFormatter
        # root logger setup
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter())
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
        ```

## Reminder for AI Agents
- IMPORTANT! Do not run any .py scripts. For now, each script is heavy and messy, I prefer to test things and terminate those process by myself.
- Do not run git add, commit, checkout, reset, etc.... Version control should be manually handled by human users.
- Third_party/ contains submodules from other repo. Do not try to modify them.
- This project currently is messy and contains many needless scripts, I know, but don't try to reconstruct, just try to focus on the urgent tasks for now.


## Credits
Full license details for these dependencies are available in the `NOTICE.md` file.

*   **GraspGen:** [https://github.com/NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
*   **Segment Anything Model 2 (SAM2):** [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
*   **FoundationStereo:** [https://github.com/NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo)
*   **GroundingDINO:** [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

