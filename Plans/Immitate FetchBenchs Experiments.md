# Immitate FetchBenchs Experiments

## Goal
- Make simple experiments, to prove that if we sort the estimated grasp poses using other ways, instead of sorting using default dicriminator score(which usually only consider the target shape), we can get motion plannable grasp more ealier.

- The way can be simple, easy deterministic methods like common_utils/qualification.py cup_qualification actually can help a lot.

## Specs
Fetchbench has many detailed configs. They're too complicated, we're not doing that.

### Scene
Create and save 2 fixed scenario, each contains:
- Big Table
- TM5S Robot arm (can ignore the gripper for now)
- 3 Objects, 1 as target, 2 as obstacles.

### Scripts
We will need two scripts.
1. GraspGen Part, It would be similar to what scripts/workflow_with_isaacsim.py does, it will:
    - Load a scene
    - Generate 200 grasps for target
    - Filter grasps by collision detection module from original graspgen repo.
    - Send 2 things to second part:
        1. scene info it loaded
        2. grasp poses
    - Create the order by discriminator
    - Create the order by custom deterministic methods.
    - Accept second part's reponse, and analyze how many time will taken if we using each order before actually finding a possible motion plan
2. Isaac Sim + cuRobo part, It would be similar to what isaac-sim2real/sync_with_ROS2.py does, it will:
    - Accept the message from first part
    - Load the scene
    - Try to motion plan ALL grasp poses
        - the full move should contain:
            1. Default joint state to pre-grasp pose
            2. pre-grasp pose to grasp pose (disable the target's collision box, or else it won't be possible)
            3. move up 100 mm. (also disable target's collision box)
            4. Goes back to default joint state (also disable target's collision box)
        - log the time taken and if it's successful
    - send back to first part
    - the physics and animation can be ignored, for now.

    