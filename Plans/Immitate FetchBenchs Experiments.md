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
    
## Implementation Plan
### 1. Setup Fixed Scenarios via Direct Mesh Loading
We will bypass GroundingDINO/SAM2 entirely and directly load FetchBench object meshes.
*   **Target Object:** `FetchBench-CORL2024-uv/assets/benchmark_objects/Cup/[UUID]/mesh.obj`
*   **Obstacles:** Pick two other objects (e.g., a Box and a Bowl from FetchBench).
*   **Transformation:** Apply fixed 3D translations to place the Target at a reachable coordinate (e.g., `x=0.5, y=0.0, z=0.0`) and the obstacles nearby (`y=0.2` and `y=-0.2`).
*   **Direct Point Cloud Generation:** Use `trimesh.sample.sample_surface(mesh, 2048)` to sample the target's point cloud. 
*   **Obstacle Bounding Boxes:** Extract the `.bounds` of the obstacle meshes to generate the `max` and `min` coordinate dictionaries required by `isaacsim_utils`.

### 2. GraspGen Evaluator Script (`scripts/experiment_graspgen.py`)
This script replaces `workflow_with_isaacsim.py` and removes all UI (`ControlPanel` / `Meshcat`).
*   **Generate:** Pass the sampled target point cloud into `GraspGenSampler.run_inference()` to generate 200 grasps.
*   **Collision Filter:** Sample points from the obstacles to act as the "scene" and run `filter_colliding_grasps(scene_pc, grasps)` to cull visually impossible grasps.
*   **Sorting:** 
    *   *List A (Discriminator):* Retain GraspGen's default order.
    *   *List B (Heuristic):* Filter by `cup_qualifier`, then sort by `angle_offset_rad` (aligning the gripper approach vector with the object's center).
*   **Socket Comm:** Send the `scene_data` and *both* lists of grasps to Isaac Sim via a newly assigned socket port.

### 3. Isaac Sim Motion Planner (`scripts/experiment_isaacsim.py`)
A stripped-down, purely headless version of `sync_with_ROS2.py`.
*   **Load Env:** Initializes `SimulationApp(headless=True)`. Loads the `TM5S` arm, the table, and the two obstacle cuboids.
*   **Evaluate:** Loops through the grasps. For each grasp, it uses `cuRobo` to evaluate:
    1. Default State ➡️ Pre-grasp
    2. Pre-grasp ➡️ Grasp
    3. Grasp ➡️ Lift 100mm
    4. Lift ➡️ Default State
*   **Logging:** Measures the *exact time taken* to find the first fully successful motion plan using Order A vs Order B, and print a clear comparison table at the end.

    