#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
# Don't understand how these import helps, like at all
try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")
"""

# Standard Library
import argparse
import os
import json
import time
import numpy as np
from isaacsim_utils.socket_communication import NonBlockingJSONSender, NonBlockingJSONReceiver


parser = argparse.ArgumentParser()
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--robot", type=str, default="tm5s.yml", help="robot configuration to load"
)
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)

parser.add_argument(
    "--graspfile",
    type=str,
    default="./grasps/isaac_grasp_scene_20251009_172355_transformed.yaml",
    help="grasp file to load",
)

parser.add_argument(
    "--pot",
    type=str,
    default="./scene/1027_pot.usdz",
    help="object file to load",
)

parser.add_argument(
    "--table",
    type=str,
    default="./scene/1029_table.usdz",
    help="object file to load",
)

parser.add_argument(
    "--bluecup",
    type=str,
    default="./scene/1027_Blue_thermos_cup.usdz",
    help="object file to load",
)

parser.add_argument(
    "--greencup",
    type=str,
    default="./scene/1029_Green_thermos_cup.usdz",
    help="object file to load",
)
"""
parser.add_argument(
    "--object",
    type=str,
    default="./grasps/object_20250926_162723.usd",
    help="object file to load",
)
parser.add_argument(
    "--scene",
    type=str,
    default="./scene/1027_usdz.usdz",
    help="scene file to load",
)

parser.add_argument("--graspfile", type=str, default="/home/j300/GenPointCloud/output/isaac_grasp_scene_20251007_104221.yaml", help="grasp file to load")
parser.add_argument("--object", type=str, default="/home/j300/GenPointCloud/output/object_20251007_104221.usd", help="object file to load")
parser.add_argument("--scene", type=str, default="/home/j300/GenPointCloud/output/scene_20251007_104221.usd", help="scene file to load")
"""
args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp(  # noqa: E402
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)


from isaacsim_utils.helper import add_extensions, add_robot_to_scene  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid, sphere  # noqa: E402

# import omni.isaac.core.utils.prims as prims_utils  # noqa: E402
from omni.isaac.core.utils.types import ArticulationAction  # noqa: E402


######### CuRobo ########
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType  # noqa: E402
from curobo.geom.types import WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.types.robot import JointState  # noqa: E402
from curobo.util.logger import log_error, setup_curobo_logger  # noqa: E402
from curobo.util.usd_helper import UsdHelper  # noqa: E402
from curobo.util_file import (  # noqa: E402
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (  # noqa: E402
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)


############################################################
MANUAL_MOVE = np.array([0.1, 0.0, 0.3])
MANUAL_MOVE = np.array([0.0, 0.0, 0.0])
MANUAL_MOVE_BLUECUP = np.array(
    [0.026065930647736046, 1.636318075221711, 0.012234069244568768]
)
MANUAL_MOVE_GREENCUP = np.array([0.03982783504705867, 1.778285295774281, 0.0])
MANUAL_MOVE_TABLE = np.array([-1.2361903358728599, 0.0, -0.7671155871219498])
MANUAL_MOVE_POT = np.array(
    [-0.8507747729708208, 0.3795653053965786, 0.006251720482707973]
)


def cmd_to_move(cmd_plan):
    return cmd_plan.position.cpu().numpy().tolist()


def save_plan(moves):
    # Create a timestamp for the unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plan_filename = os.path.join(output_dir, f"motion_{timestamp}.json")
    points = []
    for move in moves:
        point = {"positions": move.positions, "command": move.command, "duration": 2.0}
        points.append(point)
    """
    # Find indices of finger joints
    finger_joint_indices = [
        i
        for i, name in enumerate(cmd_plan.joint_names)
        if "finger" in name
    ]

    # Get the number of joints to keep
    num_joints = len(cmd_plan.joint_names) - len(finger_joint_indices)

    # Prepare the data for saving in the new format
    points = []
    time_from_start = 2.0

    positions_np = cmd_plan.position.cpu().numpy()

    for i in range(len(positions_np)):
        # Filter out finger joints from positions
        positions = [
            float(pos)
            for j, pos in enumerate(positions_np[i])
            if j not in finger_joint_indices
        ]

        point = {
            "positions": positions,
            "velocities": [0.005] * num_joints,
            "time_from_start_sec": round(time_from_start, 2),
        }
        points.append(point)
        time_from_start += 0.1
    """
    # Save the data to a JSON file
    with open(plan_filename, "w") as f:
        json.dump(points, f, indent=4)
    print(f"Successfully saved first motion plan to {plan_filename}")


def init_pose_matric(motion_gen):
    pose_metric = None
    if args.constrain_grasp_approach:
        pose_metric = PoseCostMetric.create_grasp_approach_metric()
    if args.reach_partial_pose is not None:
        reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
        pose_metric = PoseCostMetric(
            reach_partial_pose=True, reach_vec_weight=reach_vec
        )
    if args.hold_partial_pose is not None:
        hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
        pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
    return pose_metric


class Move:
    def __init__(self, command):
        self.positions = []
        self.command = command


COMMANDS = [{}]


def main():
    setup_curobo_logger("warn")
    # create a curobo motion gen instance:
    # num_targets = 0
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage

    ###### Setup Robot ######
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = (
            args.external_robot_configs_path
        )
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    # default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    # default_config = [-0.0031330717463317853, -0.782597719463971, 0.0013149953555822147, -2.3538521161513803, 0.006049369975929311, 1.5787788643767775, 0.7724911821319892, 1.0, 1.0]
    default_config = [
        0.04642576,
        -0.32005848,
        2.44768465,
        1.07295861,
        -1.52266015,
        3.12450588,
    ]
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    ###### Setup Collision table ######
    # don't really understand, just keep it for now
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    ###### add our object and scene into the world ######
    # only visual though
    """
    if args.table is not None:
        table_path = os.path.abspath(args.table)
        add_reference_to_stage(table_path, "/World/table")
        table_prim = XFormPrim("/World/table")
        current_pos_table, current_rot_table = table_prim.get_world_pose()
        table_prim.set_world_pose(
            position=current_pos_table + MANUAL_MOVE_TABLE,
            orientation=current_rot_table,
        )
    if args.object is not None:
        asset_path = os.path.abspath(args.object)
        add_reference_to_stage(asset_path, "/World/object")
        obj_prim = XFormPrim("/World/object")
        current_pos, current_rot = obj_prim.get_world_pose()
        obj_prim.set_world_pose(
            position=current_pos + MANUAL_MOVE, orientation=current_rot
        )
    if args.scene is not None:
        scene_path = os.path.abspath(args.scene)
        add_reference_to_stage(scene_path, "/World/scene")
        scene_prim = XFormPrim("/World/scene")
        current_pos_scene, current_rot_scene = scene_prim.get_world_pose()
        scene_prim.set_world_pose(
            position=current_pos_scene + MANUAL_MOVE,
            orientation=current_rot_scene,
        )
    if args.pot is not None:
        pot_path = os.path.abspath(args.pot)
        add_reference_to_stage(pot_path, "/World/pot")
        pot_prim = XFormPrim("/World/pot")
        current_pos_pot, current_rot_pot = pot_prim.get_world_pose()
        pot_prim.set_world_pose(
            position=current_pos_pot + MANUAL_MOVE_POT,
            orientation=current_rot_pot,
        )
    if args.bluecup is not None:
        bluecup_path = os.path.abspath(args.bluecup)
        add_reference_to_stage(bluecup_path, "/World/bluecup")
        bluecup_prim = XFormPrim("/World/bluecup")
        current_pos_bluecup, current_rot_bluecup = bluecup_prim.get_world_pose()
        bluecup_prim.set_world_pose(
            position=current_pos_bluecup + MANUAL_MOVE_BLUECUP,
            orientation=current_rot_bluecup,
        )
    if args.greencup is not None:
        greencup_path = os.path.abspath(args.greencup)
        add_reference_to_stage(greencup_path, "/World/greencup")
        greencup_prim = XFormPrim("/World/greencup")
        current_pos_greencup, current_rot_greencup = greencup_prim.get_world_pose()
        greencup_prim.set_world_pose(
            position=current_pos_greencup + MANUAL_MOVE_GREENCUP,
            orientation=current_rot_greencup,
        )
    """
    ###### Motion Generation Config ######
    trajopt_tsteps = 32
    trajopt_dt = None
    optimize_dt = True
    max_attempts = 10
    trim_steps = None
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False

    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100
    tensor_args = TensorDeviceType()

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )

    motion_gen = MotionGen(motion_gen_config)
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    ### add a no obstacle world
    zero_obstacles = usd_help.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[
            robot_prim_path,
            "/World/defaultGroundPlane",
            "/curobo",
            "/World/table",
        ],
    ).get_collision_check_world()

    ###### Pose matrice initialization
    pose_metric = init_pose_matric(motion_gen)

    ###### states ######

    # new_id = 0
    # idle = 0

    cmd_plan = None
    cmd_idx = 0
    # my_world.scene.add_default_ground_plane()
    articulation_controller = robot.get_articulation_controller()
    tick = 0
    spheres = None
    past_cmd = None

    # pose_metric = None
    wait_ros2 = False
    graspgen_receiver = NonBlockingJSONReceiver(port=9878)
    graspgen_sender = NonBlockingJSONSender(port=9879)
    ros2_receiver = NonBlockingJSONReceiver(port=9877)
    ros2_sender = NonBlockingJSONSender(port=9876)
    plans = []
    cmd_plans = []
    last_joint_states = default_config
    successs = 0
    temp_cuboid_paths = []
    sim_js = robot.get_joints_state()
    sim_js_names = robot.dof_names
    while simulation_app.is_running():
        # Catpure info from both ROS2 and GraspGen
        if len(plans) == 0:  # accept new plans
            graspgen_datas = graspgen_receiver.capture_data()
            # print("---------------------------",graspgen_data, "---------------------------")
            if graspgen_datas is not None:
                print("-------------Received new action--------------")
                for graspgen_data in graspgen_datas:
                    if temp_cuboid_paths:
                        for path in temp_cuboid_paths:
                            stage.RemovePrim(path)
                        temp_cuboid_paths = []
                    # Create new temporary obstacles in the Isaac Sim stage
                    if "obstacles" in graspgen_data and graspgen_data["obstacles"]:
                        for i, obs_data in enumerate(graspgen_data["obstacles"]):
                            mass_center = obs_data["mass_center"]
                            std = obs_data["std"]
                            prim_path = f"/World/temp_obstacle_{i}"
                            cuboid.FixedCuboid(
                                prim_path=prim_path,
                                position=np.array(mass_center),
                                scale=np.array([std[0] * 5, std[1] * 3, std[2] * 5]),
                                color=np.array([0.0, 0.0, 1.0]),  # Blue
                                # physics=True,
                            )
                            temp_cuboid_paths.append(prim_path)
                    # Table
                    prim_path = "/World/temp_obstacle_table"
                    cuboid.FixedCuboid(
                        prim_path=prim_path,
                        position=np.array([0, 0, -1.97]),
                        scale=np.array([4, 4, 4]),
                        color=np.array([0.0, 0.0, 1.0]),  # Blue
                        # physics=True,
                    )
                    temp_cuboid_paths.append(prim_path)
                    # Get all obstacles from the stage, including the new temporary ones
                    obstacles = usd_help.get_obstacles_from_stage(
                        only_paths=["/World"],
                        reference_prim_path=robot_prim_path,
                        ignore_substring=[
                            robot_prim_path,
                            "/World/defaultGroundPlane",
                            "/curobo",
                            "/World/table",
                        ],
                    ).get_collision_check_world()

                    # Update the motion planner's world
                    # motion_gen.update_world(obstacles)
                    print(graspgen_data)
                    for move in graspgen_data["moves"]:
                        if "no_obstacles" in move:
                            motion_gen.update_world(zero_obstacles)
                        else:
                            motion_gen.update_world(obstacles)
                        if move["type"] == "gripper":
                            plans.append(move)
                            continue
                        ## start serious curobo motion planning
                        if "constraint" in move:
                            pose_metric = PoseCostMetric(
                                hold_partial_pose=True,
                                hold_vec_weight=motion_gen.tensor_args.to_device(
                                    move["constraint"]
                                ),
                            )
                            print(
                                "CONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINT"
                            )
                        # else:
                        #     pose_metric = None
                        if "goal" in move:
                            ik_goal = Pose(
                                position=tensor_args.to_device(move["goal"][:3]),
                                quaternion=tensor_args.to_device(move["goal"][3:]),
                            )

                            plan_config.pose_cost_metric = pose_metric
                            cu_js = JointState(
                                position=tensor_args.to_device(last_joint_states),
                                velocity=tensor_args.to_device(sim_js.velocities)
                                * 0.0,  # * 0.0,
                                acceleration=tensor_args.to_device(sim_js.velocities)
                                * 0.0,
                                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                                joint_names=sim_js_names,
                            )

                            result = motion_gen.plan_single(
                                cu_js.unsqueeze(0), ik_goal, plan_config
                            )
                        elif "joints_goal" in move:
                            print("ALRIGHT?0")
                            joints_goal = JointState(
                                position=tensor_args.to_device(move["joints_goal"]),
                                velocity=tensor_args.to_device(sim_js.velocities)
                                * 0.0,  # * 0.0,
                                acceleration=tensor_args.to_device(sim_js.velocities)
                                * 0.0,
                                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                                joint_names=sim_js_names,
                            )
                            print("ALRIGHT?1")

                            plan_config.pose_cost_metric = pose_metric
                            cu_js = JointState(
                                position=tensor_args.to_device(last_joint_states),
                                velocity=tensor_args.to_device(sim_js.velocities)
                                * 0.0,  # * 0.0,
                                acceleration=tensor_args.to_device(sim_js.velocities)
                                * 0.0,
                                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                                joint_names=sim_js_names,
                            )
                            print("ALRIGHT?2")
                            result = motion_gen.plan_single_js(
                                cu_js.unsqueeze(0),
                                joints_goal.unsqueeze(0),
                                plan_config,
                            )
                            print("ALRIGHT?3")

                        succ = result.success.item()  # ik_result.success.item()

                        if succ:
                            print("YES YES YES?")
                            new_cmd_plan = result.get_interpolated_plan()
                            new_cmd_plan = motion_gen.get_full_js(new_cmd_plan)
                            # get only joint names that are in both:
                            idx_list = []
                            common_js_names = []
                            for x in sim_js_names:
                                if x in new_cmd_plan.joint_names:
                                    idx_list.append(robot.get_dof_index(x))
                                    common_js_names.append(x)

                            new_cmd_plan = new_cmd_plan.get_ordered_joint_state(
                                common_js_names
                            )

                            positions = cmd_to_move(new_cmd_plan)
                            plans.append(
                                {
                                    "type": move["type"],
                                    "wait_time": move["wait_time"],
                                    # "cmd_plan": cmd_plan, # only for later reuse by isaacsim, not for ROS2
                                    "joints_values": positions,
                                }
                            )
                            cmd_plans.append(new_cmd_plan)
                            last_joint_states = positions[-1]
                            new_cmd_plan = None
                        else:
                            print("This plan failed.")
                            break

                    else:  # success!
                        print(
                            "-------------Successfully handled new action--------------"
                        )
                        graspgen_sender.send_data({"message": "Success"})
                        break  # stop trying other acts
                else:  # all graspgen_datas failed
                    # clean up plans
                    plans = []
                    cmd_plans = []
                    graspgen_sender.send_data({"message": "Fail"})

        if not wait_ros2 and len(plans) > 0:
            wait_ros2 = True
            plan = plans.pop(0)
            ros2_sender.send_data(plan)
            if plan["type"] == "arm":
                cmd_idx = 0
                cmd_plan = cmd_plans.pop(0)
            # print(wait_ros2, successs)

        if wait_ros2:
            ros2_response = ros2_receiver.capture_data()
            if ros2_response is not None and ros2_response["message"] == "Success":
                print("receiver successfulness.")
                wait_ros2 = False
                successs += 1
                # print(wait_ros2, successs)
        # Step
        my_world.step(render=True)
        step_index = my_world.current_time_step_index
        ###### Print the press play hint ######
        if not my_world.is_playing():
            if tick % 100 == 0:
                print("**** Click Play to start simulation *****")
            tick += 1
            continue

        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]),
                joint_indices=idx_list,
            )
        if step_index < 20:
            continue
        """ skil the check because it prints too many needless logs
        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects))

            motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")
        """

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        # Handle reactive mode
        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0
        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
        ###### update past_pose
        # past_pose = cube_position
        # past_orientation = cube_orientation
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # print("======art_action: ", art_action.joint_positions)
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None

    simulation_app.close()


if __name__ == "__main__":
    main()
