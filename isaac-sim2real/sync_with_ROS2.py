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
import queue
from threading import Lock
import numpy as np
from isaacsim_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from isaacsim_utils import port_config



############################################################

# Third Party
from omni.isaac.kit import SimulationApp  # noqa: E402

def get_headless_mode():
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument(
        "--headless_mode",
        type=str,
        default=None,
        help="To run headless, use one of [native, websocket], webrtc might not work.",
    )
    peek_args, _ = peek_parser.parse_known_args()
    return peek_args.headless_mode
simulation_app = SimulationApp(  # noqa: E402
    {
        "headless": get_headless_mode() is not None,
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
from curobo.geom.types import Cuboid, WorldConfig  # noqa: E402
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


def cmd_to_move(cmd_plan):
    return cmd_plan.position.cpu().numpy().tolist()


def init_pose_matric(args, motion_gen):
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


# dataclass
class Move:
    def __init__(self, ROS2_move: dict, cmd_plan: list):
        self.ROS2_move = ROS2_move
        self.cmd_plan = cmd_plan


def parse_args():
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
    return parser.parse_args()

def main():
    args = parse_args()
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
        1.37296326,
        0.08553859,
        1.05554023,
        2.76803983,
        -1.48792809,
        3.09947786,
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

    ###### Motion Generation Config ######
    trajopt_tsteps = 32
    trajopt_dt = None
    optimize_dt = True
    max_attempts = 4
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
    motion_gen_lock = Lock()
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, get_headless_mode())

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
    # zero_obstacles = usd_help.get_obstacles_from_stage(
    #     only_paths=["/World"],
    #     reference_prim_path=robot_prim_path,
    #     ignore_substring=[
    #         robot_prim_path,
    #         "/World/defaultGroundPlane",
    #         "/curobo",
    #         "/World/table",
    #     ],
    # ).get_collision_check_world()

    ###### Pose matrice initialization
    pose_metric = init_pose_matric(args, motion_gen)

    ###### states ######

    # new_id = 0
    # idle = 0

    planned_action_queue = queue.Queue()
    ROS2_fail_queue = queue.Queue()  # put message in if ROS2 fail
    cmd_plan = None
    cmd_idx = 0
    # my_world.scene.add_default_ground_plane()
    articulation_controller = robot.get_articulation_controller()
    tick = 0
    spheres = None
    past_cmd = None

    # pose_metric = None
    wait_ros2 = False
    # graspgen_receiver = NonBlockingJSONReceiver(port=port_config.GRASPGEN_TO_ISAACSIM)
    # graspgen_sender = NonBlockingJSONSender(port=port_config.ISAACSIM_TO_GRASPGEN)
    ros2_receiver = NonBlockingJSONReceiver(port=port_config.ROS2_TO_ISAACSIM)
    ros2_sender = NonBlockingJSONSender(port=port_config.ISAACSIM_TO_ROS2)
    # plans = []
    # cmd_plans = []
    # last_joint_states = default_config
    # successs = 0
    # temp_cuboid_paths = []
    sim_js = robot.get_joints_state()
    sim_js_names = robot.dof_names
    planned_action_moves: list = []
    idx_list = [0, 1, 2, 3, 4, 5]
    temp_cuboid_paths = []

    graspgen_receiver = NonBlockingJSONReceiver(port=port_config.GRASPGEN_TO_ISAACSIM)
    graspgen_sender = NonBlockingJSONSender(port=port_config.ISAACSIM_TO_GRASPGEN)
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
    default_config = [
        1.37296326,
        0.08553859,
        1.05554023,
        2.76803983,
        -1.48792809,
        3.09947786,
    ]
    last_joint_states = default_config
    temp_cuboid_paths = []
    # idx_list = [0,1,2,3,4,5]
    common_js_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    graspgen_eof = False

    while simulation_app.is_running():
        # try handle first
        for _ in range(1):
            if not ROS2_fail_queue.empty():
                # clear plan
                while not planned_action_queue.empty():
                    planned_action_queue.get()
                notice = (
                    ROS2_fail_queue.get()
                )  # catch the fail, let isaacsim continue to move
                print("get Notice", notice)
                if notice["message"] == "Abort":
                    last_joint_states = default_config
                    graspgen_sender.send_data({"message": "Abort"})
                    # eat datas
                    for _ in range(5):
                        time.sleep(0.1)
                        graspgen_receiver.capture_data()
                elif notice["message"] == "ROS2 Complete":
                    if graspgen_eof:
                        graspgen_sender.send_data({"message": "EOF and ROS2 Complete"})
                    else:
                        print("ROS2 complete but not EOF yet.")
                else:
                    raise ValueError("Unknown message")

                continue

            graspgen_datas = graspgen_receiver.capture_data()
            if graspgen_datas is None:
                continue
            if graspgen_datas[0] == "EOF":
                graspgen_eof = True
                continue
            elif graspgen_datas[0] == "Reset_to_default":
                last_joint_states = default_config
                continue
            print("-------------Received new action--------------")
            graspgen_eof = False
            for graspgen_data in graspgen_datas:
                before_move_joints = last_joint_states
                curobo_planned_action_moves: list[Move] = []
                for move in graspgen_data["moves"]:
                    # handle temp obstacles
                    # if temp_cuboid_paths:
                    #     for path in temp_cuboid_paths:
                    #         stage.RemovePrim(path)  # this may race condition
                    #     temp_cuboid_paths = []
                    # Table # race condition, I guess? Yes this shit will race condition
                    # prim_path = "/World/temp_obstacle_table"
                    # cuboid.FixedCuboid(
                    #     prim_path=prim_path,
                    #     position=np.array([0, 0, -1.97]),
                    #     scale=np.array([4, 4, 4]),
                    #     color=np.array([0.0, 0.0, 1.0]),  # Blue
                    #     # physics=True,
                    # )
                    # temp_cuboid_paths.append(prim_path)
                    cuboids = []
                    cuboids.append(
                        Cuboid(
                            name="table",
                            pose=[0, 0, -1.97] + [1, 0, 0, 0],
                            dims=[4, 4, 4],
                        )
                    )
                    for i, obstacle_name in enumerate(graspgen_data["obstacles"]):
                        if not (
                            "ignore_obstacles" in move
                            and obstacle_name in move["ignore_obstacles"]
                        ):
                            middle_point = np.mean(
                                [
                                    graspgen_data["obstacles"][obstacle_name]["max"],
                                    graspgen_data["obstacles"][obstacle_name]["min"],
                                ],
                                axis=0,
                            )
                            scale = np.array(
                                graspgen_data["obstacles"][obstacle_name]["max"]
                            ) - np.array(
                                graspgen_data["obstacles"][obstacle_name]["min"]
                            )
                            cuboids.append(
                                Cuboid(
                                    name=f"obs_{i}",
                                    pose=middle_point.tolist() + [1, 0, 0, 0],
                                    dims=scale.tolist(),
                                )
                            )

                            # Race condition
                            # prim_path = f"/World/temp_obstacle_{i}"
                            # cuboid.FixedCuboid(
                            #     prim_path=prim_path,
                            #     position=np.array(middle_point),
                            #     scale=[scale[0], scale[1], scale[2] * 1.1],
                            #     color=np.array([0.0, 0.0, 1.0]),  # Blue
                            #     # physics=True,
                            # )
                            # temp_cuboid_paths.append(prim_path)
                    # Get all obstacles from the stage, including the new temporary ones
                    # obstacles = usd_help.get_obstacles_from_stage(
                    #     only_paths=["/World"],
                    #     reference_prim_path=robot_prim_path,
                    #     ignore_substring=[
                    #         robot_prim_path,
                    #         "/World/defaultGroundPlane",
                    #         "/curobo",
                    #         "/World/table",
                    #     ],
                    # ).get_collision_check_world()
                    obstacles = WorldConfig(cuboid=cuboids)

                    # motion_gen.update_world(world)
                    if "no_obstacles" in move:
                        with motion_gen_lock:
                            motion_gen.update_world(zero_obstacles)
                    else:
                        with motion_gen_lock:
                            motion_gen.update_world(obstacles)
                    # start handle move
                    if move["type"] == "gripper":
                        curobo_planned_action_moves.append(Move(move, None))
                        continue
                    ## start serious curobo motion planning
                    # if "constraint" in move:
                    #     pose_metric = PoseCostMetric(
                    #         hold_partial_pose=True,
                    #         hold_vec_weight=motion_gen.tensor_args.to_device(
                    #             move["constraint"]
                    #         ),
                    #     )
                    #     print(
                    #         "CONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINTCONSTRAINT"
                    #     )
                    # else:
                    #     pose_metric = None
                    print("curoboing")
                    if "goal" in move:
                        ik_goal = Pose(
                            position=tensor_args.to_device(move["goal"][:3]),
                            quaternion=tensor_args.to_device(move["goal"][3:]),
                        )

                        plan_config.pose_cost_metric = pose_metric
                        curobo_cu_js = JointState(
                            position=tensor_args.to_device(last_joint_states),
                            velocity=tensor_args.to_device(sim_js.velocities)
                            * 0.0,  # * 0.0,
                            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                            joint_names=sim_js_names,
                        )

                        with motion_gen_lock:
                            result = motion_gen.plan_single(
                                curobo_cu_js.unsqueeze(0), ik_goal, plan_config
                            )
                    elif "joints_goal" in move:
                        print("ALRIGHT?0")
                        joints_goal = JointState(
                            position=tensor_args.to_device(move["joints_goal"]),
                            velocity=tensor_args.to_device(sim_js.velocities)
                            * 0.0,  # * 0.0,
                            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                            joint_names=sim_js_names,
                        )
                        print("ALRIGHT?1")

                        plan_config.pose_cost_metric = pose_metric
                        curobo_cu_js = JointState(
                            position=tensor_args.to_device(last_joint_states),
                            velocity=tensor_args.to_device(sim_js.velocities)
                            * 0.0,  # * 0.0,
                            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                            joint_names=sim_js_names,
                        )
                        print("ALRIGHT?2")
                        with motion_gen_lock:
                            result = motion_gen.plan_single_js(
                                curobo_cu_js.unsqueeze(0),
                                joints_goal.unsqueeze(0),
                                plan_config,
                            )
                        print("ALRIGHT?3")

                    succ = result.success.item()  # ik_result.success.item()
                    if succ:
                        print("successful")
                    else:
                        print("not successful")
                    if succ:
                        print("YES YES YES?")
                        with motion_gen_lock:
                            new_cmd_plan = result.get_interpolated_plan()
                            new_cmd_plan = motion_gen.get_full_js(new_cmd_plan)
                        # get only joint names that are in both:
                        # idx_list = []
                        # common_js_names = []
                        # for x in sim_js_names:
                        #     if x in new_cmd_plan.joint_names:
                        #         idx_list.append(robot.get_dof_index(x))
                        #         common_js_names.append(x)

                        new_cmd_plan = new_cmd_plan.get_ordered_joint_state(
                            common_js_names
                        )
                        # The following code block shows how to prune the plan to keep only the first and last waypoints
                        if "no_curobo" in move:
                            new_cmd_plan = JointState(
                                position=new_cmd_plan.position[[0, -1]],
                                velocity=new_cmd_plan.velocity[[0, -1]],
                                acceleration=new_cmd_plan.acceleration[[0, -1]],
                                jerk=new_cmd_plan.jerk[[0, -1]],
                                joint_names=new_cmd_plan.joint_names,
                            )
                            print(
                                "---------------------------------------------------------only keep first and last"
                            )
                        positions = cmd_to_move(new_cmd_plan)
                        ROS2_move = {
                            "type": move["type"],
                            "wait_time": move["wait_time"],
                            # "cmd_plan": cmd_plan, # only for later reuse by isaacsim, not for ROS2
                            "joints_values": positions,
                        }

                        curobo_planned_action_moves.append(
                            Move(ROS2_move, new_cmd_plan)
                        )
                        last_joint_states = positions[-1]
                    else:
                        print("This plan failed.")
                        last_joint_states = before_move_joints
                        break

                else:  # success!
                    print("-------------Successfully handled new action--------------")
                    graspgen_sender.send_data({"message": "Success"})
                    planned_action_queue.put(
                        {"moves": curobo_planned_action_moves, "obstacles": cuboids}
                    )
                    break  # stop trying other acts
            else:  # all graspgen_datas failed
                graspgen_sender.send_data({"message": "Fail"})
        # end of handle section
        # make sure the thread has catched and handled the issue
        # if not ROS2_fail_queue.empty():
        # continue

        if wait_ros2:
            ros2_response = ros2_receiver.capture_data()
            if ros2_response is not None:
                wait_ros2 = False
                if ros2_response["message"] == "Success":
                    print("receiver successfulness.")
                    if len(planned_action_moves) == 0 and planned_action_queue.empty():
                        ROS2_fail_queue.put({"message": "ROS2 Complete"})
                        print("ROS2 Complete, go check that!")
                    # Can continue to do the following steps, no need to stuck
                elif ros2_response["message"] == "Fail":
                    print("receiver failedness.")
                    # reset simulation robot position
                    robot.set_joint_positions(default_config, idx_list)
                    robot._articulation_view.set_max_efforts(
                        values=np.array([5000 for i in range(len(idx_list))]),
                        joint_indices=idx_list,
                    )
                    cmd_plan = None
                    planned_action_moves = []
                    # Finished handled
                    ROS2_fail_queue.put(
                        {"message": "Abort"}
                    )  # tell that thread to handle
                    continue  # Go and please stuck

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
        with motion_gen_lock:
            cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            with motion_gen_lock:
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
        if not wait_ros2 and len(planned_action_moves) > 0:
            # planned action
            wait_ros2 = True
            move: Move = planned_action_moves.pop(0)
            # For ROS2
            ros2_sender.send_data(move.ROS2_move)
            # For isaac sim animation
            cmd_idx = 0
            cmd_plan = move.cmd_plan

        if len(planned_action_moves) == 0 and not planned_action_queue.empty():
            # currently no plan but we have more in queue, can start grab a new planned_action and apply here.
            planned_action = planned_action_queue.get()
            planned_action_moves: list = planned_action["moves"]
            # visualize cuboids
            if temp_cuboid_paths:
                for path in temp_cuboid_paths:
                    stage.RemovePrim(path)  # this may race condition
                temp_cuboid_paths = []
            cube: Cuboid
            for i, cube in enumerate(planned_action["obstacles"]):
                # race condition???
                prim_path = f"/World/temp_obstacle_{i}"
                cuboid.VisualCuboid(
                    prim_path=prim_path,
                    position=np.array(cube.pose[:3]),
                    scale=np.array(cube.dims),
                    color=np.array([0.0, 0.0, 1.0]),  # Blue
                )
                temp_cuboid_paths.append(prim_path)
    simulation_app.close()


if __name__ == "__main__":
    main()
