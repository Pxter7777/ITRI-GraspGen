import argparse
import time
import numpy as np

from isaacsim_utils import network_config
from isaacsim_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": "1920", "height": "1080"})

from isaacsim_utils.helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

def get_cuboid_list(obstacles: dict) -> list:
    cuboids = []
    cuboids.append(
        Cuboid(
            name="table",
            pose=[0, 0, -1.97] + [1, 0, 0, 0],
            dims=[4, 4, 4],
        )
    )
    for i, (_obs_name, bounds) in enumerate(obstacles.items()):
        middle_point = np.mean([bounds["max"], bounds["min"]], axis=0)
        scale = np.array(bounds["max"]) - np.array(bounds["min"])
        cuboids.append(
            Cuboid(
                name=f"obs_{i}",
                pose=middle_point.tolist() + [1, 0, 0, 0],
                dims=scale.tolist(),
            )
        )
    return cuboids

def basic_world_config():
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    return WorldConfig(cuboid=world_cfg_table.cuboid)

def basic_motion_gen(tensor_args, robot_cfg, world_cfg):
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.05,
        collision_cache={"obb": 30, "mesh": 100},
        optimize_dt=True,
    )
    return MotionGen(motion_gen_config)

def basic_plan_config():
    return MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=4,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )

def still_joint_states(joint_states: list, tensor_args: TensorDeviceType, sim_js_names):
    return JointState(
        position=tensor_args.to_device(joint_states),
        velocity=tensor_args.to_device([0.0] * len(joint_states)),
        acceleration=tensor_args.to_device([0.0] * len(joint_states)),
        jerk=tensor_args.to_device([0.0] * len(joint_states)),
        joint_names=sim_js_names,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="tm5s.yml", help="robot configuration to load")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_curobo_logger("warn")
    my_world = World(stage_units_in_meters=1.0)

    robot_cfg_path = get_robot_configs_path()
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    world_cfg = basic_world_config()
    tensor_args = TensorDeviceType()
    motion_gen = basic_motion_gen(tensor_args, robot_cfg, world_cfg)
    plan_config = basic_plan_config()

    print("Warming up Curobo...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    add_extensions(simulation_app, None)

    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)

    zero_obstacles = usd_help.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[robot_prim_path, "/World/defaultGroundPlane", "/curobo", "/World/table"],
    ).get_collision_check_world()

    default_config = [1.37296326, 0.08553859, 1.05554023, 2.76803983, -1.48792809, 3.09947786]

    sender = NonBlockingJSONSender(port=network_config.EXPERIMENT_ISAACSIM_TO_GRASPGEN_PORT)
    receiver = NonBlockingJSONReceiver(port=network_config.EXPERIMENT_GRASPGEN_TO_ISAACSIM_PORT)

    sim_js_names = robot.dof_names

    print("Isaac Sim Experiment Node Ready.")

    while simulation_app.is_running():
        data = receiver.capture_data()
        if data is None:
            time.sleep(0.1)
            continue

        if data.get("message") == "EXPERIMENT_START":
            print("Received handshake from GraspGen. Responding READY.")
            sender.send_data({"message": "READY"})
            continue

        if data.get("message") == "EOF":
            print("Experiment finished. Exiting.")
            break

        order_name = data.get("order")
        acts = data.get("acts")

        if not order_name or not acts:
            print("Invalid packet. Ignoring.")
            continue

        print(f"Evaluating {len(acts)} grasps for Order {order_name}...")

        start_time = time.time()
        success_found = False

        for grasp_idx, act in enumerate(acts):
            print(f"  Attempting Grasp {grasp_idx+1}/{len(acts)}")
            last_joint_states = default_config
            moves = act["moves"]
            cuboids = get_cuboid_list(act["obstacles"])
            obstacles = WorldConfig(cuboid=cuboids)

            grasp_plannable = True

            for move in moves:
                if move["type"] == "gripper":
                    continue

                if "no_obstacles" in move:
                    motion_gen.update_world(zero_obstacles)
                else:
                    motion_gen.update_world(obstacles)

                curobo_cu_js = still_joint_states(last_joint_states, tensor_args, sim_js_names)

                if "goal" in move:
                    ik_goal = Pose(
                        position=tensor_args.to_device(move["goal"][:3]),
                        quaternion=tensor_args.to_device(move["goal"][3:]),
                    )
                    result = motion_gen.plan_single(curobo_cu_js.unsqueeze(0), ik_goal, plan_config)
                elif "joints_goal" in move:
                    joints_goal = JointState(
                        position=tensor_args.to_device(move["joints_goal"]),
                        velocity=tensor_args.to_device([0.0]*len(sim_js_names)),
                        acceleration=tensor_args.to_device([0.0]*len(sim_js_names)),
                        jerk=tensor_args.to_device([0.0]*len(sim_js_names)),
                        joint_names=sim_js_names,
                    )
                    result = motion_gen.plan_single_js(curobo_cu_js.unsqueeze(0), joints_goal.unsqueeze(0), plan_config)

                if result.success.item():
                    new_cmd_plan = result.get_interpolated_plan()
                    new_cmd_plan = motion_gen.get_full_js(new_cmd_plan)
                    positions = new_cmd_plan.position.cpu().numpy().tolist()
                    last_joint_states = positions[-1]
                else:
                    grasp_plannable = False
                    break # Stop evaluating this grasp's move sequence

            if grasp_plannable:
                success_found = True
                end_time = time.time()
                time_taken = end_time - start_time
                print(f"✅ Success found at grasp {grasp_idx+1}! Time taken: {time_taken:.2f}s")
                sender.send_data({"order": order_name, "success": True, "time_taken": time_taken})
                break

        if not success_found:
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"❌ No plannable grasp found in Order {order_name}. Time taken: {time_taken:.2f}s")
            sender.send_data({"order": order_name, "success": False, "time_taken": time_taken})

    simulation_app.close()

if __name__ == "__main__":
    main()
