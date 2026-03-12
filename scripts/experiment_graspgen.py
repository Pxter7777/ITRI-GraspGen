import argparse
import logging
import json
import time
import os
import numpy as np
import trimesh

from common_utils import config, network_config
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from common_utils.custom_logger import CustomFormatter
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
from common_utils.qualification import is_qualified, get_left_up_and_front
from common_utils.movesets import grab_and_pour_and_place_back_curobo_by_rotation
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_pointcloud,
)
from common_utils.graspgen_utils import start_meshcat_server, open_meshcat_url

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run headless GraspGen experiment.")
    parser.add_argument(
        "--scene",
        type=str,
        default="data_for_test/experiment_scenario_1.json",
        help="Path to scenario",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default=str(config.GRIPPER_CFG),
        help="Path to gripper config",
    )
    parser.add_argument(
        "--no-vis", action="store_true", help="Disable Meshcat visualization"
    )
    return parser.parse_args()


def angle_offset_rad(grasp: np.ndarray) -> float:
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    return angle_diff


def flip_grasp(grasp: np.ndarray) -> np.ndarray:
    flipped_grasp = grasp.copy()
    flipped_grasp[:3, 0] = -grasp[:3, 0]
    flipped_grasp[:3, 1] = -grasp[:3, 1]
    return flipped_grasp


def flip_upside_down_grasps(grasps: np.ndarray) -> np.ndarray:
    flipped_grasps = []
    for grasp in grasps:
        _grasp = grasp.copy()
        _, up, _ = get_left_up_and_front(_grasp)
        if up[2] < 0:
            _grasp = flip_grasp(_grasp)
        flipped_grasps.append(_grasp)
    return np.array(flipped_grasps)


def _load_and_transform_mesh(mesh_path, pose, scale):
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_scale(scale)
    mesh.apply_translation(pose)
    return mesh


def main():
    args = parse_args()
    logger.info(f"Loading scene {args.scene}")

    with open(args.scene, "r") as f:
        light_scene_data = json.load(f)

    # 0. Meshcat setup
    vis = None
    if not args.no_vis:
        start_meshcat_server()
        open_meshcat_url("http://127.0.0.1:7000/static/")
        vis = create_visualizer()

    # 1. Dynamically load target
    target_info = light_scene_data["target"]
    target_name = target_info["name"]
    target_mesh = _load_and_transform_mesh(
        target_info["mesh_path"], target_info["pose"], target_info["scale"]
    )

    obj_pc, *rest = trimesh.sample.sample_surface(target_mesh, 2048)
    obj_pc = np.array(obj_pc, dtype=np.float32)
    obj_colors = np.tile([255, 255, 255], (len(obj_pc), 1))

    # 2. Dynamically load obstacles and compute bounds
    obstacles_cuboids = []
    scene_pc = obj_pc.copy()
    scene_colors = obj_colors.copy()

    # Add regular obstacles
    for obs_name, obs_info in light_scene_data["obstacles"].items():
        obs_mesh = _load_and_transform_mesh(
            obs_info["mesh_path"], obs_info["pose"], obs_info["scale"]
        )
        bounds = obs_mesh.bounds
        obstacles_cuboids.append(
            {"name": obs_name, "min": bounds[0].tolist(), "max": bounds[1].tolist()}
        )

        obs_pts, *rest = trimesh.sample.sample_surface(obs_mesh, 1024)
        scene_pc = np.vstack((scene_pc, np.array(obs_pts)))
        scene_colors = np.vstack(
            (scene_colors, np.tile([255, 0, 0], (len(obs_pts), 1)))
        )

    # Add robot meshes if present
    if "robot" in light_scene_data:
        for i, r_mesh_info in enumerate(light_scene_data["robot"]["meshes"]):
            r_mesh = _load_and_transform_mesh(
                r_mesh_info["mesh_path"], r_mesh_info["pose"], r_mesh_info["scale"]
            )
            r_pts, *rest = trimesh.sample.sample_surface(r_mesh, 1024)
            scene_pc = np.vstack((scene_pc, np.array(r_pts)))
            scene_colors = np.vstack(
                (scene_colors, np.tile([128, 128, 128], (len(r_pts), 1)))
            )

            # Also treat robot base as an obstacle for collision filtering
            bounds = r_mesh.bounds
            obstacles_cuboids.append(
                {
                    "name": f"robot_mesh_{i}",
                    "min": bounds[0].tolist(),
                    "max": bounds[1].tolist(),
                }
            )

    if vis:
        visualize_pointcloud(vis, "pc_scene", scene_pc, scene_colors, size=0.005)

    # Construct the full scene_data dict that our old movesets.py functions expect
    scene_data = {
        "scene_info": {},
        "object_infos": {target_name: {"points": obj_pc.tolist()}},
        "obstacles": {
            obs["name"]: {"min": obs["min"], "max": obs["max"]}
            for obs in obstacles_cuboids
        },
    }

    # Init GraspGen
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # 3. Generate Grasps
    logger.info("Generating grasps...")
    grasps, grasp_conf = GraspGenSampler.run_inference(
        obj_pc,
        grasp_sampler,
        grasp_threshold=0.8,
        num_grasps=200,
        min_grasps=80,
        max_tries=20,
    )
    grasps = grasps.cpu().numpy()
    grasps[:, 3, 3] = 1
    grasps = flip_upside_down_grasps(grasps)

    # 4. Collision Filter
    gripper_info = get_gripper_info(gripper_name)
    gripper_collision_mesh = gripper_info.collision_mesh
    collision_free_mask = filter_colliding_grasps(
        scene_pc=scene_pc,
        grasp_poses=grasps,
        gripper_collision_mesh=gripper_collision_mesh,
        collision_threshold=0.0003,
    )

    valid_grasps = grasps[collision_free_mask]
    logger.info(f"{len(valid_grasps)} grasps passed collision filter.")

    if len(valid_grasps) == 0:
        logger.error("No valid grasps found after collision filtering. Aborting.")
        return

    # Visualize filtered grasps
    if vis:
        min_point = np.percentile(obj_pc, 3, axis=0)
        max_point = np.percentile(obj_pc, 97, axis=0)
        for j, grasp in enumerate(grasps):
            is_collision_free = collision_free_mask[j]
            is_qual = is_qualified(grasp, "cup_qualifier", min_point, max_point)

            if not is_collision_free:
                color = [255, 0, 0]  # Red = collision
            elif not is_qual:
                color = [255, 165, 0]  # Orange = free but not qualified
            else:
                color = [0, 255, 0]  # Green = free and qualified

            visualize_grasp(
                vis,
                f"GraspGen/{j:03d}/grasp",
                grasp,
                color=color,
                gripper_name=gripper_name,
                linewidth=1.5,
            )

    # 5. Sort List A: Discriminator (Original Order)
    order_a_grasps = list(valid_grasps.copy())

    # 6. Sort List B: Heuristic (Cup Qualifier + Angle Offset)
    min_point = np.percentile(obj_pc, 3, axis=0)
    max_point = np.percentile(obj_pc, 97, axis=0)

    custom_filter_mask = np.array(
        [
            is_qualified(grasp, "cup_qualifier", min_point, max_point)
            for grasp in valid_grasps
        ]
    )
    heuristic_grasps = valid_grasps[custom_filter_mask]

    if len(heuristic_grasps) == 0:
        logger.warning(
            "No grasps passed the cup_qualifier! Falling back to un-qualified heuristic."
        )
        heuristic_grasps = valid_grasps.copy()

    order_b_grasps = sorted(heuristic_grasps, key=angle_offset_rad)

    # Format actions for Isaac Sim
    def build_full_acts(grasp_list):
        acts = []
        for grasp in grasp_list:
            act = grab_and_pour_and_place_back_curobo_by_rotation(
                target_name, grasp, [target_name], scene_data
            )
            acts.append(act)
        return acts

    logger.info("Building actions...")
    order_a_acts = build_full_acts(order_a_grasps)
    order_b_acts = build_full_acts(order_b_grasps)

    # 7. Socket Comm
    sender = NonBlockingJSONSender(
        port=network_config.EXPERIMENT_GRASPGEN_TO_ISAACSIM_PORT
    )
    receiver = NonBlockingJSONReceiver(
        port=network_config.EXPERIMENT_ISAACSIM_TO_GRASPGEN_PORT
    )

    logger.info("Waiting for Isaac Sim to be ready...")

    # Send a handshake
    while True:
        sender.send_data({"message": "EXPERIMENT_START", "scene": args.scene})
        response = receiver.capture_data()
        if response is not None and response.get("message") == "READY":
            break
        time.sleep(1)

    logger.info("Isaac Sim is ready. Sending Order A (Discriminator)...")
    sender.send_data({"order": "A", "acts": order_a_acts})

    time_a = None
    time_b = None

    while True:
        response = receiver.capture_data()
        if response is not None:
            if response.get("order") == "A":
                time_a = response.get("time_taken")
                logger.info(f"Order A completed in {time_a:.2f} seconds.")
                logger.info("Sending Order B (Heuristic)...")
                sender.send_data({"order": "B", "acts": order_b_acts})
            elif response.get("order") == "B":
                time_b = response.get("time_taken")
                logger.info(f"Order B completed in {time_b:.2f} seconds.")
                break
        time.sleep(0.5)

    sender.send_data({"message": "EOF"})

    logger.info("====== EXPERIMENT RESULTS ======")
    logger.info(f"Scenario: {args.scene}")
    logger.info(
        f"Time to first plannable grasp (Order A - Discriminator): {time_a:.2f} s"
    )
    logger.info(
        f"Time to first plannable grasp (Order B - Heuristic)    : {time_b:.2f} s"
    )
    logger.info("================================")


if __name__ == "__main__":
    main()
