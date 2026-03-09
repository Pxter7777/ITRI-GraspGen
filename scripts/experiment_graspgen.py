import argparse
import logging
import json
import time
import numpy as np

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

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run headless GraspGen experiment.")
    parser.add_argument("--scene", type=str, default="data_for_test/experiment_scenario_1.json", help="Path to scenario")
    parser.add_argument("--gripper_config", type=str, default=str(config.GRIPPER_CFG), help="Path to gripper config")
    parser.add_argument("--target_name", type=str, default="target_cup", help="Target object name")
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


def main():
    args = parse_args()
    logger.info(f"Loading scene {args.scene}")

    with open(args.scene, "r") as f:
        scene_data = json.load(f)

    # Convert obstacle min/max logic to list format for later
    obstacles_cuboids = []
    for name, bounds in scene_data["obstacles"].items():
        obstacles_cuboids.append({
            "name": name,
            "min": bounds["min"],
            "max": bounds["max"]
        })

    obj_pc = np.array(scene_data["object_infos"][args.target_name]["points"])

    # Init GraspGen
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # 1. Generate Grasps
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

    # 2. Collision Filter (using points from the obstacles if we had them, but we only have bounds)
    # Since we have FetchBench meshes, generating a real scene_pc is slow.
    # To mimic original behavior, we can skip explicit collision filtering here and let CuRobo do it,
    # OR we can generate a pseudo pointcloud from the obstacle bounds for GraspGen filtering.
    # Let's generate a pseudo pointcloud for obstacles to ensure graspgen filtering works.
    scene_pc = obj_pc.copy()
    for obs in obstacles_cuboids:
        # sample points inside bounding box
        num_obs_pts = 1000
        pts = np.random.uniform(low=obs["min"], high=obs["max"], size=(num_obs_pts, 3))
        scene_pc = np.vstack((scene_pc, pts))

    gripper_info = get_gripper_info(gripper_name)
    gripper_collision_mesh = gripper_info.collision_mesh
    collision_free_mask = filter_colliding_grasps(
        scene_pc=scene_pc,
        grasp_poses=grasps,
        gripper_collision_mesh=gripper_collision_mesh,
        collision_threshold=0.03,
    )

    valid_grasps = grasps[collision_free_mask]
    logger.info(f"{len(valid_grasps)} grasps passed collision filter.")

    if len(valid_grasps) == 0:
        logger.error("No valid grasps found after collision filtering. Aborting.")
        return

    # 3. Sort List A: Discriminator (Original Order)
    order_a_grasps = list(valid_grasps.copy())

    # 4. Sort List B: Heuristic (Cup Qualifier + Angle Offset)
    min_point = np.percentile(obj_pc, 3, axis=0)
    max_point = np.percentile(obj_pc, 97, axis=0)

    custom_filter_mask = np.array(
        [is_qualified(grasp, "cup_qualifier", min_point, max_point) for grasp in valid_grasps]
    )
    heuristic_grasps = valid_grasps[custom_filter_mask]

    if len(heuristic_grasps) == 0:
        logger.warning("No grasps passed the cup_qualifier! Falling back to un-qualified heuristic.")
        heuristic_grasps = valid_grasps.copy()

    order_b_grasps = sorted(heuristic_grasps, key=angle_offset_rad)

    # Format actions for Isaac Sim
    def build_full_acts(grasp_list):
        acts = []
        for grasp in grasp_list:
            # We use grab_and_pour_and_place_back_curobo_by_rotation logic
            act = grab_and_pour_and_place_back_curobo_by_rotation(
                args.target_name, grasp, ["target_cup"], scene_data
            )
            acts.append(act)
        return acts

    logger.info("Building actions...")
    order_a_acts = build_full_acts(order_a_grasps)
    order_b_acts = build_full_acts(order_b_grasps)

    # 5. Socket Comm
    sender = NonBlockingJSONSender(port=network_config.EXPERIMENT_GRASPGEN_TO_ISAACSIM_PORT)
    receiver = NonBlockingJSONReceiver(port=network_config.EXPERIMENT_ISAACSIM_TO_GRASPGEN_PORT)

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
    logger.info(f"Time to first plannable grasp (Order A - Discriminator): {time_a:.2f} s")
    logger.info(f"Time to first plannable grasp (Order B - Heuristic)    : {time_b:.2f} s")
    logger.info("================================")

if __name__ == "__main__":
    main()
