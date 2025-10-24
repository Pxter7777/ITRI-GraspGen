import numpy as np
import logging
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from common_utils.qualification import is_qualified_with_name

def get_left_up_and_front(grasp: np.array):
    left = grasp[:3, 0]
    up = grasp[:3, 1]
    front = grasp[:3, 2]
    return left, up, front


def is_qualified(grasp: np.array, mass_center, obj_std):
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    if up[2] < 0.95:
        return False
    # if front[0] < 0.8:
    #    return False
    # if front[1] < -0.2:
    #    return False

    # Rule: planar 2D angle between grasp approach (front) vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(30):
        return False

    if position[2] < 0.056:  # for safety
        return False
    if position[2] > mass_center[2] + obj_std[2]:  # too high
        return False
    if position[2] < mass_center[2] - obj_std[2]:  # too low
        return False
    return True


class GraspGenerator:
    def __init__(self, gripper_config, grasp_threshold, num_grasps, topk_num_grasps):
        self.grasp_cfg = load_grasp_cfg(gripper_config)
        self.gripper_name = self.grasp_cfg.data.gripper_name
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        self.grasp_threshold = grasp_threshold
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps

    def auto_select_valid_cup_grasp(self, pointcloud: np.array) -> np.array:
        mass_center = np.mean(pointcloud, axis=0)
        std = np.std(pointcloud, axis=0)
        try:
            num_try = 0
            while True:
                num_try += 1
                logging.info(f"try #{num_try}")
                grasps, grasp_conf = GraspGenSampler.run_inference(
                    pointcloud,
                    self.grasp_sampler,
                    grasp_threshold=self.grasp_threshold,
                    num_grasps=self.num_grasps,
                    topk_num_grasps=self.topk_num_grasps,
                )
                grasps = grasps.cpu().numpy()
                grasps[:, 3, 3] = 1
                qualified_grasps = np.array(
                    [grasp for grasp in grasps if is_qualified(grasp, mass_center, std)]
                )
                if len(qualified_grasps) > 0:
                    return qualified_grasps[0]
        except KeyboardInterrupt:  # manual stop if too many fail
            logging.info("Manually stopping generating grasps")
            return None
    def flexible_auto_select_valid_grasp(self, obj: dict, qualifier:str) -> np.array:
        mass_center = np.mean(obj["points"], axis=0)
        std = np.std(obj["points"], axis=0)

        num_try = 0
        while True:
            num_try += 1
            logging.info(f"try #{num_try}")
            grasps, grasp_conf = GraspGenSampler.run_inference(
                obj["points"],
                self.grasp_sampler,
                grasp_threshold=self.grasp_threshold,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.topk_num_grasps,
            )
            grasps = grasps.cpu().numpy()
            grasps[:, 3, 3] = 1
            qualified_grasps = np.array(
                [grasp for grasp in grasps if is_qualified_with_name(grasp, qualifier, mass_center, std)]
            )
            if len(qualified_grasps) > 0:
                return qualified_grasps[0]

