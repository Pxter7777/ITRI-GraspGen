"""Grasp pose qualification filters based on geometric heuristics."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_left_up_and_front(
    grasp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract left, up, and front direction vectors from a 4x4 grasp matrix.

    Args:
        grasp (np.ndarray): A 4x4 grasp transformation matrix.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The left, up, and front
            direction vectors.
    """
    left = grasp[:3, 0]
    up = grasp[:3, 1]
    front = grasp[:3, 2]
    return left, up, front


def cup_qualifier(
    grasp: np.ndarray,
    min_point: np.ndarray,
    max_point: np.ndarray,
) -> bool:
    """Filter grasp poses suitable for grasping a cup from the side.

    Args:
        grasp (np.ndarray): A 4x4 grasp transformation matrix.
        min_point (np.ndarray): The 3rd-percentile bounding point of the object.
        max_point (np.ndarray): The 97th-percentile bounding point of the object.

    Returns:
        bool: True if the grasp passes all qualification checks.
    """
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    position += front * 0.20  # offset
    if abs(left[2]) > 0.2:
        return False
    if up[2] < 0.85:
        return False
    # Rule: planar 2D angle between grasp approach (front)
    # vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(90):
        return False

    if position[2] < 0.05:  # for safety
        return False
    z_range = max_point[2] - min_point[2]
    if position[2] > min_point[2] + z_range * 0.75:  # too high
        return False
    if position[2] < min_point[2] + z_range * 0.3:  # too low
        return False
    return True


def small_cup_qualifier(
    grasp: np.ndarray, mass_center: np.ndarray, obj_std: np.ndarray
) -> bool:
    """Filter grasp poses suitable for grasping a small cup.

    Args:
        grasp (np.ndarray): A 4x4 grasp transformation matrix.
        mass_center (np.ndarray): The mass center of the object point cloud.
        obj_std (np.ndarray): The standard deviation of the object point cloud.

    Returns:
        bool: True if the grasp passes all qualification checks.
    """
    position = grasp[:3, 3].tolist()
    _left, up, front = get_left_up_and_front(grasp)
    if up[2] < 0.7:
        return False
    # Rule: planar 2D angle between grasp approach (front)
    # vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(90):
        return False

    # if position[2] < 0.05:  # for safety
    #     return False
    # if position[2] > mass_center[2] + obj_std[2] * 2:  # too high
    #     return False
    # if position[2] < mass_center[2] - obj_std[2] * 1.5:  # too low
    #     return False
    return True  #


def small_cube_qualifier(
    grasp: np.ndarray,
    mass_center: np.ndarray,
    obj_std: np.ndarray,
) -> bool:
    """Filter grasp poses suitable for grasping a small cube from above.

    Args:
        grasp (np.ndarray): A 4x4 grasp transformation matrix.
        mass_center (np.ndarray): The mass center of the object point cloud.
        obj_std (np.ndarray): The standard deviation of the object point cloud.

    Returns:
        bool: True if the grasp passes all qualification checks.
    """
    position = grasp[:3, 3].tolist()
    _left, _up, front = get_left_up_and_front(grasp)
    if front[0] < 0:
        return False
    if front[2] > -0.2:  # not facing down
        return False
    # Rule: planar 2D angle between grasp approach (front)
    # vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(30):
        return False

    if position[2] < 0.05:  # for safety
        return False
    if position[2] < mass_center[2] - obj_std[2]:  # too low
        return False
    return True


qualifier_dict = {
    "small_cup_qualifier": small_cup_qualifier,
    "cup_qualifier": cup_qualifier,
    "small_cube_qualifier": small_cube_qualifier,
}


def is_qualified(
    grasp: np.ndarray, qualifier: str, min_point: np.ndarray, max_point: np.ndarray
) -> bool:
    """Run the named qualifier function against a grasp pose.

    Args:
        grasp (np.ndarray): A 4x4 grasp transformation matrix.
        qualifier (str): Name of the qualifier function to use.
        min_point (np.ndarray): The 3rd-percentile bounding point of the object.
        max_point (np.ndarray): The 97th-percentile bounding point of the object.

    Returns:
        bool: True if the grasp is qualified.

    Raises:
        KeyError: If the qualifier name is not found.
    """
    if qualifier not in qualifier_dict:
        logger.error(f"There is no such qualifier: {qualifier}")
        raise KeyError
    qualification_method = qualifier_dict[qualifier]
    return qualification_method(grasp, min_point, max_point)
