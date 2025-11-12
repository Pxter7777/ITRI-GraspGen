import trimesh
import logging
import numpy as np
from common_utils.qualification import get_left_up_and_front

HOME_SIGNAL = [326.8, -140.2, 212.6, 90.0, 0, 90.0]

logger = logging.getLogger(__name__)


def pick_and_pour_and_put_back(grasp: np.array) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p * 1000 for p in position]
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)

    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL, "wait_time": 0.0})


def grab_and_pour_and_place_back(
    grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p * 1000 for p in position]
    logger.info(position)
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific fixed poses
    if isinstance(args[0], list):
        ready_pour_position = args[0]
    elif isinstance(args[0], str):
        obj_points = scene_data["object_infos"][args[0]]["points"]
        mass_center = np.mean(obj_points, axis=0)
        mass_center = [p * 1000 for p in mass_center]
        # std = np.std(obj_points, axis=0)
        ready_pour_position = [
            mass_center[0] - 175,
            mass_center[1] + 150,
            mass_center[2] + 250,
        ]
    ready_pour_pose = ready_pour_position + [90, 0, 90]
    pour_pose = ready_pour_position + [-90, -55, -90]
    before_grasp_position = [p - f * 60 for p, f in zip(position, front, strict=False)]
    grasp_position = [p + f * 60 for p, f in zip(position, front, strict=False)]
    after_grasp_position = grasp_position[:2] + [grasp_position[2] + 250]

    release_position = grasp_position[:2] + [grasp_position[2] + 5]
    after_release_position = before_grasp_position
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append(
        {
            "type": "move_arm",
            "goal": before_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "move_arm",
            "goal": grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "goal": "grab"})
    moves.append(
        {
            "type": "move_arm",
            "goal": after_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "move_arm", "goal": ready_pour_pose, "wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": pour_pose, "wait_time": 1.0})
    moves.append({"type": "move_arm", "goal": ready_pour_pose, "wait_time": 0.0})
    moves.append(
        {
            "type": "move_arm",
            "goal": after_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "move_arm",
            "goal": release_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "goal": "release"})
    moves.append(
        {
            "type": "move_arm",
            "goal": after_release_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    return moves

def grab_and_pour_and_place_back_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> dict:
    obstacles = []
    for obj_name in scene_data["object_infos"]:
        if target_name != obj_name:
            obstacles.append(
                {
                    "mass_center": list(
                        np.mean(scene_data["object_infos"][obj_name]["points"], axis=0)
                    ),
                    "std": list(
                        np.std(scene_data["object_infos"][obj_name]["points"], axis=0)
                    ),
                }
            )
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific fixed poses
    if isinstance(args[0], list):
        ready_pour_position = args[0]
    elif isinstance(args[0], str):
        obj_points = scene_data["object_infos"][args[0]]["points"]
        mass_center = np.mean(obj_points, axis=0)
        # std = np.std(obj_points, axis=0)
        ready_pour_position = [
            mass_center[0] - 0.175,
            mass_center[1] + 0.150,
            # mass_center[2] + 0.250,
            mass_center[2] + 0.150,
        ]
    ready_pour_pose = ready_pour_position + [0.5, 0.5, 0.5, 0.5]
    pour_pose = ready_pour_position + [-0.271, 0.653, -0.271, 0.653]
    before_grasp_position = [
        p - f * 0.100 for p, f in zip(position, front, strict=False)
    ]
    grasp_position = [p + f * 0.060 for p, f in zip(position, front, strict=False)]
    # after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.250]

    release_position = grasp_position[:2] + [grasp_position[2] + 0.005]
    after_release_position = before_grasp_position
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "grip_type": "close", "wait_time": 1.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append({"type": "arm", "goal": ready_pour_pose, "wait_time": 0.0})
    moves.append({"type": "arm", "goal": pour_pose, "wait_time": 1.0})
    moves.append({"type": "arm", "goal": ready_pour_pose, "wait_time": 0.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append(
        {
            "type": "arm",
            "goal": release_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {
            "type": "arm",
            "goal": after_release_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )

    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def grab_and_pour_and_place_back_curobo_by_rotation(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> dict:
    obstacles = []
    for obj_name in scene_data["object_infos"]:
        if target_name != obj_name:
            obstacles.append(
                {
                    "mass_center": list(
                        np.mean(scene_data["object_infos"][obj_name]["points"], axis=0)
                    ),
                    "std": list(
                        np.std(scene_data["object_infos"][obj_name]["points"], axis=0)
                    ),
                }
            )
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()

    # Grasp Position
    before_grasp_position = [
        p - f * 0.060 for p, f in zip(position, front, strict=False)
    ]
    grasp_position = [p + f * 0.060 for p, f in zip(position, front, strict=False)]

    # specific fixed poses
    if isinstance(args[0], list):
        ready_pour_position = args[0]
        ready_pour_rotation = [0.5, 0.5, 0.5, 0.5]
        pour_rotation = [-0.271, 0.653, -0.271, 0.653]
    elif isinstance(args[0], str):
        obj_points = scene_data["object_infos"][args[0]]["points"]
        mass_center = np.mean(obj_points, axis=0)
        ## Ready pour position
        grasp_angle = np.arctan2(grasp_position[1], grasp_position[0])
        target_angle = np.arctan2(mass_center[1], mass_center[0])

        # compute radius using mass_center[0] and mass_center[1]
        radius = np.linalg.norm(mass_center[:2]) - 0.10

        angle_diff = target_angle - grasp_angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        if angle_diff < 0:  # Clockwise
            goal_angle = target_angle + np.deg2rad(5)
        else:  # Counter-clockwise
            goal_angle = target_angle - np.deg2rad(5)
        ready_pour_position = [
            radius * np.cos(goal_angle),
            radius * np.sin(goal_angle),
            mass_center[2] + 0.250,
        ]
        q_z_rotation = trimesh.transformations.quaternion_about_axis(
            goal_angle, [0, 0, 1]
        )
        q_base = np.array([0.5, 0.5, 0.5, 0.5])
        ready_pour_rotation = trimesh.transformations.quaternion_multiply(
            q_z_rotation, q_base
        ).tolist()
        if angle_diff < 0:  # Clockwise
            pour_angle = np.deg2rad(135)
        else:  # Counter-clockwise
            pour_angle = - np.deg2rad(135)
        # apply pour_angle on ready_pour_rotation using vector[mass_center[0], mass_center[1], 0] as axis:
        pour_axis = np.array([ready_pour_position[0], ready_pour_position[1], 0])
        axis_norm = np.linalg.norm(pour_axis)
        if axis_norm > 1e-6:  # Avoid division by zero
            pour_axis /= axis_norm
            q_pour = trimesh.transformations.quaternion_about_axis(
                pour_angle, pour_axis
            )
            pour_rotation = trimesh.transformations.quaternion_multiply(
                q_pour, np.array(ready_pour_rotation)
            ).tolist()
        else:
            # Axis is zero, cannot determine pour direction. Fallback to a default pour.
            raise ValueError(f"axis_norm={axis_norm}")
            #pour_rotation = [-0.271, 0.653, -0.271, 0.653]

    ready_pour_pose = ready_pour_position + ready_pour_rotation
    pour_pose = ready_pour_position + pour_rotation
    
    # after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.250]

    release_position = grasp_position[:2] + [grasp_position[2] + 0.005]
    after_release_position = before_grasp_position
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append(
        {
            "type": "arm",
            "goal": before_grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "arm",
            "goal": grasp_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "grip_type": "close", "wait_time": 1.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append({"type": "arm", "goal": ready_pour_pose, "wait_time": 0.0})
    moves.append({"type": "arm", "goal": pour_pose, "wait_time": 1.0})
    moves.append({"type": "arm", "goal": ready_pour_pose, "wait_time": 0.0})
    # moves.append(
    #     {
    #         "type": "arm",
    #         "goal": after_grasp_position + quaternion_orientation,
    #         "wait_time": 0.0,
    #     }
    # )
    moves.append(
        {
            "type": "arm",
            "goal": release_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "grip_type": "open", "wait_time": 1.0})
    moves.append(
        {
            "type": "arm",
            "goal": after_release_position + quaternion_orientation,
            "wait_time": 0.0,
        }
    )

    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act


def grab_and_drop(grasp: np.array, args: list, scene_data: dict) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p * 1000 for p in position]
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific drop point
    drop_pose = args[0]

    before_grasp_position = [p - f * 60 for p, f in zip(position, front, strict=False)]
    grasp_position = [p + f * 50 for p, f in zip(position, front, strict=False)]
    after_grasp_position = grasp_position[:2] + [grasp_position[2] + 200]
    # forward_signal = HOME_SIGNAL
    # forward_signal[0] += 200
    # moves.append({"type": "move_arm", "goal": forward_signal, "wait_time": 0.0})
    moves.append(
        {
            "type": "move_arm",
            "goal": before_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append(
        {
            "type": "move_arm",
            "goal": grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "gripper", "goal": "grab"})
    moves.append(
        {
            "type": "move_arm",
            "goal": after_grasp_position + euler_orientation,
            "wait_time": 0.0,
        }
    )
    moves.append({"type": "move_arm", "goal": drop_pose, "wait_time": 0.5})
    moves.append({"type": "gripper", "goal": "release"})
    moves.append({"type": "move_arm", "goal": HOME_SIGNAL, "wait_time": 0.0})
    return moves


def move_to(grasp: np.array, args: list, scene_data: dict) -> list[dict]:
    pose = args[0] + [90, 0, 90]
    moves = []
    moves.append({"type": "move_arm", "goal": pose, "wait_time": 0.0})
    return moves


def move_to_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    pose = args[0] + [0.5, 0.5, 0.5, 0.5]
    obstacles = []
    for obj_name in scene_data["object_infos"]:
        obstacles.append(
            {
                "mass_center": list(
                    np.mean(scene_data["object_infos"][obj_name]["points"], axis=0)
                ),
                "std": list(
                    np.std(scene_data["object_infos"][obj_name]["points"], axis=0)
                ),
            }
        )
    moves = []
    moves.append({"type": "arm", "goal": pose, "wait_time": 0.0})
    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act

def joints_rad_move_to_curobo(
    target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    joints_goal = args[0]
    obstacles = []
    for obj_name in scene_data["object_infos"]:
        obstacles.append(
            {
                "mass_center": list(
                    np.mean(scene_data["object_infos"][obj_name]["points"], axis=0)
                ),
                "std": list(
                    np.std(scene_data["object_infos"][obj_name]["points"], axis=0)
                ),
            }
        )
    moves = []
    moves.append({"type": "arm", "joints_goal": joints_goal, "wait_time": 0.0})
    full_act = {"moves": moves, "obstacles": obstacles}
    return full_act
    


action_dict = {
    "grab_and_pour_and_place_back": grab_and_pour_and_place_back,
    "grab_and_pour_and_place_back_curobo": grab_and_pour_and_place_back_curobo_by_rotation,
    "grab_and_drop": grab_and_drop,
    "move_to": move_to,
    "move_to_curobo": move_to_curobo,
    "joints_rad_move_to_curobo": joints_rad_move_to_curobo,
}


def act(action: str, grasp: np.array, args: list, scene_data: dict) -> list[dict]:
    if action not in action_dict:
        logger.error(f"There is no such action: {action}")
    action_method = action_dict[action]
    return action_method(grasp, args, scene_data)


def act_with_name(
    action: str, target_name: str, grasp: np.array, args: list, scene_data: dict
) -> list[dict]:
    if action not in action_dict:
        logger.error(f"There is no such action: {action}")
    action_method = action_dict[action]
    return action_method(target_name, grasp, args, scene_data)
