import trimesh
import logging
import numpy as np
from common_utils.qualification import get_left_up_and_front
from dataclasses import dataclass, asdict, field
from typing import Literal

HOME_SIGNAL = [326.8, -140.2, 212.6, 90.0, 0, 90.0]

logger = logging.getLogger(__name__)

MoveType = Literal[
    "gripper",
    "sequence_joint_rad",
    "single_pose_meter_quaternion",
    "single_pose_joint_rad",
]
GripType = Literal["open", "close", "half_open", "close_tight"]


@dataclass
class SingleRobotMove:
    """
    This class will be used for all who handles robot move command, since I figured that many info can be reused.
    So, basically, the scripts, including scripts/mia_server.py, isaacsim2real/sync_with_ROS2.py, ROS2_server/gripper_server.
    They don't need to fully translate it. They take what they need, and modify part of it.
    For example, sync_with_ROS2.py may accept single_pose_meter_quaternion type, and transfer it to sequence_joint_rad type based on the info.
    """

    type: MoveType  # "gripper", "sequence_joint_rad", "single_pose_meter_quaternion", "single_pose_joint_rad"
    """
    There are four types
    "gripper": modify gripper state, it won't move the arm.
    "sequence_joint_rad": move the arm given a list of jointstates rad.
    "single_pose_meter_quaternion": move the arm given a single gripper pose(or called end effector/EE), with meter and quaternion.
    "single_pose_joint_rad": move the arm given a single jointstate rad.
    """
    grip_type: GripType | None = (
        None  # "open", "close", only useful when type is "gripper".
    )
    wait_time: float = 0.0
    # vel, acc, blend will only have effect under "sequence_joint_rad"
    vel: int = 60
    acc: int = 300
    blend: int = 100
    no_curobo: bool = False
    no_obstacles: bool = False
    ignore_obstacles: list[str] = field(default_factory=list)
    sequence_joint_rad_goals: list[list[float]] | None = None
    single_pose_meter_quaternion_goal: list[float] | None = None
    single_pose_joint_rad_goal: list[float] | None = None


def grab_and_pour_and_place_back(
    grasp: np.ndarray, args: list, scene_data: dict
) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p * 1000 for p in position]
    logger.info(position)
    euler_rad = trimesh.transformations.euler_from_matrix(grasp)
    euler_orientation: list[float] = np.rad2deg(euler_rad).tolist()  # type: ignore
    # euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    # euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific fixed poses
    ready_pour_position: list[float] = []
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
    target_name: str, grasp: np.ndarray, args: list, scene_data: dict
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
    ready_pour_position: list[float] = []
    if isinstance(args[0], list):
        ready_pour_position: list[float] = args[0]
    elif isinstance(args[0], str):
        obj_points = scene_data["object_infos"][args[0]]["points"]
        mass_center = np.mean(obj_points, axis=0)
        # std = np.std(obj_points, axis=0)
        ready_pour_position: list[float] = [
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
    target_name: str, grasp: np.ndarray, args: list, scene_data: dict
) -> dict:
    obstacles = scene_data["obstacles"]
    moves: list[SingleRobotMove] = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    logger.debug(position)
    quaternion_orientation = list(trimesh.transformations.quaternion_from_matrix(grasp))
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()

    # Grasp Position
    before_grasp_position = [
        p - f * 0.050 for p, f in zip(position, front, strict=False)
    ]
    grasp_position = [p + f * 0.048 for p, f in zip(position, front, strict=False)]

    # specific fixed poses
    if isinstance(args[0], list):
        middle_point = np.array(args[0])
    elif isinstance(args[0], str):
        obj_bounding_box = scene_data["obstacles"][args[0]]
        middle_point = np.mean(
            [
                obj_bounding_box["max"],
                obj_bounding_box["min"],
            ],
            axis=0,
        )
    ## Ready pour position
    grasp_angle = np.arctan2(grasp_position[1], grasp_position[0])
    target_angle = np.arctan2(middle_point[1], middle_point[0])

    # compute radius using mass_center[0] and mass_center[1]
    radius = np.linalg.norm(middle_point[:2]) - 0.20

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
        middle_point[2] + 0.200,
    ]
    q_z_rotation = trimesh.transformations.quaternion_about_axis(goal_angle, [0, 0, 1])
    q_y_rotation = trimesh.transformations.quaternion_about_axis(
        -np.arcsin(front[2]), [0, 1, 0]
    )
    q_base = np.array([0.5, 0.5, 0.5, 0.5])
    q_base_tilt = trimesh.transformations.quaternion_multiply(
        q_y_rotation, q_base
    ).tolist()
    ready_pour_rotation = trimesh.transformations.quaternion_multiply(
        q_z_rotation, q_base_tilt
    ).tolist()
    if angle_diff < 0:  # Clockwise
        pour_angle = np.deg2rad(160)
    else:  # Counter-clockwise
        pour_angle = -np.deg2rad(160)
    # apply pour_angle on ready_pour_rotation using vector[mass_center[0], mass_center[1], 0] as axis:
    local_rotation_axis = [0, 0, 1]

    q_pour = trimesh.transformations.quaternion_about_axis(
        pour_angle, local_rotation_axis
    )
    ready_pour_pose = ready_pour_position + ready_pour_rotation

    pour_rotation = trimesh.transformations.quaternion_multiply(
        np.array(ready_pour_rotation), q_pour
    ).tolist()

    pour_pose3 = ready_pour_position + pour_rotation

    # after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.250]

    release_position = grasp_position[:2] + [grasp_position[2] + 0.005]
    after_release_position = before_grasp_position
    # moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=before_grasp_position
            + quaternion_orientation,
        )
    )
    moves.append(
        # {
        #     "type": "arm",
        #     "goal": grasp_position + quaternion_orientation,
        #     "wait_time": 0.0,
        #     "no_obstacles": "yesyesyes",
        #     "no_curobo": True,
        #     "ignore_obstacles": [target_name],
        # }
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=grasp_position + quaternion_orientation,
            no_curobo=True,
            no_obstacles=True,
        )
    )
    moves.append(SingleRobotMove(type="gripper", grip_type="close", wait_time=1.0))
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=ready_pour_pose,
            ignore_obstacles=[target_name],
        )
    )

    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=pour_pose3,
            no_curobo=True,
            wait_time=1.0,
        )
    )
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=ready_pour_pose,
            no_curobo=True,
        )
    )
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=release_position + quaternion_orientation,
            ignore_obstacles=[target_name],
        )
    )
    moves.append(SingleRobotMove(type="gripper", grip_type="open", wait_time=1.0))
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=after_release_position
            + quaternion_orientation,
            no_curobo=True,
            no_obstacles=True,
        )
    )
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=after_release_position[:2]
            + [after_release_position[2] + 0.1]
            + quaternion_orientation,
            no_curobo=True,
            no_obstacles=True,
        )
    )

    full_act = {"moves": [asdict(move) for move in moves], "obstacles": obstacles}
    return full_act


def grab_and_drop(grasp: np.ndarray, args: list, scene_data: dict) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p * 1000 for p in position]
    euler_rad = trimesh.transformations.euler_from_matrix(grasp)
    euler_orientation: list[float] = np.rad2deg(euler_rad).tolist()  # type: ignore
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


def move_to(grasp: np.ndarray, args: list, scene_data: dict) -> list[dict]:
    pose = args[0] + [90, 0, 90]
    moves = []
    moves.append({"type": "move_arm", "goal": pose, "wait_time": 0.0})
    return moves


def move_to_curobo(
    target_name: str, grasp: np.ndarray, args: list, scene_data: dict
) -> dict:
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


def joints_rad_move_to_curobo(args: list, scene_data: dict) -> dict:
    joints_goal = args[0]
    obstacles = scene_data["obstacles"]
    moves: list[SingleRobotMove] = []
    moves.append(
        SingleRobotMove(
            type="single_pose_joint_rad", single_pose_joint_rad_goal=joints_goal
        )
    )
    # moves.append(SingleRobotMove(type="gripper"))
    full_act = {"moves": [asdict(move) for move in moves], "obstacles": obstacles}
    return full_act


def open_grip() -> dict:
    obstacles = []
    moves = []
    moves.append(SingleRobotMove(type="gripper", grip_type="open", wait_time=1.0))
    full_act = {"moves": [asdict(move) for move in moves], "obstacles": obstacles}
    return full_act


action_dict = {
    "grab_and_pour_and_place_back": grab_and_pour_and_place_back,
    "grab_and_pour_and_place_back_curobo": grab_and_pour_and_place_back_curobo_by_rotation,
    "grab_and_drop": grab_and_drop,
    "move_to": move_to,
    "move_to_curobo": move_to_curobo,
    "joints_rad_move_to_curobo": joints_rad_move_to_curobo,
    "open_grip": open_grip,
}


def act(action: str, grasp: np.ndarray, args: list, scene_data: dict) -> list[dict]:
    if action not in action_dict:
        logger.error(f"There is no such action: {action}")
    action_method = action_dict[action]
    return action_method(grasp, args, scene_data)


def act_with_name(
    action: str,
    args: list | None = None,
    grasps: list[np.ndarray] | None = None,
    scene_data: dict | None = None,
    target_name: str | None = None,
) -> list[dict]:
    if action == "grab_and_pour_and_place_back_curobo":
        if grasps is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo doesn't allow grasps to be None"
            )
        if target_name is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo doesn't allow target_name to be None"
            )
        if scene_data is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo doesn't allow scene_data to be None"
            )
        if args is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo doesn't allow args to be None"
            )
        return [
            grab_and_pour_and_place_back_curobo_by_rotation(
                target_name, grasp, args, scene_data
            )
            for grasp in grasps
        ]
    elif action == "joints_rad_move_to_curobo":
        if args is None:
            raise ValueError("joints_rad_move_to_curobo doesn't allow args to be None")
        if scene_data is None:
            raise ValueError(
                "joints_rad_move_to_curobo doesn't allow scene_data to be None"
            )
        return [joints_rad_move_to_curobo(args, scene_data)]
    elif action == "open_grip":
        return [open_grip()]
    else:
        raise ValueError(f"There is no such action: {action}")
