"""Robot move primitives and action dispatch for the grasping pipeline."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import trimesh

from common_utils.qualification import get_left_up_and_front
from common_utils.scene_data import SceneData

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
    """Unified robot move command shared across all three processes.

    Scripts like mia_server.py, sync_with_ROS2.py, and gripper_server.py
    each consume the fields they need and may transform the move type
    (e.g. single_pose_meter_quaternion to sequence_joint_rad).

    Attributes:
        type (MoveType): The move type. There are four types:
            "gripper": modify gripper state, it won't move the arm.
            "sequence_joint_rad": move the arm given a list of jointstates rad.
            "single_pose_meter_quaternion": move the arm given a single
                gripper pose (end effector/EE), meter and quaternion.
            "single_pose_joint_rad": move the arm given a single jointstate rad.
        grip_type (GripType | None): Gripper action, only used when type
            is "gripper".
        wait_time (float): Time to wait after the move in seconds.
        vel (int): Velocity parameter for sequence_joint_rad moves.
        acc (int): Acceleration parameter for sequence_joint_rad moves.
        blend (int): Blend parameter for sequence_joint_rad moves.
        no_curobo (bool): Whether to skip cuRobo motion planning.
        no_obstacles (bool): Whether to ignore all obstacles.
        ignore_obstacles (list[str]): Names of obstacles to ignore.
        sequence_joint_rad_goals (list[list[float]] | None): Joint goals
            for sequence moves.
        single_pose_meter_quaternion_goal (list[float] | None): Pose goal
            as position and quaternion.
        single_pose_joint_rad_goal (list[float] | None): Joint goal for
            single pose moves.
    """

    type: MoveType
    grip_type: GripType | None = (
        None  # "open", "close", only useful when type is "gripper".
    )
    wait_time: float = 0.0
    # vel, acc, blend will only have effect under "sequence_joint_rad"
    vel: int = 40
    acc: int = 20
    blend: int = 100
    no_curobo: bool = False
    no_obstacles: bool = False
    ignore_obstacles: list[str] = field(default_factory=list)
    sequence_joint_rad_goals: list[list[float]] | None = None
    single_pose_meter_quaternion_goal: list[float] | None = None
    single_pose_joint_rad_goal: list[float] | None = None


def grab_and_pour_and_place_back_curobo_by_rotation(
    target_name: str,
    grasp: np.ndarray,
    args: list[object],
    scene_data: SceneData,
) -> dict[str, object]:
    """Generate a grab-pour-release move sequence using rotational pouring.

    Args:
        target_name (str): Name of the target object to grasp.
        grasp (np.ndarray): A 4x4 grasp transformation matrix.
        args (list[object]): Additional arguments (pour target position or name).
        scene_data (SceneData): Scene data with obstacle information.

    Returns:
        dict[str, object]: The full action dictionary with moves and obstacles.

    Raises:
        TypeError: If args[0] is neither a list nor a string.
        ValueError: If the pour axis norm is near zero.
    """
    obstacles = scene_data.obstacle_infos
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
        obj_bounding_box = scene_data.obstacle_infos[args[0]]
        middle_point = np.array(
            np.mean([obj_bounding_box.max, obj_bounding_box.min], axis=0)
        )
    else:
        raise TypeError(f"Unexpected args[0] type: {type(args[0])}")
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
        pour_angle = np.deg2rad(45)
    else:  # Counter-clockwise
        pour_angle = -np.deg2rad(45)
    # apply pour_angle on ready_pour_rotation using
    # vector[mass_center[0], mass_center[1], 0] as axis:
    pour_axis = np.array([ready_pour_position[0], ready_pour_position[1], 0])
    axis_norm = np.linalg.norm(pour_axis)
    if axis_norm > 1e-6:  # Avoid division by zero
        pour_axis /= axis_norm
        q_pour = trimesh.transformations.quaternion_about_axis(pour_angle, pour_axis)
        pour_rotation1 = trimesh.transformations.quaternion_multiply(
            q_pour, np.array(ready_pour_rotation)
        ).tolist()
        pour_rotation2 = trimesh.transformations.quaternion_multiply(
            q_pour, np.array(pour_rotation1)
        ).tolist()
        pour_rotation3 = trimesh.transformations.quaternion_multiply(
            q_pour, np.array(pour_rotation2)
        ).tolist()
    else:
        # Axis is zero, cannot determine pour direction. Fallback to a default pour.
        raise ValueError(f"axis_norm={axis_norm}")
        # pour_rotation = [-0.271, 0.653, -0.271, 0.653]

    ready_pour_pose = ready_pour_position + ready_pour_rotation
    # pour_pose1 = ready_pour_position + pour_rotation1
    # pour_pose2 = ready_pour_position + pour_rotation2
    pour_pose3 = ready_pour_position + pour_rotation3

    # after_grasp_position = grasp_position[:2] + [grasp_position[2] + 0.250]

    release_position = [*grasp_position[:2], grasp_position[2] + 0.005]
    after_release_position = before_grasp_position
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=before_grasp_position
            + quaternion_orientation,
        )
    )
    moves.append(
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
            single_pose_meter_quaternion_goal=(
                release_position + quaternion_orientation
            ),
            ignore_obstacles=[target_name],
        )
    )
    moves.append(SingleRobotMove(type="gripper", grip_type="open", wait_time=1.0))
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=(
                after_release_position + quaternion_orientation
            ),
            no_curobo=True,
            no_obstacles=True,
        )
    )
    moves.append(
        SingleRobotMove(
            type="single_pose_meter_quaternion",
            single_pose_meter_quaternion_goal=[
                *after_release_position[:2],
                after_release_position[2] + 0.1,
                *quaternion_orientation,
            ],
            no_curobo=True,
            no_obstacles=True,
        )
    )

    full_act: dict[str, object] = {
        "moves": [asdict(move) for move in moves],
        "obstacles": {name: obs.model_dump() for name, obs in obstacles.items()},
    }
    return full_act


def joints_rad_move_to_curobo(
    args: list[object], scene_data: SceneData
) -> dict[str, object]:
    """Generate a single joint-space move through cuRobo.

    Args:
        args (list[object]): List whose first element is the joint goal.
        scene_data (SceneData): Scene data with obstacle information.

    Returns:
        dict[str, object]: The full action dictionary with moves and obstacles.
    """
    joints_goal = args[0]
    obstacles = scene_data.obstacle_infos
    moves: list[SingleRobotMove] = []
    moves.append(
        SingleRobotMove(
            type="single_pose_joint_rad",
            single_pose_joint_rad_goal=joints_goal,  # type: ignore[reportArgumentType]
        )
    )
    moves.append(SingleRobotMove(type="gripper"))
    full_act: dict[str, object] = {
        "moves": [asdict(move) for move in moves],
        "obstacles": {name: obs.model_dump() for name, obs in obstacles.items()},
    }
    return full_act


def open_grip() -> dict[str, object]:
    """Generate a gripper-open command.

    Returns:
        dict[str, object]: The full action dictionary with moves and obstacles.
    """
    obstacles: list[object] = []
    moves: list[SingleRobotMove] = []
    moves.append(SingleRobotMove(type="gripper", grip_type="open", wait_time=1.0))
    full_act: dict[str, object] = {
        "moves": [asdict(move) for move in moves],
        "obstacles": obstacles,
    }
    return full_act


action_dict = {
    "grab_and_pour_and_place_back_curobo": (
        grab_and_pour_and_place_back_curobo_by_rotation
    ),
    "joints_rad_move_to_curobo": joints_rad_move_to_curobo,
    "open_grip": open_grip,
}


def act_with_name(
    action: str,
    args: list[object] | None = None,
    grasps: list[np.ndarray] | None = None,
    scene_data: SceneData | None = None,
    target_name: str | None = None,
) -> list[dict[str, object]]:
    """Dispatch a named action to the appropriate move generator.

    Args:
        action (str): The action name to dispatch.
        args (list[object] | None): Optional arguments for the action.
        grasps (list[np.ndarray] | None): Optional list of grasp matrices.
        scene_data (SceneData | None): Optional scene data.
        target_name (str | None): Optional target object name.

    Returns:
        list[dict[str, object]]: A list of action dictionaries.

    Raises:
        ValueError: If required arguments are None or the action is unknown.
    """
    if action == "grab_and_pour_and_place_back_curobo":
        if grasps is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo doesn't allow grasps to be None"
            )
        if target_name is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo"
                " doesn't allow target_name to be None"
            )
        if scene_data is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo"
                " doesn't allow scene_data to be None"
            )
        if args is None:
            raise ValueError(
                "grab_and_pour_and_place_back_curobo doesn't allow args to be None"
            )
        return [
            grab_and_pour_and_place_back_curobo_by_rotation(
                target_name,
                grasp,
                args,
                scene_data,
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
        return [
            joints_rad_move_to_curobo(
                args,
                scene_data,
            )
        ]
    elif action == "open_grip":
        return [open_grip()]
    else:
        raise ValueError(f"There is no such action: {action}")
