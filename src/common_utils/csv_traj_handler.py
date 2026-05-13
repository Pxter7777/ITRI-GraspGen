"""Load and replay CSV-recorded robot trajectories."""

import csv
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from common_utils.actions_format_checker import ObstacleBound
from common_utils.movesets import SingleRobotMove

logger = logging.getLogger(__name__)


class Mode(Enum):
    """Gripper/movement modes parsed from CSV trajectories.

    Attributes:
        MOVE: Movement mode.
        OPEN: Gripper open mode.
        HALF_OPEN: Gripper half-open mode.
        CLOSE: Gripper close mode.
        CLOSE_TIGHT: Gripper tight-close mode.
    """

    MOVE = 1
    OPEN = 2
    HALF_OPEN = 3
    CLOSE = 4
    CLOSE_TIGHT = 5


status_open = [0, 0, 0]
status_close = [1, 0, 0]
status_half_open = [0, 1, 0]
status_close_tight = [1, 1, 0]


class Movement:
    """A single movement step with a mode and optional joint values.

    Args:
        mode (Mode): The movement mode.
        joint_value (list[float] | None): Optional joint values for the step.
    """

    def __init__(self, mode: Mode, joint_value: list[float] | None = None) -> None:
        self.mode = mode
        self.joints_values = []
        self.joint_value = joint_value


def load_trajectory_from_csv(command: str) -> list[Movement]:
    """Parse a CSV trajectory file into a list of Movement steps.

    Args:
        command (str): The trajectory command name (maps to a CSV filename).

    Returns:
        list[Movement]: The parsed movement steps.

    Raises:
        FileNotFoundError: If the CSV directory or file does not exist.
    """
    base_dir = Path("~").expanduser() / "RobotSnackServing-csv"
    if not base_dir.exists():
        raise FileNotFoundError(
            "~/RobotSnackServing-csv is missing."
            " Is https://github.com/hongalicia/"
            "RobotSnackServing-csv.git cloned into ~/ ?"
        )

    file_path = base_dir / "trajectories" / f"{command}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} doesn't exist.")

    movements: list[Movement] = []
    index = 0
    gripper_prev = None

    with open(file_path, newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if index == 0:
                index += 1
                continue

            joint_values = (row[0][1 : len(row[0]) - 1]).split(", ")
            joint_values_float = []

            for joint in joint_values:
                joint_values_float.append(float(joint))

            if joint_values_float[6:9] == status_open:  # fully open
                if gripper_prev is None or gripper_prev != status_open:
                    move = Movement(Mode.OPEN)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            elif joint_values_float[6:9] == status_close:  # fully close
                if gripper_prev is None or gripper_prev != status_close:
                    move = Movement(Mode.CLOSE)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            elif joint_values_float[6:9] == status_half_open:  # half open
                if gripper_prev is None or gripper_prev != status_half_open:
                    move = Movement(Mode.HALF_OPEN)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            elif joint_values_float[6:9] == status_close_tight:  # close tight
                if gripper_prev is None or gripper_prev != status_close_tight:
                    move = Movement(Mode.CLOSE_TIGHT)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            else:
                move = Movement(Mode.MOVE, joint_values_float[0:6])

            movements.append(move)
            gripper_prev = joint_values_float[6:9]
    return movements


@dataclass
class SpeedParam:
    """Velocity, acceleration, and blend parameters for a trajectory.

    Attributes:
        vel (int): Velocity parameter.
        acc (int): Acceleration parameter.
        blend (int): Blend parameter.
    """

    vel: int = 60
    acc: int = 200
    blend: int = 100


# A defaultdict automatically returns SpeedParam() (the defaults: 40, 20, 100) if a key is missing!
SPEED_PARAM_DICT = defaultdict(SpeedParam)
# Specific overrides
SPEED_PARAM_DICT["spoon_peanuts_down"] = SpeedParam(vel=60, acc=200)
SPEED_PARAM_DICT["spoon_peanuts_up"] = SpeedParam(vel=60, acc=200)
# SPEED_PARAM_DICT["drop_peanuts"] = SpeedParam(vel=60, acc=450)
# SPEED_PARAM_DICT["open_1st_lid"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["open_2nd_lid"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["close_1st_lid"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["close_2nd_lid"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["grab_1st_batter"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["grab_2nd_batter"] = SpeedParam(vel=100, acc=500)
SPEED_PARAM_DICT["drop_1st_batter"] = SpeedParam(vel=40, acc=250)
# SPEED_PARAM_DICT["drop_2nd_batter"] = SpeedParam(vel=50, acc=500)
# SPEED_PARAM_DICT["pour_1st_batter"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["pour_2nt_batter"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["grab_fork"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["drop_fork"] = SpeedParam(vel=100, acc=500)
SPEED_PARAM_DICT["get_1st_waffle"] = SpeedParam(vel=50, acc=500, blend=80)
SPEED_PARAM_DICT["get_2nd_waffle"] = SpeedParam(
    vel=40, acc=500, blend=80
)  # why different than 1st?
# SPEED_PARAM_DICT["close_1st_lid"] = SpeedParam(vel=100, acc=500)
# SPEED_PARAM_DICT["close_2nd_lid"] = SpeedParam(vel=100, acc=500)
SPEED_PARAM_DICT["get_2nd_waffle_top_lid"] = SpeedParam(
    vel=35, acc=500, blend=100
)  # what is this?
# SPEED_PARAM_DICT["drop_waffle"] = SpeedParam(vel=40, acc=500)
# SPEED_PARAM_DICT["go_to_default"] = SpeedParam(vel=100, acc=500)


def run_trajectory(
    command: str,
    obstacles: list[object] | None = None,
    no_need_curobo: bool = False,
) -> list[dict[str, object]]:
    """Convert a CSV trajectory into a list of SingleRobotMove dicts.

    Args:
        command (str): The trajectory command name.
        obstacles (list[object] | None): Optional list of obstacle objects.
        no_need_curobo (bool): Whether to skip cuRobo motion planning.

    Returns:
        list[dict[str, object]]: The list of serialized move dictionaries.

    Raises:
        TypeError: If a MOVE node has a None joint_value.
    """
    if obstacles is None:
        obstacles = []
    nodes = load_trajectory_from_csv(command)
    p = SPEED_PARAM_DICT[command]
    vel, acc, blend = p.vel, p.acc, p.blend
    logger.debug(f"{vel} {acc} {blend}")
    parsed_nodes: list[Movement] = []
    last_is_move = False
    for node in nodes:
        if node.mode == Mode.MOVE:
            if node.joint_value is None:
                raise TypeError(f"Expected list[float], got {type(node.joint_value)}")
            if last_is_move:
                parsed_nodes[-1].joints_values.append(
                    list(np.deg2rad(node.joint_value))
                )
            else:
                parsed_nodes.append(node)
                parsed_nodes[-1].joints_values.append(
                    list(np.deg2rad(node.joint_value))
                )
                last_is_move = True
        else:
            last_is_move = False
            parsed_nodes.append(node)
    nodes = parsed_nodes

    movements: list[SingleRobotMove] = []
    # append positions
    move_at_least_once = False
    for node in nodes:
        print("mode:", node.mode, "joint_value:", node.joint_value)
        if node.mode == Mode.OPEN:
            movements.append(
                SingleRobotMove(type="gripper", grip_type="open", wait_time=1.5)
            )
        elif node.mode == Mode.CLOSE:
            movements.append(
                SingleRobotMove(type="gripper", grip_type="close", wait_time=0.5)
            )
        elif node.mode == Mode.HALF_OPEN:
            movements.append(
                SingleRobotMove(type="gripper", grip_type="half_open", wait_time=0.5)
            )
        elif node.mode == Mode.CLOSE_TIGHT:
            movements.append(
                SingleRobotMove(type="gripper", grip_type="close_tight", wait_time=0.9)
            )
        elif node.mode == Mode.MOVE:
            if not move_at_least_once and not no_need_curobo:
                move_at_least_once = True
                movements.append(
                    SingleRobotMove(
                        type="single_pose_joint_rad",
                        single_pose_joint_rad_goal=node.joints_values[0],
                        vel=60,
                        acc=300,
                        blend=100,
                    )
                )
            movements.append(
                SingleRobotMove(
                    type="sequence_joint_rad",
                    sequence_joint_rad_goals=node.joints_values,
                    vel=vel,
                    acc=acc,
                    blend=blend,
                    no_curobo=True,
                )
            )

    return [asdict(move) for move in movements]


def csv_act(
    command: str,
    obstacles: dict[str, ObstacleBound] | None = None,
    no_need_curobo: bool = False,
) -> list[dict[str, object]]:
    """Run a named CSV trajectory and wrap it as a sendable action list.

    Args:
        command (str): The trajectory command name.
        obstacles (dict[str, ObstacleBound] | None): Optional obstacle bounds.
        no_need_curobo (bool): Whether to skip cuRobo motion planning.

    Returns:
        list[dict[str, object]]: The wrapped action list.
    """
    if obstacles is None:
        obstacles = {}
    return [
        {
            "moves": run_trajectory(command, no_need_curobo=no_need_curobo),
            "obstacles": {k: v.model_dump() for k, v in obstacles.items()},
        }
    ]
