import csv
import numpy as np
from enum import Enum
from pathlib import Path
from common_utils.movesets import SingleRobotMove
from dataclasses import asdict


class Mode(Enum):
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
    def __init__(self, mode, joint_value=None):
        self.mode = mode
        self.joints_values = []
        self.joint_value = joint_value


def load_trajectory_from_csv(command: str) -> list[Movement]:
    base_dir = Path("~").expanduser() / "RobotSnackServing"
    if not base_dir.exists():
        raise FileNotFoundError(
            "~/RobotSnackServing is missing. Is https://github.com/hongalicia/RobotSnackServing-csv.git cloned into ~/ ?"
        )

    file_path = base_dir / "trajectories" / f"{command}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} doesn't exist.")

    movements: list[Movement] = []
    index = 0
    gripper_prev = None

    with open(file_path, "r", newline="") as file:
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
                if gripper_prev is None or (
                    gripper_prev is not None and gripper_prev != status_open
                ):
                    move = Movement(Mode.OPEN)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            elif joint_values_float[6:9] == status_close:  # fully close
                if gripper_prev is None or (
                    gripper_prev is not None and gripper_prev != status_close
                ):
                    move = Movement(Mode.CLOSE)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            elif joint_values_float[6:9] == status_half_open:  # half open
                if gripper_prev is None or (
                    gripper_prev is not None and gripper_prev != status_half_open
                ):
                    move = Movement(Mode.HALF_OPEN)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            elif joint_values_float[6:9] == status_close_tight:  # half open
                if gripper_prev is None or (
                    gripper_prev is not None and gripper_prev != status_close_tight
                ):
                    move = Movement(Mode.CLOSE_TIGHT)
                else:
                    move = Movement(Mode.MOVE, joint_values_float[0:6])
            else:
                move = Movement(Mode.MOVE, joint_values_float[0:6])

            movements.append(move)
            gripper_prev = joint_values_float[6:9]
    return movements


def run_trajectory(
    command: str, obstacles: list | None = None, vel=40, acc=20, blend=100
):
    if obstacles is None:
        obstacles = []
    nodes = load_trajectory_from_csv(command)
    parsed_nodes: list[Movement] = []
    print("Number of nodes:", len(nodes))
    last_is_move = False
    for node in nodes:
        if node.mode == Mode.MOVE:
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
                SingleRobotMove(type="gripper", grip_type="close", wait_time=0.9)
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
            if move_at_least_once:
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
            else:
                move_at_least_once = True
                movements.append(
                    SingleRobotMove(
                        type="sequence_joint_rad",
                        sequence_joint_rad_goals=node.joints_values,
                        vel=vel,
                        acc=acc,
                        blend=blend,
                    )
                )
    full_act = {"moves": [asdict(move) for move in movements], "obstacles": obstacles}
    return full_act
