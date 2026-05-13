"""Helpers for packing grasp data and dispatching to the robot gripper."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from common_utils.graspgen_utils import get_left_up_and_front

logger = logging.getLogger(__name__)


def pack_grasp_euler(grasp: np.ndarray) -> str:
    """Pack a 4x4 grasp matrix into a JSON temp file with Euler angles.

    Args:
        grasp (np.ndarray): A 4x4 homogeneous transformation matrix.

    Returns:
        str: Path to the temporary JSON file.
    """
    position = grasp[:3, 3].tolist()

    euler_orientation = np.rad2deg(
        np.array(trimesh.transformations.euler_from_matrix(grasp))
    ).tolist()
    _, _, front = get_left_up_and_front(grasp)
    data = {
        "position": position,
        "euler_orientation": euler_orientation,
        "forward_vec": front.tolist(),
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp"
    ) as tmp:
        json.dump(data, tmp)
        logging.info(f"Grasp data saved to {tmp.name}")
        return tmp.name


def pack_moves(moves: list[dict[str, object]]) -> str:
    """Write a list of move dicts to a JSON temp file.

    Args:
        moves (list[dict[str, object]]): The move dictionaries to serialize.

    Returns:
        str: Path to the temporary JSON file.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="/tmp"
    ) as tmp:
        json.dump(moves, tmp)
        logger.info(f"Grasp data saved to {tmp.name}")
        return tmp.name


def send_cup_grasp_to_robot(grasp: np.ndarray) -> None:
    """Send a single grasp pose to the robot via quick_grip.py.

    Args:
        grasp (np.ndarray): A 4x4 homogeneous transformation matrix.
    """
    temp_grasp_file = pack_grasp_euler(grasp)
    # Construct the absolute path to quick_grip.py
    # Assuming quick_grip.py is in the project root, one level up from common_utils
    quick_grip_path = str(Path(__file__).resolve().parent / "quick_grip.py")
    command = [
        "/usr/bin/python3",
        quick_grip_path,
        "--input",
        temp_grasp_file,
    ]
    logging.info(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)
    Path(temp_grasp_file).unlink()
    logging.info("quick_grip.py executed successfully and temp file removed.")


def send_moves_to_robot(moves: list[dict[str, object]]) -> None:
    """Send a list of moves to the robot via quick_grip2.py.

    Args:
        moves (list[dict[str, object]]): The move dictionaries to send.
    """
    temp_moves_file = pack_moves(moves)
    quick_grip2_path = str(Path(__file__).resolve().parent / "quick_grip2.py")
    command = [
        "/usr/bin/python3",
        quick_grip2_path,
        "--input",
        temp_moves_file,
    ]
    logger.info(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)
    Path(temp_moves_file).unlink()
    logger.info("quick_grip.py executed successfully and temp file removed.")
