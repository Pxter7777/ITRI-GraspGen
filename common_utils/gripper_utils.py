import tempfile
import trimesh
import json
import logging
import os
import subprocess
import numpy as np
from common_utils.graspgen_utils import get_left_up_and_front


def pack_grasp_euler(grasp: np.array):
    position = grasp[:3, 3].tolist()

    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
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


def send_cup_grasp_to_robot(grasp: np.array):
    temp_grasp_file = pack_grasp_euler(grasp)
    # Construct the absolute path to quick_grip.py
    # Assuming quick_grip.py is in the project root, one level up from common_utils
    quick_grip_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quick_grip.py"
    )
    command = [
        "/usr/bin/python3",
        quick_grip_path,
        "--input",
        temp_grasp_file,
    ]
    logging.info(f"Executing: {' '.join(command)}")
    subprocess.run(command, check=True)
    os.remove(temp_grasp_file)
    logging.info("quick_grip.py executed successfully and temp file removed.")
