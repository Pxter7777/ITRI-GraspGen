import argparse
import json
import os
import platform
import signal
import subprocess
import time
import webbrowser
import atexit
import tkinter as tk
from threading import Thread
from pathlib import Path

import numpy as np
import pye57

from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_pointcloud,
)
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]

def transform(original_pointcloud, transformation_matrix):
    original_pc_homogeneous = np.hstack(
        (
            original_pointcloud,
            np.ones((original_pointcloud.shape[0], 1)),
        )
    )
    transformed_object_pc_homogeneous = (
        transformation_matrix @ original_pc_homogeneous.T
    ).T
    transformed_pointcloud = transformed_object_pc_homogeneous[:, :3]
    return transformed_pointcloud


def silent_transform_multiple_obj_with_name_dict(scene_data: dict) -> dict:
    config_filepath = PROJECT_ROOT_DIR / "configs" / "transform_config.json"
    with open(config_filepath, "rb") as f:
        transform_data = json.load(f)
    # Build Transformation matrix
    translation_matrix = np.array(
        [
            [1, 0, 0, transform_data["tx"]],
            [0, 1, 0, transform_data["ty"]],
            [0, 0, 1, transform_data["tz"]],
            [0, 0, 0, 1],
        ]
    )
    rotation = R.from_euler(
        "xyz",
        [transform_data["rr"], transform_data["rp"], transform_data["ry"]],
        degrees=True,
    )
    rotation_matrix = np.identity(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    transformation = translation_matrix @ rotation_matrix

    # actual transformation
    for name in scene_data["object_infos"]:
        scene_data["object_infos"][name]["points"] = transform(
            scene_data["object_infos"][name]["points"], transformation
        )
    scene_data["scene_info"]["pc_color"] = [
        transform(np.array(scene_data["scene_info"]["pc_color"][0]), transformation)
    ]
    return scene_data

