# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import glob
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

import numpy as np
import torch

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a scene point cloud after GraspGen inference, for entire scene"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="",
        help="Directory containing JSON files with point cloud data",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="",
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.80,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=-1,
        help="Number of top grasps to return when return_topk is True",
    )
    return parser.parse_args()


def start_meshcat_server():
    """Starts the meshcat-server and registers a cleanup function."""
    print("Starting meshcat-server...")
    meshcat_server_process = subprocess.Popen(
        "meshcat-server", shell=True, preexec_fn=os.setsid
    )

    @atexit.register
    def cleanup_meshcat_server():
        if meshcat_server_process.poll() is None:
            print("Terminating meshcat-server...")
            os.killpg(os.getpgid(meshcat_server_process.pid), signal.SIGTERM)

    time.sleep(2)  # Wait for server to start
    return meshcat_server_process


def open_meshcat_url(url):
    """Opens the given URL in a web browser."""
    print(f"\n--- Meshcat Visualizer ---\nURL: {url}\n--------------------------\n")
    try:
        system = platform.system()
        if system == "Linux":
            release_info = platform.release().lower()
            if "microsoft" in release_info or "wsl" in release_info:
                subprocess.run(
                    ["powershell.exe", "-c", f'Start-Process "{url}"'],
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(["xdg-open", url], stderr=subprocess.DEVNULL)
        else:
            webbrowser.open(url)
    except Exception:
        pass  # If opening fails, the user has the URL printed.


def validate_args(args):
    """Validates the command-line arguments."""
    if args.sample_data_dir == "":
        raise ValueError("sample_data_dir is required")
    if args.gripper_config == "":
        raise ValueError("gripper_config is required")
    if not os.path.exists(args.sample_data_dir):
        raise FileNotFoundError(
            f"sample_data_dir {args.sample_data_dir} does not exist"
        )
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100


def process_and_visualize_scene(vis, json_file):
    """Loads scene data, processes point clouds, and visualizes them."""
    print(json_file)
    vis.delete()

    with open(json_file, "rb") as f:
        data = json.load(f)

    obj_pc = np.array(data["object_info"]["pc"])
    obj_pc_color = np.array(data["object_info"]["pc_color"])

    full_pc_key = "pc_color" if "pc_color" in data["scene_info"] else "full_pc"
    xyz_scene = np.array(data["scene_info"][full_pc_key])[0]
    xyz_scene_color = np.array(data["scene_info"]["img_color"]).reshape(1, -1, 3)[
        0, :, :
    ]

    VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]
    mask_within_bounds = np.all((xyz_scene > VIZ_BOUNDS[0]), 1)
    mask_within_bounds = np.logical_and(
        mask_within_bounds, np.all((xyz_scene < VIZ_BOUNDS[1]), 1)
    )
    xyz_scene = xyz_scene[mask_within_bounds]
    xyz_scene_color = xyz_scene_color[mask_within_bounds]

    visualize_pointcloud(vis, "pc_scene", xyz_scene, xyz_scene_color, size=0.0025)

    obj_pc, _ = point_cloud_outlier_removal(torch.from_numpy(obj_pc))
    obj_pc = obj_pc.cpu().numpy()

    visualize_pointcloud(vis, "pc_obj", obj_pc, obj_pc_color, size=0.005)

    return obj_pc


def generate_and_visualize_grasps(vis, obj_pc, grasp_sampler, gripper_name, args):
    """Generates grasps and visualizes them."""
    method = "GraspGen"
    grasps, grasp_conf = GraspGenSampler.run_inference(
        obj_pc,
        grasp_sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk_num_grasps,
    )

    if len(grasps) > 0:
        grasp_conf = grasp_conf.cpu().numpy()
        grasps = grasps.cpu().numpy()
        grasps[:, 3, 3] = 1
        print(
            f"[{method}] Scores with min {grasp_conf.min():.3f} and max {grasp_conf.max():.3f}"
        )

        map_method2color = {"GraspGen": [0, 185, 0]}
        for j, grasp in enumerate(grasps):
            color = map_method2color[method]
            visualize_grasp(
                vis,
                f"{method}/{j:03d}/grasp",
                grasp,
                color=color,
                gripper_name=gripper_name,
                linewidth=1.5,
            )
        input("Press Enter to continue to next scene...")
    else:
        print(f"[{method}] No grasps found! Skipping to next scene...")


def create_control_panel():
    """Creates and runs the tkinter control panel."""
    root = tk.Tk()
    root.title("Control Panel")
    button = tk.Button(root, text="Print Hello", command=lambda: print("Hello World"))
    button.pack(padx=20, pady=20)
    root.mainloop()


def main():
    """Main function to run the GraspGen demo."""
    start_meshcat_server()
    open_meshcat_url("http://127.0.0.1:7000/static/")

    gui_thread = Thread(target=create_control_panel, daemon=True)
    gui_thread.start()

    args = parse_args()
    validate_args(args)

    json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))

    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    vis = create_visualizer()

    for json_file in json_files:
        obj_pc = process_and_visualize_scene(vis, json_file)
        generate_and_visualize_grasps(vis, obj_pc, grasp_sampler, gripper_name, args)


if __name__ == "__main__":
    main()
