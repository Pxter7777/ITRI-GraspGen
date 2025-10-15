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
import trimesh

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal_with_color
from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format
from src import config


class AppState:
    def __init__(self):
        self.grasps = None
        self.qualified_grasps = None
        self.selected_grasp_pool = None
        self.grasp_conf = None
        self.current_grasp_index = 0
        self.display_unqualified_grasps = False


app_state = AppState()


class ControlPanel:
    def __init__(
        self,
        root,
        vis,
        json_files,
        grasp_sampler,
        gripper_name,
        args,
    ):
        self.root = root
        self.vis = vis
        self.json_files = json_files
        self.grasp_sampler = grasp_sampler
        self.gripper_name = gripper_name
        self.args = args
        self.root.title("Control Panel")

        self.custom_filename_var = tk.BooleanVar()
        self.display_disqualified_var = tk.BooleanVar(value=False)
        self.filename_entry_var = tk.StringVar()
        self.selected_file = tk.StringVar()

        # Dropdown for JSON files
        self.selected_file.set(json_files[0] if json_files else "")
        self.json_dropdown = tk.OptionMenu(self.root, self.selected_file, *json_files)
        self.json_dropdown.pack(padx=20, pady=5)

        # Load button
        self.load_button = tk.Button(
            self.root, text="Load Scene", command=self.load_scene
        )
        self.load_button.pack(padx=20, pady=5)

        save_button = tk.Button(
            self.root, text="Save Grasp Euler", command=self.save_grasp_euler
        )
        save_button.pack(padx=20, pady=5)

        self.filename_checkbox = tk.Checkbutton(
            self.root,
            text="Custom Savefile Name",
            var=self.custom_filename_var,
            command=self.toggle_filename_entry,
        )
        self.filename_checkbox.pack(pady=5)

        self.filename_entry = tk.Entry(
            self.root, textvariable=self.filename_entry_var, state=tk.DISABLED
        )
        self.filename_entry.pack(padx=20, pady=5)
        
        self.display_disqualified_checkbox = tk.Checkbutton(
            self.root,
            text="Display Disqualified Grasps",
            var=self.display_disqualified_var,
            command=self.toggle_grasp_display,
        )
        self.display_disqualified_checkbox.pack(pady=5)
        


        # Grasp navigation frame
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=5)

        self.prev_button = tk.Button(nav_frame, text="<", command=self.prev_grasp)
        self.prev_button.pack(side=tk.LEFT, padx=5)


        self.next_button = tk.Button(nav_frame, text=">", command=self.next_grasp)
        self.next_button.pack(side=tk.LEFT, padx=5)

    def load_scene(self):
        selected = self.selected_file.get()
        if selected:
            app_state.selected_grasps = []
            app_state.current_grasp_index = 0
            load_and_process_scene(
                self.vis,
                selected,
                self.grasp_sampler,
                self.gripper_name,
                self.args,
            )

    def toggle_filename_entry(self):
        if self.custom_filename_var.get():
            self.filename_entry.config(state=tk.NORMAL)
        else:
            self.filename_entry.config(state=tk.DISABLED)
    def save_grasp_euler(self):
        grasp = app_state.selected_grasp_pool[app_state.current_grasp_index]
        position = grasp[:3, 3].tolist()
        euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
        euler_orientation = np.rad2deg(euler_orientation).tolist()
        data = {
            "position" : position,
            "euler_orientation" : euler_orientation
        }
        
        input_filename = self.selected_file.get()
        base_name = os.path.basename(input_filename)
        timestamp_from_file = os.path.splitext(base_name)[0]
        json_filename = f"grasp_{timestamp_from_file}.json"
        json_filepath = os.path.join("output", json_filename)

        with open(json_filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"saved to {json_filepath}")
        os.kill(os.getpid(), signal.SIGINT)
    """
    def save_isaac_grasps(self):
        if app_state.grasps is not None and app_state.grasp_conf is not None:
            if self.custom_filename_var.get():
                filename = self.filename_entry_var.get()
                if not filename.endswith(".yaml"):
                    filename += ".yaml"
            else:
                input_filename = self.selected_file.get()
                base_name = os.path.basename(input_filename)
                timestamp_from_file = os.path.splitext(base_name)[0]
                filename = f"isaac_grasp_{timestamp_from_file}.yaml"

            output_path = os.path.join("output", filename)

            selected_grasps_indices = app_state.selected_grasps
            if not selected_grasps_indices:
                print("No selected grasps to save.")
                return

            selected_grasps = app_state.grasps[selected_grasps_indices]
            selected_confs = app_state.grasp_conf[selected_grasps_indices]

            print(f"Saving {len(selected_grasps)} grasps to {output_path}...")
            save_to_isaac_grasp_format(
                grasps=selected_grasps,
                confidences=selected_confs,
                output_path=output_path,
            )
            print("Save complete.")
        else:
            print("No grasps available to save.")
    """
    def quickload(self):
        if self.args.filename:
            # Find the full path from json_files that matches the filename
            found_file = None
            for f in self.json_files:
                if os.path.basename(f) == self.args.filename:
                    found_file = f
                    break

            if found_file:
                self.selected_file.set(found_file)
                self.filename_entry_var.set(self.args.filename)
                self.load_scene()
            else:
                print(f"File not found in json_files: {self.args.filename}")

    def run(self):
        self.root.after(100, self.quickload)
        self.root.mainloop()

    def next_grasp(self, event=None):
        if app_state.selected_grasp_pool is not None:
            app_state.current_grasp_index = (app_state.current_grasp_index + 1) % len(
                app_state.selected_grasp_pool
            )
            self._update_grasps()

    def prev_grasp(self, event=None):
        if app_state.selected_grasp_pool is not None:
            app_state.current_grasp_index = (app_state.current_grasp_index - 1) % len(
                app_state.selected_grasp_pool
            )
            self._update_grasps()



    def toggle_grasp_display(self):
        app_state.display_unqualified_grasps = self.display_disqualified_var.get()
        if app_state.display_unqualified_grasps:
            app_state.selected_grasp_pool = app_state.grasps
        else:
            app_state.selected_grasp_pool = app_state.qualified_grasps
        app_state.current_grasp_index = 0
        self._update_grasps()


    def _update_grasps(self):
        update_grasp_visualization(
            self.vis,
            self.gripper_name,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasps on a scene point cloud after GraspGen inference, for entire scene"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default=str(config.GRASPGEN_SCENE_DIR),
        help="Directory containing JSON files with point cloud data",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default=str(config.GRIPPER_CFG),
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.70,
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
        default=5,
        help="Number of top grasps to return when return_topk is True",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="Specific JSON file to process. If not specified, a GUI will be presented to choose from sample_data_dir",
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

    obj_pc, _, obj_pc_color, _ = point_cloud_outlier_removal_with_color(
        torch.from_numpy(obj_pc), torch.from_numpy(obj_pc_color)
    )
    obj_pc = obj_pc.cpu().numpy()
    obj_pc_color = obj_pc_color.cpu().numpy()

    visualize_pointcloud(vis, "pc_obj", obj_pc, obj_pc_color, size=0.005)

    return obj_pc

def get_right_up_and_front(grasp: np.array):
    right = grasp[:3, 0]
    up = grasp[:3, 1]
    front = grasp[:3, 2]
    return right, up, front
def is_qualified(grasp: np.array):
    right, up, front = get_right_up_and_front(grasp)
    if up[2] < 0.9:
        return False
    if front[0] < -0.5:
        return False
    return True

def update_grasp_visualization(vis, gripper_name):
    if app_state.selected_grasp_pool is None:
        return

    vis["GraspGen"].delete()

    for j, grasp in enumerate(app_state.selected_grasp_pool):
        is_current = j == app_state.current_grasp_index


        color = [0, 185, 0]  # Default color for non-selected grasps
        if is_current:
            color = [255, 0, 0]  # Highlight color for current grasp
            print(grasp)
            right, up, front = get_right_up_and_front(grasp)
            origin = grasp[:3, 3]
            vector_length = 0.1
            num_points = 10

            # Right vector (red)
            right_points = np.linspace(
                origin, origin + right * vector_length, num_points
            )
            right_colors = np.tile([255, 0, 0], (num_points, 1))
            visualize_pointcloud(
                vis, f"GraspGen/{j:03d}/right", right_points, right_colors, size=0.005
            )

            # Up vector (green)
            up_points = np.linspace(origin, origin + up * vector_length, num_points)
            up_colors = np.tile([0, 255, 0], (num_points, 1))
            visualize_pointcloud(
                vis, f"GraspGen/{j:03d}/up", up_points, up_colors, size=0.005
            )

            # Front vector (blue)
            front_points = np.linspace(
                origin, origin + front * vector_length, num_points
            )
            front_colors = np.tile([0, 0, 255], (num_points, 1))
            visualize_pointcloud(
                vis, f"GraspGen/{j:03d}/front", front_points, front_colors, size=0.005
            )
        visualize_grasp(
            vis,
            f"GraspGen/{j:03d}/grasp",
            grasp,
            color=color,
            gripper_name=gripper_name,
            linewidth=2.5 if is_current else 1.5,
        )


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
        app_state.grasp_conf = grasp_conf.cpu().numpy()
        app_state.grasps = grasps.cpu().numpy()
        app_state.grasps[:, 3, 3] = 1
        app_state.qualified_grasps = np.array([grasp for grasp in app_state.grasps if is_qualified(grasp)])
        if app_state.display_unqualified_grasps:
            app_state.selected_grasp_pool = app_state.grasps
        else:
            app_state.selected_grasp_pool = app_state.qualified_grasps
        print(
            f"[{method}] Scores with min {app_state.grasp_conf.min():.3f} and max {app_state.grasp_conf.max():.3f}"
        )
        update_grasp_visualization(vis, gripper_name)
    else:
        print(f"[{method}] No grasps found! Skipping to next scene...")
        app_state.grasps = None
        app_state.grasp_conf = None


def create_control_panel(vis, json_files, grasp_sampler, gripper_name, args):
    """Creates and runs the tkinter control panel."""
    root = tk.Tk()
    panel = ControlPanel(root, vis, json_files, grasp_sampler, gripper_name, args)
    panel.run()


def load_and_process_scene(vis, json_file, grasp_sampler, gripper_name, args):
    """Loads a scene, processes it, and visualizes grasps."""
    obj_pc = process_and_visualize_scene(vis, json_file)
    generate_and_visualize_grasps(vis, obj_pc, grasp_sampler, gripper_name, args)


def main():
    """Main function to run the GraspGen demo."""
    start_meshcat_server()
    open_meshcat_url("http://127.0.0.1:7000/static/")

    args = parse_args()
    validate_args(args)

    json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {args.sample_data_dir}")
        return

    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    vis = create_visualizer()

    # Start the GUI in a separate thread.
    gui_thread = Thread(
        target=create_control_panel,
        args=(vis, json_files, grasp_sampler, gripper_name, args),
        daemon=True,
    )
    gui_thread.start()

    # Keep the main thread alive to handle signals
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
