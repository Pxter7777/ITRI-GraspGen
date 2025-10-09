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
import glob

import numpy as np

from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_pointcloud,
)
from scipy.spatial.transform import Rotation as R

class AppState:
    def __init__(self):
        self.pc = None
        self.pc_color = None
        self.original_pc = None
        self.transformation = np.identity(4)

app_state = AppState()

class ControlPanel:
    def __init__(self, root, vis, json_files, args):
        self.root = root
        self.vis = vis
        self.json_files = json_files
        self.args = args
        self.root.title("Control Panel")

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

        # Transformation controls
        transform_frame = tk.Frame(self.root, borderwidth=2, relief="groove")
        transform_frame.pack(padx=10, pady=10)
        
        tk.Label(transform_frame, text="Translation").grid(row=0, column=0, columnspan=2)
        self.trans_x = self.create_slider(transform_frame, "X:", -2.0, 2.0, 0.01, 1, self.on_transform_change)
        self.trans_y = self.create_slider(transform_frame, "Y:", -2.0, 2.0, 0.01, 2, self.on_transform_change)
        self.trans_z = self.create_slider(transform_frame, "Z:", -2.0, 2.0, 0.01, 3, self.on_transform_change)

        tk.Label(transform_frame, text="Rotation (degrees)").grid(row=4, column=0, columnspan=2)
        self.rot_r = self.create_slider(transform_frame, "Roll:", -180, 180, 1, 5, self.on_transform_change)
        self.rot_p = self.create_slider(transform_frame, "Pitch:", -180, 180, 1, 6, self.on_transform_change)
        self.rot_y = self.create_slider(transform_frame, "Yaw:", -180, 180, 1, 7, self.on_transform_change)

        self.save_button = tk.Button(
            self.root, text="Save Transformed PC", command=self.save_transformed_pc
        )
        self.save_button.pack(padx=20, pady=5)

    def create_slider(self, parent, label, from_, to, resolution, row, command):
        tk.Label(parent, text=label).grid(row=row, column=0)
        slider = tk.Scale(parent, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL, length=200, command=command)
        slider.grid(row=row, column=1)
        return slider

    def on_transform_change(self, value):
        self.apply_transform()

    def load_scene(self):
        selected = self.selected_file.get()
        if selected:
            load_and_process_scene(self.vis, selected)

    def apply_transform(self):
        if app_state.original_pc is None:
            return

        # Get values from sliders
        tx = self.trans_x.get()
        ty = self.trans_y.get()
        tz = self.trans_z.get()
        rr = self.rot_r.get()
        rp = self.rot_p.get()
        ry = self.rot_y.get()

        # Create transformation matrix
        translation_matrix = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

        rotation = R.from_euler('xyz', [rr, rp, ry], degrees=True)
        rotation_matrix = np.identity(4)
        rotation_matrix[:3, :3] = rotation.as_matrix()
        
        app_state.transformation = translation_matrix @ rotation_matrix

        # Apply transformation
        original_pc_homogeneous = np.hstack((app_state.original_pc, np.ones((app_state.original_pc.shape[0], 1))))
        transformed_pc_homogeneous = (app_state.transformation @ original_pc_homogeneous.T).T
        app_state.pc = transformed_pc_homogeneous[:, :3]

        # Update visualization
        update_visualization(self.vis)

    def save_transformed_pc(self):
        if app_state.pc is None:
            print("No point cloud to save.")
            return

        input_filename = self.selected_file.get()
        base_name = os.path.basename(input_filename)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_transformed.json"
        output_path = os.path.join("output", output_filename)
        
        os.makedirs("output", exist_ok=True)

        data_to_save = {
            "transformation_matrix": app_state.transformation.tolist(),
            "point_cloud": {
                "points": app_state.pc.tolist(),
                "colors": app_state.pc_color.tolist()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Saved transformed point cloud and matrix to {output_path}")


    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Manually transform a point cloud.")
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="output",
        help="Directory containing JSON files with point cloud data",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="Specific JSON file to process.",
    )
    return parser.parse_args()


def start_meshcat_server():
    print("Starting meshcat-server...")
    meshcat_server_process = subprocess.Popen(
        "meshcat-server", shell=True, preexec_fn=os.setsid
    )

    @atexit.register
    def cleanup_meshcat_server():
        if meshcat_server_process.poll() is None:
            print("Terminating meshcat-server...")
            os.killpg(os.getpgid(meshcat_server_process.pid), signal.SIGTERM)

    time.sleep(2)
    return meshcat_server_process

def open_meshcat_url(url):
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
        pass

def load_and_process_scene(vis, json_file):
    print(f"Loading scene from {json_file}")
    vis.delete()

    with open(json_file, "rb") as f:
        data = json.load(f)

    pc_list = []
    pc_color_list = []

    if "object_info" in data and "pc" in data["object_info"] and len(data["object_info"]["pc"]) > 0:
        pc_list.append(np.array(data["object_info"]["pc"]))
        pc_color_list.append(np.array(data["object_info"]["pc_color"]))

    if "scene_info" in data and "pc_color" in data["scene_info"] and len(data["scene_info"]["pc_color"]) > 0:
        # scene_info stores pc and color differently
        scene_pc_with_color = np.array(data["scene_info"]["pc_color"][0])
        scene_colors = np.array(data["scene_info"]["img_color"][0])
        pc_list.append(scene_pc_with_color)
        pc_color_list.append(scene_colors)

    if "point_cloud" in data: # for loading already transformed clouds
        pc = np.array(data["point_cloud"]["points"])
        pc_color = np.array(data["point_cloud"]["colors"])
        app_state.original_pc = pc
        app_state.pc = pc
        app_state.pc_color = pc_color
        update_visualization(vis)
        return

    if not pc_list:
        print("Could not find point cloud data in JSON file.")
        return

    pc = np.concatenate(pc_list, axis=0)
    pc_color = np.concatenate(pc_color_list, axis=0)

    app_state.original_pc = pc
    app_state.pc = pc
    app_state.pc_color = pc_color
    
    update_visualization(vis)

def update_visualization(vis):
    if app_state.pc is not None:
        visualize_pointcloud(vis, "pointcloud", app_state.pc, app_state.pc_color, size=0.005)

def create_control_panel(vis, json_files, args):
    root = tk.Tk()
    panel = ControlPanel(root, vis, json_files, args)
    panel.run()


def main():
    start_meshcat_server()
    open_meshcat_url("http://127.0.0.1:7000/static/")

    args = parse_args()

    json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {args.sample_data_dir}")
        return

    vis = create_visualizer()

    gui_thread = Thread(
        target=create_control_panel,
        args=(vis, json_files, args),
        daemon=True,
    )
    gui_thread.start()

    if args.filename:
        if os.path.exists(args.filename):
            load_and_process_scene(vis, args.filename)
        else:
            print(f"File not found: {args.filename}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
