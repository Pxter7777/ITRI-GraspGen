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
import pye57

from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_pointcloud,
)
from scipy.spatial.transform import Rotation as R


class AppState:
    def __init__(self):
        self.object_pc = None
        self.object_pc_color = None
        self.original_object_pc = None
        self.scene_pc = None
        self.scene_pc_color = None
        self.original_scene_pc = None
        self.transformation = np.identity(4)


class PCInfo:
    def __init__(self):
        self.object_pc = None
        self.object_pc_color = None
        self.scene_pc = None
        self.scene_pc_color = None


app_state = AppState()


def save_pointcloud(info: PCInfo, output_path: os.path):
    data_to_save = {
        "object_info": {
            "pc": info.object_pc.tolist() if info.object_pc is not None else [],
            "pc_color": info.object_pc_color.tolist()
            if info.object_pc_color is not None
            else [],
        },
        "scene_info": {
            "pc_color": [info.scene_pc.tolist()] if info.scene_pc is not None else [],
            "img_color": [info.scene_pc_color.tolist()]
            if info.scene_pc_color is not None
            else [],
        },
        "grasp_info": {"grasp_poses": [], "grasp_conf": []},
    }

    with open(output_path, "w") as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Saved transformed point cloud and matrix to {output_path}")


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

        self.trans_x_var = self.create_control(
            transform_frame, "X:", -2.0, 2.0, 0.01, 1
        )
        self.trans_y_var = self.create_control(
            transform_frame, "Y:", -2.0, 2.0, 0.01, 2
        )
        self.trans_z_var = self.create_control(
            transform_frame, "Z:", -2.0, 2.0, 0.01, 3
        )
        self.rot_r_var = self.create_control(transform_frame, "Roll:", -180, 180, 1, 5)
        self.rot_p_var = self.create_control(transform_frame, "Pitch:", -180, 180, 1, 6)
        self.rot_y_var = self.create_control(transform_frame, "Yaw:", -180, 180, 1, 7)

        self.save_button = tk.Button(
            self.root,
            text="Save Transformed PC (JSON)",
            command=self.save_transformed_pc,
        )
        self.save_button.pack(padx=20, pady=5)

        self.save_e57_button = tk.Button(
            self.root, text="Save as .e57", command=self.save_as_e57
        )
        self.save_e57_button.pack(padx=20, pady=5)
        self.save_transform_button = tk.Button(
            self.root,
            text="Save Transform Config (JSON)",
            command=self.save_transform_config,
        )
        self.save_transform_button.pack(padx=20, pady=5)

    def create_control(self, parent, label, from_, to, resolution, row):
        var = tk.DoubleVar()
        var.trace_add("write", self.on_transform_change)

        tk.Label(parent, text=label).grid(row=row, column=0)
        tk.Scale(
            parent,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            length=200,
            variable=var,
        ).grid(row=row, column=1)
        tk.Entry(parent, textvariable=var, width=10).grid(row=row, column=2)
        return var

    def on_transform_change(self, *args):
        self.apply_transform()

    def load_scene(self):
        selected = self.selected_file.get()
        if selected:
            load_and_process_scene(self.vis, selected)

    def apply_transform(self):
        if app_state.original_object_pc is None and app_state.original_scene_pc is None:
            return

        # Get values from sliders
        tx = self.trans_x_var.get()
        ty = self.trans_y_var.get()
        tz = self.trans_z_var.get()
        rr = self.rot_r_var.get()
        rp = self.rot_p_var.get()
        ry = self.rot_y_var.get()

        # Create transformation matrix
        translation_matrix = np.array(
            [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]
        )

        rotation = R.from_euler("xyz", [rr, rp, ry], degrees=True)
        rotation_matrix = np.identity(4)
        rotation_matrix[:3, :3] = rotation.as_matrix()

        app_state.transformation = translation_matrix @ rotation_matrix

        # Apply transformation to object pc
        if app_state.original_object_pc is not None:
            app_state.object_pc = transform(
                app_state.original_object_pc, app_state.transformation
            )
            """
            original_object_pc_homogeneous = np.hstack(
                (
                    app_state.original_object_pc,
                    np.ones((app_state.original_object_pc.shape[0], 1)),
                )
            )
            transformed_object_pc_homogeneous = (
                app_state.transformation @ original_object_pc_homogeneous.T
            ).T
            app_state.object_pc = transformed_object_pc_homogeneous[:, :3]
            """

        # Apply transformation to scene pc
        if app_state.original_scene_pc is not None:
            app_state.scene_pc = transform(
                app_state.original_scene_pc, app_state.transformation
            )
            """            original_scene_pc_homogeneous = np.hstack(
                (
                    app_state.original_scene_pc,
                    np.ones((app_state.original_scene_pc.shape[0], 1)),
                )
            )
            transformed_scene_pc_homogeneous = (
                app_state.transformation @ original_scene_pc_homogeneous.T
            ).T
            app_state.scene_pc = transformed_scene_pc_homogeneous[:, :3]
            """

        # Update visualization
        update_visualization(self.vis)

    def save_transformed_pc(self):
        if app_state.object_pc is None and app_state.scene_pc is None:
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
            "object_info": {
                "pc": app_state.object_pc.tolist()
                if app_state.object_pc is not None
                else [],
                "pc_color": app_state.object_pc_color.tolist()
                if app_state.object_pc_color is not None
                else [],
            },
            "scene_info": {
                "pc_color": [app_state.scene_pc.tolist()]
                if app_state.scene_pc is not None
                else [],
                "img_color": [app_state.scene_pc_color.tolist()]
                if app_state.scene_pc_color is not None
                else [],
            },
            "grasp_info": {"grasp_poses": [], "grasp_conf": []},
        }

        with open(output_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Saved transformed point cloud and matrix to {output_path}")

    def save_transform_config(self):
        if app_state.object_pc is None and app_state.scene_pc is None:
            print("No point cloud to save.")
            return

        input_filename = self.selected_file.get()
        base_name = os.path.basename(input_filename)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_transform_config.json"
        output_path = os.path.join("transform_config", output_filename)

        tx = self.trans_x_var.get()
        ty = self.trans_y_var.get()
        tz = self.trans_z_var.get()
        rr = self.rot_r_var.get()
        rp = self.rot_p_var.get()
        ry = self.rot_y_var.get()

        data_to_save = {"tx": tx, "ty": ty, "tz": tz, "rr": rr, "rp": rp, "ry": ry}
        with open(output_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Saved transforme config {output_path}")

    def save_as_e57(self):
        if app_state.object_pc is None and app_state.scene_pc is None:
            print("No point cloud to save.")
            return

        input_filename = self.selected_file.get()
        base_name = os.path.basename(input_filename)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_transformed.e57"
        output_path = os.path.join("output", output_filename)

        os.makedirs("output", exist_ok=True)

        pc_list = []
        pc_color_list = []
        if app_state.object_pc is not None:
            pc_list.append(app_state.object_pc)
            pc_color_list.append(app_state.object_pc_color)
        if app_state.scene_pc is not None:
            pc_list.append(app_state.scene_pc)
            pc_color_list.append(app_state.scene_pc_color)

        combined_pc = np.concatenate(pc_list, axis=0)
        combined_colors = np.concatenate(pc_color_list, axis=0)

        e57_data = dict()
        e57_data["cartesianX"] = combined_pc[:, 0]
        e57_data["cartesianY"] = combined_pc[:, 1]
        e57_data["cartesianZ"] = combined_pc[:, 2]
        e57_data["colorRed"] = combined_colors[:, 0]
        e57_data["colorGreen"] = combined_colors[:, 1]
        e57_data["colorBlue"] = combined_colors[:, 2]

        with pye57.E57(output_path, mode="w") as e57_write:
            e57_write.write_scan_raw(e57_data)

        print(f"Saved combined transformed point cloud to {output_path}")

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Manually transform a point cloud.")
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="../output",
        help="Directory containing JSON files with point cloud data",
    )
    parser.add_argument(
        "--transform-config",
        type=str,
        default="",
        help="Specific JSON file to process.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="",
        help="Specific JSON file to process.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="quick transform without any gui and meshcat",
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


def load_scene(json_file) -> PCInfo:
    print(f"Loading scene from {json_file}")
    with open(json_file, "rb") as f:
        data = json.load(f)
    info = PCInfo()

    if (
        "object_info" in data
        and "pc" in data["object_info"]
        and len(data["object_info"]["pc"]) > 0
    ):
        info.object_pc = np.array(data["object_info"]["pc"])
        info.object_pc_color = np.array(data["object_info"]["pc_color"])

    if (
        "scene_info" in data
        and "pc_color" in data["scene_info"]
        and len(data["scene_info"]["pc_color"]) > 0
    ):
        info.scene_pc = np.array(data["scene_info"]["pc_color"][0])
        info.scene_pc_color = np.array(data["scene_info"]["img_color"][0])

    if info.object_pc is None and info.scene_pc is None:
        print("Could not find point cloud data in JSON file.")
        exit(1)

    return info


def load_and_process_scene(vis, json_file):
    # print(f"Loading scene from {json_file}")
    vis.delete()

    # with open(json_file, "rb") as f:
    #    data = json.load(f)

    # Reset state
    app_state.object_pc = None
    app_state.object_pc_color = None
    app_state.original_object_pc = None
    app_state.scene_pc = None
    app_state.scene_pc_color = None
    app_state.original_scene_pc = None

    info = load_scene(json_file)
    app_state.object_pc = info.object_pc
    app_state.object_pc_color = info.object_pc_color
    app_state.original_object_pc = info.object_pc
    app_state.scene_pc = info.scene_pc
    app_state.scene_pc_color = info.scene_pc_color
    app_state.original_scene_pc = info.scene_pc

    update_visualization(vis)


def update_visualization(vis):
    pc_list = []
    pc_color_list = []
    if app_state.object_pc is not None:
        pc_list.append(app_state.object_pc)
        pc_color_list.append(app_state.object_pc_color)
    if app_state.scene_pc is not None:
        pc_list.append(app_state.scene_pc)
        pc_color_list.append(app_state.scene_pc_color)

    if not pc_list:
        return

    pc = np.concatenate(pc_list, axis=0)
    pc_color = np.concatenate(pc_color_list, axis=0)

    if pc is not None:
        visualize_pointcloud(vis, "pointcloud", pc, pc_color, size=0.005)


def create_control_panel(vis, json_files, args):
    root = tk.Tk()
    panel = ControlPanel(root, vis, json_files, args)
    panel.run()


def quick_transform(args):
    # check and load transform config
    if args.transform_config == "":
        print("Please provide transform config")
        exit(1)
    transform_filename = os.path.join("transform_config", args.transform_config)
    with open(transform_filename, "rb") as f:
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

    # check and load pointcloud
    if args.sample_data_dir == "":
        print("Please provide pointcloud dir")
        exit(1)
    if args.filename == "":
        print("Please provide poincloud filename")
        exit(1)
    pointcloud_filename = os.path.join(args.sample_data_dir, args.filename)
    info = load_scene(pointcloud_filename)
    info.object_pc = transform(info.object_pc, transformation)
    info.scene_pc = transform(info.scene_pc, transformation)

    # save
    input_filename = pointcloud_filename
    base_name = os.path.basename(input_filename)
    name, ext_ = os.path.splitext(base_name)
    output_filename = f"{name}_transformed.json"
    output_path = os.path.join("output", output_filename)

    save_pointcloud(info, output_path)


def silent_transform(pointcloud: dict, config_filename: str) -> dict:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_filepath = os.path.join(
        current_file_dir, "transform_config", config_filename
    )
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

    # check and load pointcloud

    info = pointcloud
    info["object_info"]["pc"] = transform(info["object_info"]["pc"], transformation)
    info["scene_info"]["pc_color"] = [
        transform(np.array(info["scene_info"]["pc_color"][0]), transformation)
    ]

    return info


def silent_transform_multiple(pointcloud: dict, config_filename: str) -> dict:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_filepath = os.path.join(
        current_file_dir, "transform_config", config_filename
    )
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

    # check and load pointcloud

    info = pointcloud
    for i in range(len(info["objects_info"])):
        info["objects_info"][i]["pc"] = transform(
            info["objects_info"][i]["pc"], transformation
        )
    info["scene_info"]["pc_color"] = [
        transform(np.array(info["scene_info"]["pc_color"][0]), transformation)
    ]

    return info


def silent_transform_multiple_obj_with_name(
    scene_data: dict, config_filename: str
) -> dict:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_filepath = os.path.join(
        current_file_dir, "transform_config", config_filename
    )
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
    for i in range(len(scene_data["object_infos"])):
        scene_data["object_infos"][i]["points"] = transform(
            scene_data["object_infos"][i]["points"], transformation
        )
    scene_data["scene_info"]["pc_color"] = [
        transform(np.array(scene_data["scene_info"]["pc_color"][0]), transformation)
    ]
    return scene_data


def silent_transform_multiple_obj_with_name_dict(
    scene_data: dict, config_filename: str
) -> dict:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_filepath = os.path.join(
        current_file_dir, "transform_config", config_filename
    )
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


def main():
    args = parse_args()
    if args.quick:
        quick_transform(args)
        exit(0)

    start_meshcat_server()
    open_meshcat_url("http://127.0.0.1:7000/static/")

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
