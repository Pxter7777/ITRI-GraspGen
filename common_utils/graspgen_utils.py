import numpy as np
import logging
import os
import atexit
import subprocess
import signal
import time
import webbrowser
import platform
import tkinter as tk
from threading import Thread
import queue
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from common_utils.qualification import is_qualified_with_name
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_pointcloud,
)


logger = logging.getLogger(__name__)


def is_qualified(grasp: np.array, mass_center, obj_std):
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    if up[2] < 0.95:
        return False
    # if front[0] < 0.8:
    #    return False
    # if front[1] < -0.2:
    #    return False

    # Rule: planar 2D angle between grasp approach (front) vector and grasp position vector should be small
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    if angle_diff > np.deg2rad(30):
        return False

    if position[2] < 0.056:  # for safety
        return False
    if position[2] > mass_center[2] + obj_std[2]:  # too high
        return False
    if position[2] < mass_center[2] - obj_std[2]:  # too low
        return False
    return True


class GraspGenerator:
    def __init__(self, gripper_config, grasp_threshold, num_grasps, topk_num_grasps):
        self.grasp_cfg = load_grasp_cfg(gripper_config)
        self.gripper_name = self.grasp_cfg.data.gripper_name
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        self.grasp_threshold = grasp_threshold
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps

    def auto_select_valid_cup_grasp(self, pointcloud: np.array) -> np.array:
        mass_center = np.mean(pointcloud, axis=0)
        std = np.std(pointcloud, axis=0)
        try:
            num_try = 0
            while True:
                num_try += 1
                logging.info(f"try #{num_try}")
                grasps, grasp_conf = GraspGenSampler.run_inference(
                    pointcloud,
                    self.grasp_sampler,
                    grasp_threshold=self.grasp_threshold,
                    num_grasps=self.num_grasps,
                    topk_num_grasps=self.topk_num_grasps,
                )
                grasps = grasps.cpu().numpy()
                grasps[:, 3, 3] = 1
                qualified_grasps = np.array(
                    [grasp for grasp in grasps if is_qualified(grasp, mass_center, std)]
                )
                if len(qualified_grasps) > 0:
                    return qualified_grasps[0]
        except KeyboardInterrupt:  # manual stop if too many fail
            logging.info("Manually stopping generating grasps")
            return None

    def flexible_auto_select_valid_grasp(self, obj: dict, qualifier: str) -> np.array:
        mass_center = np.mean(obj["points"], axis=0)
        std = np.std(obj["points"], axis=0)

        num_try = 0
        while True:
            num_try += 1
            logging.info(f"try #{num_try}")
            grasps, grasp_conf = GraspGenSampler.run_inference(
                obj["points"],
                self.grasp_sampler,
                grasp_threshold=self.grasp_threshold,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.topk_num_grasps,
            )
            grasps = grasps.cpu().numpy()
            grasps[:, 3, 3] = 1
            qualified_grasps = np.array(
                [
                    grasp
                    for grasp in grasps
                    if is_qualified_with_name(grasp, qualifier, mass_center, std)
                ]
            )
            if len(qualified_grasps) > 0:
                return qualified_grasps[0]


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


def get_left_up_and_front(grasp: np.array):
    left = grasp[:3, 0]
    up = grasp[:3, 1]
    front = grasp[:3, 2]
    return left, up, front


class ControlPanel:
    def __init__(
        self,
        root,
        vis,
        grasp_queue,
        grasps_to_handle_queue,
        qualifier_name: str,
        gripper_name,
    ):
        self.root = root
        self.vis = vis
        self.qualifier_name = qualifier_name
        self.gripper_name = gripper_name
        self.grasp_queue = grasp_queue
        self.grasps_to_handle_queue = grasps_to_handle_queue
        self.root.title("Control Panel")

        self.display_disqualified_var = tk.BooleanVar(value=False)
        # select_button
        select_button = tk.Button(
            self.root, text="Select Grasp", command=self._return_grasp_and_destroy
        )
        select_button.pack(padx=20, pady=5)
        # retry button
        retry_button = tk.Button(
            self.root, text="Retry", command=self._return_None_and_retry
        )
        retry_button.pack(padx=20, pady=5)

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

    def run(self):
        self.root.after(100, self._catch_grasps)
        self.root.mainloop()

    def next_grasp(self, event=None):
        if len(self.current_grasp_pool) == 0:
            return
        self.current_index = (self.current_index + 1) % len(self.current_grasp_pool)
        self._update_grasp_visualization()

    def prev_grasp(self, event=None):
        if len(self.current_grasp_pool) == 0:
            return
        self.current_index = (self.current_index - 1) % len(self.current_grasp_pool)
        self._update_grasp_visualization()

    def toggle_grasp_display(self):
        if self.display_disqualified_var.get():
            self.current_grasp_pool = self.all_grasps
        else:
            self.current_grasp_pool = self.qualified_grasps
        self.current_index = 0
        self._update_grasp_visualization()

    def _return_grasp_and_destroy(self):
        self.grasp_queue.put(self.current_grasp_pool[self.current_index])
        self.root.destroy()

    def _return_None_and_retry(self):
        self.grasp_queue.put(None)
        self._catch_grasps()

    def _catch_grasps(self):
        self.qualified_grasps, self.all_grasps = self.grasps_to_handle_queue.get()
        self.current_grasp_pool = (
            self.all_grasps
            if self.display_disqualified_var.get()
            else self.qualified_grasps
        )
        self.current_index = 0
        self._update_grasp_visualization()

    def _update_grasp_visualization(self):
        if len(self.current_grasp_pool) == 0:
            return
        self.vis["GraspGen"].delete()

        for j, grasp in enumerate(self.current_grasp_pool):
            is_current = j == self.current_index

            color = [0, 185, 0]  # Default color for non-selected grasps
            if is_current:
                color = [255, 0, 0]  # Highlight color for current grasp
                print(grasp)
                left, up, front = get_left_up_and_front(grasp)
                origin = grasp[:3, 3]
                vector_length = 0.1
                num_points = 10

                # Right vector (red)
                left_points = np.linspace(
                    origin, origin + left * vector_length, num_points
                )
                left_colors = np.tile([255, 0, 0], (num_points, 1))
                visualize_pointcloud(
                    self.vis,
                    f"GraspGen/{j:03d}/left",
                    left_points,
                    left_colors,
                    size=0.005,
                )

                # Up vector (green)
                up_points = np.linspace(origin, origin + up * vector_length, num_points)
                up_colors = np.tile([0, 255, 0], (num_points, 1))
                visualize_pointcloud(
                    self.vis, f"GraspGen/{j:03d}/up", up_points, up_colors, size=0.005
                )

                # Front vector (blue)
                front_points = np.linspace(
                    origin, origin + front * vector_length, num_points
                )
                front_colors = np.tile([0, 0, 255], (num_points, 1))
                visualize_pointcloud(
                    self.vis,
                    f"GraspGen/{j:03d}/front",
                    front_points,
                    front_colors,
                    size=0.005,
                )
            visualize_grasp(
                self.vis,
                f"GraspGen/{j:03d}/grasp",
                grasp,
                color=color,
                gripper_name=self.gripper_name,
                linewidth=2.5 if is_current else 1.5,
            )


def create_control_panel(
    vis,
    grasp_sampler,
    gripper_name,
    grasp_threshold,
    num_grasps,
    topk_num_grasps,
    grasp_queue,
):
    """Creates and runs the tkinter control panel."""
    root = tk.Tk()
    panel = ControlPanel(
        root,
        vis,
        grasp_sampler,
        gripper_name,
        grasp_threshold,
        num_grasps,
        topk_num_grasps,
        grasp_queue,
    )
    panel.run()


class GraspGeneratorUI:
    def __init__(
        self,
        gripper_config,
        grasp_threshold,
        num_grasps,
        topk_num_grasps,
        need_GUI=False,
    ):
        self.grasp_cfg = load_grasp_cfg(gripper_config)
        self.gripper_name = self.grasp_cfg.data.gripper_name
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        self.grasp_threshold = grasp_threshold
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps
        self.need_GUI = need_GUI
        ## Main init starts here
        if self.need_GUI:
            start_meshcat_server()
            open_meshcat_url("http://127.0.0.1:7000/static/")
            self.vis = create_visualizer()

    def _generate_grasps(self) -> tuple[np.array, np.array]:
        obj_name = self.action["target_name"]
        obj_pc = self.scene_data["object_infos"][obj_name]["points"]
        qualifier_name = self.action["qualifier"]
        mass_center = np.mean(obj_pc, axis=0)
        std = np.std(obj_pc, axis=0)
        grasps, grasp_conf = GraspGenSampler.run_inference(
            obj_pc,
            self.grasp_sampler,
            grasp_threshold=self.grasp_threshold,
            num_grasps=self.num_grasps,
            topk_num_grasps=self.topk_num_grasps,
        )
        grasps = grasps.cpu().numpy()
        grasps[:, 3, 3] = 1
        qualified_grasps = np.array(
            [
                grasp
                for grasp in grasps
                if is_qualified_with_name(grasp, qualifier_name, mass_center, std)
            ]
        )
        return qualified_grasps, grasps

    def _generate_grasp_silent(self) -> np.array:
        num_try = 0
        while True:
            num_try += 1
            logger.info(f"try #{num_try}")
            qualified_grasps, _ = self._generate_grasps()
            if len(qualified_grasps) > 0:
                return qualified_grasps[0]

    def generate_grasp(self, scene_data: dict, action: dict) -> np.array:
        # reload scene_data and action
        self.scene_data = scene_data
        self.action = action

        if self.need_GUI:
            # create control panel and use GUI mode
            return self._generate_grasp_with_GUI()
        else:
            return self._generate_grasp_silent()

    def _generate_grasp_with_GUI(self):
        self._visualize_scene()
        grasp_q = queue.Queue()
        grasps_to_handle_queue = queue.Queue()
        gui_thread = Thread(
            target=self._create_control_panel,
            args=(grasp_q, grasps_to_handle_queue),
            daemon=True,
        )

        gui_thread.start()
        num_try = 0
        while True:
            num_try += 1
            qualified_grasps, all_grasps = self._generate_grasps()
            # send to panel
            grasps_to_handle_queue.put((qualified_grasps, all_grasps))
            # wait panel to respond
            selected_grasp = grasp_q.get()
            if selected_grasp is not None:
                return selected_grasp

    def _create_control_panel(self, grasp_queue, grasps_to_handle_queue):
        """Creates and runs the tkinter control panel."""
        root = tk.Tk()
        panel = ControlPanel(
            root,
            self.vis,
            grasp_queue,
            grasps_to_handle_queue,
            self.action["qualifier"],
            self.gripper_name,
        )
        panel.run()

    def _visualize_scene(self):
        """Loads scene data, processes point clouds, and visualizes them."""
        self.vis.delete()
        scene_info = self.scene_data["scene_info"]
        xyz_scene = np.array(scene_info["pc_color"])[0]
        xyz_scene_color = np.array(scene_info["img_color"]).reshape(1, -1, 3)[0, :, :]

        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]
        mask_within_bounds = np.all((xyz_scene > VIZ_BOUNDS[0]), 1)
        mask_within_bounds = np.logical_and(
            mask_within_bounds, np.all((xyz_scene < VIZ_BOUNDS[1]), 1)
        )
        xyz_scene = xyz_scene[mask_within_bounds]
        xyz_scene_color = xyz_scene_color[mask_within_bounds]

        visualize_pointcloud(
            self.vis, "pc_scene", xyz_scene, xyz_scene_color, size=0.0025
        )
