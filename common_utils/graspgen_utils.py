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
from grasp_gen.robot import get_gripper_info
from common_utils.qualification import is_qualified_with_name
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps


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


def flip_grasp(grasp: np.ndarray) -> np.ndarray:
    """
    Flips a grasp by rotating it 180 degrees around its approach (front) axis.
    This negates the 'left' and 'up' vectors.
    """
    flipped_grasp = grasp.copy()
    # Negate the left and up vectors (first two columns of rotation matrix)
    flipped_grasp[:3, 0] = -grasp[:3, 0]
    flipped_grasp[:3, 1] = -grasp[:3, 1]
    return flipped_grasp


def flip_upside_down_grasps(grasps: np.ndarray) -> np.ndarray:
    """
    Flips grasps that are "upside down" (up vector's y-component is negative).
    """
    flipped_grasps = []
    for grasp in grasps:
        _grasp = grasp.copy()
        _, up, _ = get_left_up_and_front(_grasp)
        if up[2] < 0:
            _grasp = flip_grasp(_grasp)
        flipped_grasps.append(_grasp)
    return np.array(flipped_grasps)


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
        gripper_name,
    ):
        self.root = root
        self.vis = vis
        self.gripper_name = gripper_name
        self.grasp_queue = grasp_queue
        self.grasps_to_handle_queue = grasps_to_handle_queue
        self.root.title("Control Panel")

        self.apply_custom_filter_var = tk.BooleanVar(value=True)
        self.apply_collision_filter_var = tk.BooleanVar(value=True)

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

        self.apply_custom_filter_checkbox = tk.Checkbutton(
            self.root,
            text="Apply Custom Filter",
            var=self.apply_custom_filter_var,
            command=self.toggle_grasp_display,
        )
        self.apply_custom_filter_checkbox.pack(pady=5)

        self.apply_collision_filter_checkbox = tk.Checkbutton(
            self.root,
            text="Apply Collision Filter",
            var=self.apply_collision_filter_var,
            command=self.toggle_grasp_display,
        )
        self.apply_collision_filter_checkbox.pack(pady=5)

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
        grasps_mask = np.array([True for grasp in self.all_grasps])
        if self.apply_custom_filter_var.get():
            grasps_mask = grasps_mask & self.custom_filter_mask
        if self.apply_collision_filter_var.get():
            grasps_mask = grasps_mask & self.collision_free_mask
        self.current_index = 0
        self.current_grasp_pool = self.all_grasps[grasps_mask]
        self._update_grasp_visualization()

    def _return_grasp_and_destroy(self):
        self.grasp_queue.put(self.current_grasp_pool[self.current_index])
        self.root.destroy()

    def _return_None_and_retry(self):
        self.grasp_queue.put(None)
        self._catch_grasps()

    def _catch_grasps(self):
        self.all_grasps, self.custom_filter_mask, self.collision_free_mask = (
            self.grasps_to_handle_queue.get()
        )
        self.toggle_grasp_display()

    def _update_grasp_visualization(self):
        if len(self.current_grasp_pool) == 0:
            return
        self.vis["GraspGen"].delete()

        for j, grasp in enumerate(self.current_grasp_pool):
            is_current = j == self.current_index

            color = [0, 185, 0]  # Default color for non-selected grasps
            if is_current:
                color = [255, 0, 0]  # Highlight color for current grasp
                logger.debug(f"Grasp:\n{grasp}")
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


def angle_offset_rad(grasp: np.ndarray) -> float:
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    return angle_diff


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
        self.gripper_name: str = self.grasp_cfg.data.gripper_name
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        self.grasp_threshold: float = grasp_threshold
        self.num_grasps: int = num_grasps
        self.topk_num_grasps: int = topk_num_grasps
        self.need_GUI: bool = need_GUI
        self.scene_data: dict
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
            grasp_threshold=0.8,
            num_grasps=200,
            # topk_num_grasps=5,
            min_grasps=80,
            max_tries=20,
        )
        grasps = grasps.cpu().numpy()
        grasps[:, 3, 3] = 1
        grasps = flip_upside_down_grasps(grasps)
        custom_filter_mask = np.array(
            [
                is_qualified_with_name(grasp, qualifier_name, mass_center, std)
                for grasp in grasps
            ]
        )
        collision_free_mask = np.array([True for grasp in grasps])

        gripper_info = get_gripper_info(self.gripper_name)
        gripper_collision_mesh = gripper_info.collision_mesh

        # extract xyz_scene
        full_pc_key = (
            "pc_color" if "pc_color" in self.scene_data["scene_info"] else "full_pc"
        )
        xyz_scene = np.array(self.scene_data["scene_info"][full_pc_key])[0]
        for obj_name in self.scene_data["object_infos"]:
            if obj_name != self.action["target_name"]:
                logger.debug(f"Scene points before: {xyz_scene.shape[0]}")
                xyz_scene = np.vstack(
                    (xyz_scene, self.scene_data["object_infos"][obj_name]["points"])
                )
                logger.debug(f"Scene points after: {xyz_scene.shape[0]}")
        # VIZ_BOUNDS
        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]
        mask_within_bounds = np.all((xyz_scene > VIZ_BOUNDS[0]), 1)
        mask_within_bounds = np.logical_and(
            mask_within_bounds, np.all((xyz_scene < VIZ_BOUNDS[1]), 1)
        )
        xyz_scene = xyz_scene[mask_within_bounds]
        # Downsample scene point cloud for faster collision checking
        if len(xyz_scene) > 8192:
            indices = np.random.choice(len(xyz_scene), 8192, replace=False)
            xyz_scene_downsampled = xyz_scene[indices]
            logger.debug(
                f"Downsampled scene point cloud from {len(xyz_scene)} to {len(xyz_scene_downsampled)} points"
            )
        else:
            xyz_scene_downsampled = xyz_scene
            logger.debug(
                f"Scene point cloud has {len(xyz_scene)} points (no downsampling needed)"
            )
        collision_free_mask = filter_colliding_grasps(
            scene_pc=xyz_scene_downsampled,
            grasp_poses=grasps,
            gripper_collision_mesh=gripper_collision_mesh,
            collision_threshold=0.03,
        )
        return grasps, custom_filter_mask, collision_free_mask

    def _generate_grasp_silent(self) -> np.array:
        """
        Returns:
            Qualified grasps, A list of that contains multiple np.ndarray s of shape(4, 4).
        """
        GRASPS_BATCH_SIZE = 4
        num_try = 0
        qualified_grasps = []
        while True:
            num_try += 1
            logger.info(f"try #{num_try}")
            all_grasps, custom_filter_mask, collision_free_mask = (
                self._generate_grasps()
            )
            qualified_grasps.extend(
                list(all_grasps[custom_filter_mask & collision_free_mask])
            )
            if len(qualified_grasps) >= GRASPS_BATCH_SIZE:
                qualified_grasps = sorted(qualified_grasps, key=angle_offset_rad)
                return qualified_grasps[:GRASPS_BATCH_SIZE]

    def generate_grasp(self, scene_data: dict, action: dict) -> list[np.ndarray]:
        """
        Returns:
            grasps: A list of shape(4, 4) np array. only single one elements if returned from self._generate_grasp_with_GUI(), while multiple elements if returned by self._generate_grasp_silent()
        """
        # reload scene_data and action
        self.scene_data = scene_data
        self.action = action

        if self.need_GUI:
            # create control panel and use GUI mode
            return self._generate_grasp_with_GUI()
        else:
            return self._generate_grasp_silent()

    def _generate_grasp_with_GUI(self) -> list[np.ndarray]:
        """
        Returns:
            A list of that contains only one element: np.ndarray of shape(4, 4).
        """
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
            all_grasps, custom_filter_mask, collision_free_mask = (
                self._generate_grasps()
            )
            # send to panel
            grasps_to_handle_queue.put(
                (all_grasps, custom_filter_mask, collision_free_mask)
            )
            # wait panel to respond
            selected_grasp = grasp_q.get()
            if selected_grasp is not None:
                return [selected_grasp]

    def _create_control_panel(self, grasp_queue, grasps_to_handle_queue):
        """Creates and runs the tkinter control panel."""
        root = tk.Tk()
        panel = ControlPanel(
            root,
            self.vis,
            grasp_queue,
            grasps_to_handle_queue,
            # self.action["qualifier"],
            self.gripper_name,
        )
        panel.run()

    def _visualize_scene(self):
        """Loads scene data, processes point clouds, and visualizes them."""
        # self.vis.delete()
        scene_info = self.scene_data["scene_info"]
        xyz_scene = np.array(scene_info["pc_color"])[0]
        xyz_scene_color = np.array(scene_info["img_color"]).reshape(1, -1, 3)[0, :, :]

        for obj_name in self.scene_data["object_infos"]:
            xyz_scene = np.vstack(
                (xyz_scene, self.scene_data["object_infos"][obj_name]["points"])
            )
            xyz_scene_color = np.vstack(
                (xyz_scene_color, self.scene_data["object_infos"][obj_name]["colors"])
            )

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

    def _visualize_target(self):
        obj_pc = self.scene_data["object_infos"][self.action["target_name"]]["points"]
        obj_pc_color = self.scene_data["object_infos"][self.action["target_name"]][
            "colors"
        ]
        visualize_pointcloud(self.vis, "pc_obj", obj_pc, obj_pc_color, size=0.005)
