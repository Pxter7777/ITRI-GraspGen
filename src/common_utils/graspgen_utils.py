"""GraspGen inference, filtering, and meshcat-based visualization."""

import atexit
import logging
import os
import platform
import queue
import signal
import subprocess
import time
import tkinter as tk
import webbrowser
from threading import Thread

import meshcat
import numpy as np
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_pointcloud,
)
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

from common_utils.actions_format_checker import MoveItem
from common_utils.qualification import is_qualified
from common_utils.scene_data import SceneData

logger = logging.getLogger(__name__)

"""
def is_qualified(grasp: np.array, mass_center, obj_std):
    position = grasp[:3, 3].tolist()
    left, up, front = get_left_up_and_front(grasp)
    if up[2] < 0.95:
        return False
    # if front[0] < 0.8:
    #    return False
    # if front[1] < -0.2:
    #    return False

    # Rule: planar 2D angle between grasp approach (front)
    # vector and grasp position vector should be small
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
"""


def flip_grasp(grasp: np.ndarray) -> np.ndarray:
    """Flip a grasp 180 degrees around its approach axis.

    This negates the 'left' and 'up' vectors.
    """
    flipped_grasp = grasp.copy()
    # Negate the left and up vectors (first two columns of rotation matrix)
    flipped_grasp[:3, 0] = -grasp[:3, 0]
    flipped_grasp[:3, 1] = -grasp[:3, 1]
    return flipped_grasp


def flip_upside_down_grasps(grasps: np.ndarray) -> np.ndarray:
    """Flips grasps that are "upside down" (up vector's y-component is negative)."""
    flipped_grasps = []
    for grasp in grasps:
        _grasp = grasp.copy()
        _, up, _ = get_left_up_and_front(_grasp)
        if up[2] < 0:
            _grasp = flip_grasp(_grasp)
        flipped_grasps.append(_grasp)
    return np.array(flipped_grasps)


class GraspGenerator:
    """Run GraspGen inference and filter results by qualification rules.

    Args:
        gripper_config: Path to the gripper YAML config.
        grasp_threshold: Minimum confidence score for generated grasps.
        num_grasps: Number of grasps to sample per batch.
        topk_num_grasps: Number of top-scoring grasps to keep.
    """

    def __init__(
        self,
        gripper_config: str,
        grasp_threshold: float,
        num_grasps: int,
        topk_num_grasps: int,
    ) -> None:
        self.grasp_cfg = load_grasp_cfg(gripper_config)
        self.gripper_name = self.grasp_cfg.data.gripper_name
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        self.grasp_threshold = grasp_threshold
        self.num_grasps = num_grasps
        self.topk_num_grasps = topk_num_grasps

    def auto_select_valid_cup_grasp(
        self, pointcloud: np.ndarray, qualifier: str = "cup"
    ) -> np.ndarray | None:
        """Repeatedly sample grasps until a qualified one is found."""
        min_point = np.percentile(pointcloud, 3, axis=0)
        max_point = np.percentile(pointcloud, 97, axis=0)
        try:
            num_try = 0
            while True:
                num_try += 1
                logging.info(f"try #{num_try}")
                grasps, _grasp_conf = (  # type: ignore[reportAssignmentType]
                    GraspGenSampler.run_inference(
                        pointcloud,
                        self.grasp_sampler,
                        grasp_threshold=self.grasp_threshold,
                        num_grasps=self.num_grasps,
                        topk_num_grasps=self.topk_num_grasps,
                    )
                )
                grasps = grasps.cpu().numpy()
                grasps[:, 3, 3] = 1
                qualified_grasps = np.array(
                    [
                        grasp
                        for grasp in grasps
                        if is_qualified(grasp, qualifier, min_point, max_point)
                    ]
                )
                if len(qualified_grasps) > 0:
                    return qualified_grasps[0]
        except KeyboardInterrupt:  # manual stop if too many fail
            logging.info("Manually stopping generating grasps")
            return None


def start_meshcat_server() -> subprocess.Popen[bytes]:
    """Starts the meshcat-server and registers a cleanup function."""
    print("Starting meshcat-server...")
    meshcat_server_process = subprocess.Popen(
        "meshcat-server", shell=True, preexec_fn=os.setsid
    )

    @atexit.register
    def cleanup_meshcat_server() -> None:  # type: ignore[reportUnusedFunction]
        if meshcat_server_process.poll() is None:
            print("Terminating meshcat-server...")
            os.killpg(os.getpgid(meshcat_server_process.pid), signal.SIGTERM)

    time.sleep(2)  # Wait for server to start
    return meshcat_server_process


def open_meshcat_url(url: str) -> None:
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


def get_left_up_and_front(
    grasp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract left, up, and front direction vectors from a 4x4 grasp matrix."""
    left = grasp[:3, 0]
    up = grasp[:3, 1]
    front = grasp[:3, 2]
    return left, up, front


class ControlPanel:
    """Tkinter GUI for browsing and selecting grasp poses.

    Args:
        root: The tkinter root window.
        vis: Meshcat visualizer instance.
        grasp_queue: Queue to send the selected grasp back to the caller.
        grasps_to_handle_queue: Queue receiving generated grasp batches.
        gripper_name: Name of the gripper model for visualization.
    """

    all_grasps: np.ndarray
    custom_filter_mask: np.ndarray
    collision_free_mask: np.ndarray
    current_grasp_pool: np.ndarray
    current_index: int

    def __init__(
        self,
        root: tk.Tk,
        vis: meshcat.Visualizer,
        grasp_queue: queue.Queue[np.ndarray | None],
        grasps_to_handle_queue: queue.Queue[tuple[np.ndarray, np.ndarray, np.ndarray]],
        gripper_name: str,
    ) -> None:
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
            self.root,
            text="Select Grasp",
            command=self._return_grasp_and_destroy,
        )
        select_button.pack(padx=20, pady=5)
        # retry button
        retry_button = tk.Button(
            self.root,
            text="Retry",
            command=self._return_none_and_retry,
        )
        retry_button.pack(padx=20, pady=5)

        self.apply_custom_filter_checkbox = tk.Checkbutton(
            self.root,
            text="Apply Custom Filter",
            variable=self.apply_custom_filter_var,
            command=self.toggle_grasp_display,
        )
        self.apply_custom_filter_checkbox.pack(pady=5)

        self.apply_collision_filter_checkbox = tk.Checkbutton(
            self.root,
            text="Apply Collision Filter",
            variable=self.apply_collision_filter_var,
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

    def run(self) -> None:
        """Start the tkinter main loop."""
        self.root.after(100, self._catch_grasps)
        self.root.mainloop()

    def next_grasp(self, event: object = None) -> None:
        """Advance to the next grasp in the current filtered pool."""
        if len(self.current_grasp_pool) == 0:
            return
        self.current_index = (self.current_index + 1) % len(self.current_grasp_pool)
        self._update_grasp_visualization()

    def prev_grasp(self, event: object = None) -> None:
        """Go back to the previous grasp in the current filtered pool."""
        if len(self.current_grasp_pool) == 0:
            return
        self.current_index = (self.current_index - 1) % len(self.current_grasp_pool)
        self._update_grasp_visualization()

    def toggle_grasp_display(self) -> None:
        """Recompute the visible grasp pool based on active filter checkboxes."""
        grasps_mask = np.array([True for _grasp in self.all_grasps])
        if self.apply_custom_filter_var.get():
            grasps_mask = grasps_mask & self.custom_filter_mask
        if self.apply_collision_filter_var.get():
            grasps_mask = grasps_mask & self.collision_free_mask
        self.current_index = 0
        self.current_grasp_pool = self.all_grasps[grasps_mask]
        self._update_grasp_visualization()

    def _return_grasp_and_destroy(self) -> None:
        self.grasp_queue.put(self.current_grasp_pool[self.current_index])
        self.root.destroy()

    def _return_none_and_retry(self) -> None:
        self.grasp_queue.put(None)
        self._catch_grasps()

    def _catch_grasps(self) -> None:
        self.all_grasps, self.custom_filter_mask, self.collision_free_mask = (
            self.grasps_to_handle_queue.get()
        )
        self.toggle_grasp_display()

    def _update_grasp_visualization(self) -> None:
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
    vis: meshcat.Visualizer,
    grasp_queue: queue.Queue[np.ndarray | None],
    grasps_to_handle_queue: queue.Queue[tuple[np.ndarray, np.ndarray, np.ndarray]],
    gripper_name: str,
) -> None:
    """Creates and runs the tkinter control panel."""
    root = tk.Tk()
    panel = ControlPanel(
        root,
        vis,
        grasp_queue,
        grasps_to_handle_queue,
        gripper_name,
    )
    panel.run()


def angle_offset_rad(grasp: np.ndarray) -> float:
    """Compute the angular offset between the grasp approach and position vectors."""
    position = grasp[:3, 3].tolist()
    _left, _up, front = get_left_up_and_front(grasp)
    angle_front = np.arctan2(front[1], front[0])
    angle_position = np.arctan2(position[1], position[0])
    angle_diff = np.abs(angle_front - angle_position)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    return angle_diff


class GraspGeneratorUI:
    """GraspGen inference with optional meshcat visualization and tkinter GUI.

    Attributes:
        scene_data (SceneData): Scene data for the current grasp generation.
        move (MoveItem): The current move item to generate grasps for.
        vis (meshcat.Visualizer): The meshcat visualizer instance.

    Args:
        gripper_config: Path to the gripper YAML config.
        grasp_threshold: Minimum confidence score for generated grasps.
        num_grasps: Number of grasps to sample per batch.
        topk_num_grasps: Number of top-scoring grasps to keep.
        need_GUI: Whether to launch the meshcat server and tkinter panel.
    """

    scene_data: SceneData
    move: MoveItem
    vis: meshcat.Visualizer

    def __init__(
        self,
        gripper_config: str,
        grasp_threshold: float,
        num_grasps: int,
        topk_num_grasps: int,
        need_GUI: bool = False,  # noqa: N803
    ) -> None:
        self.grasp_cfg = load_grasp_cfg(gripper_config)
        self.gripper_name: str = self.grasp_cfg.data.gripper_name
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        self.grasp_threshold: float = grasp_threshold
        self.num_grasps: int = num_grasps
        self.topk_num_grasps: int = topk_num_grasps
        self.need_GUI: bool = need_GUI
        ## Main init starts here
        if self.need_GUI:
            start_meshcat_server()
            open_meshcat_url("http://127.0.0.1:7000/static/")
            self.vis = create_visualizer()

    def _generate_grasps(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one batch of grasp inference and return masks for filtering."""
        obj_name = self.move.target_name
        if obj_name is None:
            raise ValueError("target_name must not be None")
        obj_pc = self.scene_data.object_infos[obj_name].points
        qualifier_name = self.move.qualifier
        if qualifier_name is None:
            raise ValueError("qualifier must not be None")
        # mass_center = np.mean(obj_pc, axis=0)
        # std = np.std(obj_pc, axis=0)
        grasps, _grasp_conf = GraspGenSampler.run_inference(
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
        min_point = np.percentile(obj_pc, 3, axis=0)
        max_point = np.percentile(obj_pc, 97, axis=0)
        custom_filter_mask = np.array(
            [
                is_qualified(grasp, qualifier_name, min_point, max_point)
                for grasp in grasps
            ]
        )
        collision_free_mask = np.array([True for _grasp in grasps])

        gripper_info = get_gripper_info(self.gripper_name)
        gripper_collision_mesh = gripper_info.collision_mesh

        # extract xyz_scene
        xyz_scene = np.array(self.scene_data.scene_info.pc_color)[0]
        for obj_name in self.scene_data.object_infos:
            if obj_name != self.move.target_name:
                logger.debug(f"Scene points before: {xyz_scene.shape[0]}")
                xyz_scene = np.vstack(
                    (xyz_scene, self.scene_data.object_infos[obj_name].points)
                )
                logger.debug(f"Scene points after: {xyz_scene.shape[0]}")
        # VIZ_BOUNDS
        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]  # noqa: N806
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
                f"Downsampled scene point cloud from"
                f" {len(xyz_scene)} to"
                f" {len(xyz_scene_downsampled)} points"
            )
        else:
            xyz_scene_downsampled = xyz_scene
            logger.debug(
                f"Scene point cloud has {len(xyz_scene)}"
                " points (no downsampling needed)"
            )
        collision_free_mask = filter_colliding_grasps(
            scene_pc=xyz_scene_downsampled,
            grasp_poses=grasps,
            gripper_collision_mesh=gripper_collision_mesh,
            collision_threshold=0.03,
        )
        return grasps, custom_filter_mask, collision_free_mask

    def _generate_grasp_silent(self) -> list[np.ndarray]:
        """Return qualified grasps without GUI interaction.

        Returns:
            A list of shape(4, 4) np.ndarray grasp matrices.
        """
        num_try = 0
        qualified_grasps: list[np.ndarray] = []
        while True:
            num_try += 1
            logger.info(f"try #{num_try}")
            all_grasps, custom_filter_mask, collision_free_mask = (
                self._generate_grasps()
            )
            qualified_grasps.extend(
                list(all_grasps[custom_filter_mask & collision_free_mask])
            )
            if len(qualified_grasps) > 0:
                qualified_grasps = sorted(qualified_grasps, key=angle_offset_rad)
                return qualified_grasps

    def generate_grasp(self, scene_data: SceneData, move: MoveItem) -> list[np.ndarray]:
        """Generate grasps for the given scene and move, with or without GUI.

        Returns:
            A list of 4x4 grasp pose matrices.
        """
        # reload scene_data and move
        self.scene_data = scene_data
        self.move = move

        if self.need_GUI:
            # create control panel and use GUI mode
            return self._generate_grasp_with_gui()
        else:
            return self._generate_grasp_silent()

    def _generate_grasp_with_gui(self) -> list[np.ndarray]:
        """Let the user select a grasp via the tkinter panel.

        Returns:
            A single-element list containing the selected 4x4 grasp matrix.
        """
        self._visualize_scene()
        grasp_q: queue.Queue[np.ndarray | None] = queue.Queue()
        grasps_to_handle_queue: queue.Queue[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = queue.Queue()
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

    def _create_control_panel(
        self,
        grasp_queue: queue.Queue[np.ndarray | None],
        grasps_to_handle_queue: queue.Queue[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """Creates and runs the tkinter control panel."""
        root = tk.Tk()
        panel = ControlPanel(
            root,
            self.vis,
            grasp_queue,
            grasps_to_handle_queue,
            self.gripper_name,
        )
        panel.run()

    def _visualize_scene(self) -> None:
        """Loads scene data, processes point clouds, and visualizes them."""
        xyz_scene = np.array(self.scene_data.scene_info.pc_color)[0]
        xyz_scene_color = np.array(self.scene_data.scene_info.img_color).reshape(
            1, -1, 3
        )[0, :, :]

        for obj_info in self.scene_data.object_infos.values():
            xyz_scene = np.vstack((xyz_scene, obj_info.points))
            xyz_scene_color = np.vstack((xyz_scene_color, obj_info.colors))

        VIZ_BOUNDS = [[-1.5, -1.25, -0.15], [1.5, 1.25, 2.0]]  # noqa: N806
        mask_within_bounds = np.all((xyz_scene > VIZ_BOUNDS[0]), 1)
        mask_within_bounds = np.logical_and(
            mask_within_bounds,
            np.all((xyz_scene < VIZ_BOUNDS[1]), 1),
        )
        xyz_scene = xyz_scene[mask_within_bounds]
        xyz_scene_color = xyz_scene_color[mask_within_bounds]

        visualize_pointcloud(
            self.vis, "pc_scene", xyz_scene, xyz_scene_color, size=0.0025
        )

    def _visualize_target(self) -> None:
        """Visualize the target object point cloud."""
        target_name = self.move.target_name
        if target_name is None:
            raise ValueError("target_name must not be None")
        target = self.scene_data.object_infos[target_name]
        visualize_pointcloud(
            self.vis, "pc_obj", target.points, target.colors, size=0.005
        )
