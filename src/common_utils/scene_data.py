"""Lightweight scene data container shared across processes."""

from dataclasses import dataclass, field

import numpy as np

from common_utils.actions_format_checker import ObstacleBound


@dataclass
class SceneData:
    """Container for a single scene's point cloud, color, and grasp data.

    Attributes:
        object_infos (dict[str, ObjectInfo]): Per-object point cloud data.
        scene_info (SceneInfo): Full-scene point cloud and image data.
        obstacle_infos (dict[str, ObstacleBound]): Obstacle bounds by name.
        grasp_info (GraspInfo): Grasp poses and confidence scores.
    """

    @dataclass
    class ObjectInfo:
        """Per-object point cloud and color arrays.

        Attributes:
            points (np.ndarray): Object point cloud positions.
            colors (np.ndarray): Object point cloud colors.
        """

        points: np.ndarray
        colors: np.ndarray

    @dataclass
    class SceneInfo:
        """Full-scene point cloud colors and corresponding images.

        Attributes:
            pc_color (list[np.ndarray]): Scene point clouds.
            img_color (list[np.ndarray]): Scene images.
        """

        pc_color: list[np.ndarray]
        img_color: list[np.ndarray]

    @dataclass
    class GraspInfo:
        """Grasp poses and their confidence scores.

        Attributes:
            grasp_poses (list[np.ndarray]): List of 4x4 grasp matrices.
            grasp_conf (list[float]): Confidence scores per grasp.
        """

        grasp_poses: list[np.ndarray] = field(default_factory=list[np.ndarray])
        grasp_conf: list[float] = field(default_factory=list[float])

    object_infos: dict[str, ObjectInfo]
    scene_info: SceneInfo
    obstacle_infos: dict[str, ObstacleBound] = field(
        default_factory=dict[str, ObstacleBound]
    )
    grasp_info: GraspInfo = field(default_factory=GraspInfo)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            dict[str, object]: A nested dictionary of scene data.
        """
        return {
            "object_infos": {
                name: {"pc": info.points.tolist(), "pc_color": info.colors.tolist()}
                for name, info in self.object_infos.items()
            },
            "scene_info": {
                "pc_color": [arr.tolist() for arr in self.scene_info.pc_color],
                "img_color": [arr.tolist() for arr in self.scene_info.img_color],
            },
            "obstacle_infos": {
                name: obs.model_dump() for name, obs in self.obstacle_infos.items()
            },
            "grasp_info": {
                "grasp_poses": self.grasp_info.grasp_poses,
                "grasp_conf": self.grasp_info.grasp_conf,
            },
        }
