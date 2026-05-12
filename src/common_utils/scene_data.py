"""Lightweight scene data container shared across processes."""

from dataclasses import dataclass, field

import numpy as np

from common_utils.actions_format_checker import ObstacleBound


@dataclass
class SceneData:
    """Container for a single scene's point cloud, color, and grasp data."""

    @dataclass
    class ObjectInfo:
        """Per-object point cloud and color arrays."""

        points: np.ndarray
        colors: np.ndarray

    @dataclass
    class SceneInfo:
        """Full-scene point cloud colors and corresponding images."""

        pc_color: list[np.ndarray]
        img_color: list[np.ndarray]

    @dataclass
    class GraspInfo:
        """Grasp poses and their confidence scores."""

        grasp_poses: list[np.ndarray] = field(default_factory=list[np.ndarray])
        grasp_conf: list[float] = field(default_factory=list[float])

    object_infos: dict[str, ObjectInfo]
    scene_info: SceneInfo
    obstacle_infos: dict[str, ObstacleBound] = field(
        default_factory=dict[str, ObstacleBound]
    )
    grasp_info: GraspInfo = field(default_factory=GraspInfo)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
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
