import json
import datetime
import logging
import copy
import numpy as np

from pathlib import Path
from common_utils.actions_format_checker import ObstacleBound
from common_utils.scene_data import SceneData


logger = logging.getLogger(__name__)

# Project root dir
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]


def save_json(dir: str, prefix: str, data) -> True:  # save json data for test
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = (
        PROJECT_ROOT_DIR / "data_for_test" / dir / (prefix + current_time_str + ".json")
    )
    logger.info(f"save to {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def create_obstacle_info(
    scene_data: SceneData,
    extra_obstacles: dict[str, ObstacleBound] | None = None,
) -> SceneData:
    # Copy from .json
    if extra_obstacles is None:
        extra_obstacles = {}
    new_scene_data = copy.deepcopy(scene_data)
    new_scene_data.obstacle_infos = dict()
    for extra_obstacle_name in extra_obstacles:
        obstacle = extra_obstacles[extra_obstacle_name]
        new_scene_data.obstacle_infos[extra_obstacle_name] = obstacle
    # Dynamically generated obstacles
    for object_name in new_scene_data.object_infos:
        obj_pc = new_scene_data.object_infos[object_name].points
        pc_max, pc_min = (
            np.percentile(obj_pc, 97, axis=0),
            np.percentile(obj_pc, 3, axis=0),
        )
        middle_point = np.mean([pc_max, pc_min], axis=0)
        width = pc_max - middle_point
        mod_width = np.array([width[0] * 1.6, width[1] * 1.6, width[2] * 1.9])
        pc_max = middle_point + mod_width
        pc_min = middle_point - mod_width
        new_scene_data.obstacle_infos[object_name] = ObstacleBound(
            max=pc_max.tolist(), min=pc_min.tolist()
        )

    return new_scene_data


def load_extra_obstacles() -> dict[str, ObstacleBound]:
    extra_obstacles: dict[str, ObstacleBound] = dict()
    extra_obstacles_path = (
        PROJECT_ROOT_DIR / "configs" / "actions" / "extra_obstacles.json"
    )
    with open(extra_obstacles_path, "rb") as f:
        obstacles_dict = json.load(f)
        for name, obstacle in obstacles_dict.items():
            extra_obstacles[name] = ObstacleBound(**obstacle)
    return extra_obstacles
