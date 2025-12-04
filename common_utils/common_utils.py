import json
import datetime
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


def save_json(dir: str, prefix: str, data) -> True:  # save json data for test
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_file_dir)
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(
        project_root_dir, "data_for_test", dir, prefix + current_time_str + ".json"
    )
    logger.info(f"save to {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def create_obstacle_info(scene_data: dict, extra_obstacles: dict = None) -> dict:
    if extra_obstacles is None:
        extra_obstacles = {}
    new_scene_data = scene_data  # it's lazy that this isn't full copy, but I think it's fine to keep it this way for now.
    new_scene_data["obstacles"] = extra_obstacles
    for object_name in new_scene_data["object_infos"]:
        obj_pc = new_scene_data["object_infos"][object_name]["points"]
        pc_max, pc_min = (
            np.percentile(obj_pc, 97, axis=0),
            np.percentile(obj_pc, 3, axis=0),
        )
        middle_point = np.mean([pc_max, pc_min], axis=0)
        pc_max = (pc_max - middle_point) * 1.6 + middle_point
        pc_min = (pc_min - middle_point) * 1.6 + middle_point
        new_scene_data["obstacles"][object_name] = {
            "max": pc_max.tolist(),
            "min": pc_min.tolist(),
        }

    return new_scene_data
