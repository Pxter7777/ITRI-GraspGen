"""IMPORTANT!

This script won't work, because the transform config logic and path changed a bit.
also the task format was changed, so I will need to rewrite this anyway.
move it to scripts/legacy later.
"""

# use the zed camera to get the object's 3D bounding box
import argparse
import logging

import numpy as np

from common_utils import config
from common_utils.common_utils import save_json
from common_utils.log_formatter import CustomLoggingFormatter
from pointcloud_generation.pc_transform import (
    silent_transform_multiple_obj_with_name_dict,
)
from pointcloud_generation.pointcloud_generation import PointCloudGenerator

# root logger setup
handler = logging.StreamHandler()
handler.setFormatter(CustomLoggingFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Manually transform a point cloud.")
    parser.add_argument(
        "--ckpt_dir",
        default=str(config.FOUNDATIONSTEREO_CHECKPOINT),
        type=str,
        help="pretrained model path",
    )

    parser.add_argument(
        "--scale",
        default=1,
        type=float,
        help="downsize the image by scale, must be <=1",
    )
    parser.add_argument("--hiera", default=0, type=int, help="hierarchical inference")
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--out_dir", default="./output/", type=str, help="the directory to save results"
    )
    parser.add_argument(
        "--use-png",
        type=str,
        default="",
        help="Use exisiting images at data/zed_images instead of the real zed camera",
    )
    parser.add_argument(
        "--erosion_iterations",
        type=int,
        default=1,  # can be 6
        help="Number of erosion iterations for the SAM mask.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=3.0,
        help="max depth for generating pointcloud",
    )
    parser.add_argument(
        "--need-confirm",
        action="store_true",
        help="decide if we need confirm for groundingDINO detect and grasp Generation",
    )
    return parser.parse_args()


def main() -> None:
    """Capture an obstacle bounding box from a point cloud and save it.

    Raises:
        ValueError: If scene_data is None or has unexpected object count.
    """
    logger.info("starting the program")
    args = parse_args()
    pc_generator = PointCloudGenerator(args)
    scene_data = pc_generator.interactive_gui_mode()
    if scene_data is None:
        raise ValueError("scene_data is None")
    scene_data = silent_transform_multiple_obj_with_name_dict(scene_data)
    if len(scene_data.object_infos) != 1:
        raise ValueError(f"Expected 1 object, got {len(scene_data.object_infos)}")
    pc = next(iter(scene_data.object_infos.values())).points
    pc_max, pc_min = np.percentile(pc, 97, axis=0), np.percentile(pc, 3, axis=0)
    obstacle_name = input("./data/calibrate/<obstacle_name>.json: ")
    obstacle = {obstacle_name: {"max": pc_max.tolist(), "min": pc_min.tolist()}}
    save_json("calibrate", "obstacle", obstacle)


if __name__ == "__main__":
    main()
