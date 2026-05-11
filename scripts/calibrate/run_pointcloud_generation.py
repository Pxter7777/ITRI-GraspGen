import argparse
import datetime
import json
import logging
from pathlib import Path

from common_utils import config
from common_utils.custom_logger import CustomFormatter
from pointcloud_generation.pointcloud_generation import PointCloudGenerator

# root logger setup
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)


def parse_args():
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
        "--output-tag",
        default="",
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--erosion_iterations",
        type=int,
        default=0,  # can be 6
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
    parser.add_argument(
        "--use-png",
        type=str,
        default="",
        help="Use exisiting images at data/zed_images instead of the real zed camera",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pc_generator = PointCloudGenerator(args)
    scene_data = pc_generator.interactive_gui_mode()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"scene_{timestamp}.json"
    json_filepath = Path("./data/calibrate/output_pointcloud/") / json_filename

    with open(json_filepath, "w") as f:
        json.dump(scene_data.to_dict(), f, indent=4)
    logger.info(f"Scene saved to {json_filepath}")
    pc_generator.close()


if __name__ == "__main__":
    main()
