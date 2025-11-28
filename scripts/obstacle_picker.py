# use the zed camera to get the object's 3D bounding box
import logging
import argparse
import numpy as np
from common_utils.common_utils import save_json
from common_utils.custom_logger import CustomFormatter
from common_utils import config
from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from PointCloud_Generation.PC_transform import silent_transform

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
        "--use-png",
        type=str,
        default="",
        help="Use exisiting images at sample_data/zed_images instead of the real zed camera",
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
        "--transform-config",
        type=str,
        default="sim2.json",
        help="transform-config",
    )
    return parser.parse_args()


def main():
    logger.info("starting the program")
    args = parse_args()
    pc_generator = PointCloudGenerator(args)
    scene_data = pc_generator.interactive_gui_mode()
    scene_data = silent_transform(scene_data, args.transform_config)
    pc = scene_data["object_info"]["pc"]
    pc_max, pc_min = np.percentile(pc, 97, axis=0), np.percentile(pc, 3, axis=0)
    obstacle_name = input("./obstacle/<obstacle_name>.json: ")
    obstacle = {obstacle_name: {"max": pc_max.tolist(), "min": pc_min.tolist()}}
    save_json("obstacle", "obstacle", obstacle)


if __name__ == "__main__":
    main()
