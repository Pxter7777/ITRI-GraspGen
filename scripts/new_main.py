import argparse
import logging

from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from PointCloud_Generation.PC_transform import silent_transform
from common_utils import config


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
        "--transform-config",
        type=str,
        default="demo5.json",
        help="transform-config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pc_generator = PointCloudGenerator(args)
    try:
        while True:
            text = input("text start to start")
            if text == "end":
                break
            # "start" to generate pointcloud
            pointcloud = None
            if text == "start":
                pointcloud = pc_generator.interactive_gui_mode()
            if pointcloud is None:
                continue

            # transform
            pointcloud = silent_transform(pointcloud, args.transform_config)
            print(pointcloud)

    except Exception as e:
        # unknown exception
        logging.error(f"An error occurred: {e}")
    finally:
        pc_generator.close()


if __name__ == "__main__":
    main()
