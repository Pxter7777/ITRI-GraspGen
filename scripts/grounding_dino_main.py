import argparse

from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from PointCloud_Generation.PC_transform import silent_transform, silent_transform_multiple
from common_utils import config
from common_utils.graspgen_utils import GraspGenerator
from common_utils.gripper_utils import send_cup_grasp_to_robot


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
    parser.add_argument(
        "--gripper_config",
        type=str,
        default=str(config.GRIPPER_CFG),
        help="Path to gripper configuration YAML file",
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.70,
        help="Threshold for valid grasps. If -1.0, then the top 100 grasps will be ranked and returned",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="Number of grasps to generate",
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="Whether to return only the top k grasps",
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=5,
        help="Number of top grasps to return when return_topk is True",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pc_generator = PointCloudGenerator(args)
    grasp_generator = GraspGenerator(
        args.gripper_config, args.grasp_threshold, args.num_grasps, args.topk_num_grasps
    )
    try:
        while True:
            text = input("text start to start")
            if text == "end":
                break
            # "start" to generate pointcloud
            scene_data = None
            if text == "start":
                scene_data = pc_generator.silent_mode_multiple_grounding()
            if scene_data is None:
                continue
            print(scene_data)
            # transform
            scene_data = silent_transform_multiple(scene_data, args.transform_config)
            objects_pointcloud = [pcs["pc"] for pcs in transformed_pointcloud["objects_info"]]
            # GraspGen
            for object_pointcloud in objects_pointcloud:
                grasp = grasp_generator.auto_select_valid_cup_grasp(
                    object_pointcloud
                )
                if grasp is None:
                    continue

                # send the grasp to gripper
                send_cup_grasp_to_robot(grasp)
    finally:
        pc_generator.close()


if __name__ == "__main__":
    main()
