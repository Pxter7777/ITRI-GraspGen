import os
import argparse
import logging
import json
from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from PointCloud_Generation.PC_transform import silent_transform_multiple_obj_with_name
from common_utils import config
from common_utils.graspgen_utils import GraspGenerator
from common_utils.gripper_utils import send_moves_to_robot
from common_utils.actions_format_checker import is_actions_format_valid
from common_utils.movesets import act

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    force=True,
)
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
    logger.info("starting the program")
    args = parse_args()
    # Directory path handle
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_file_dir)
    try:
        pc_generator = PointCloudGenerator(args)
        grasp_generator = GraspGenerator(
            args.gripper_config,
            args.grasp_threshold,
            args.num_grasps,
            args.topk_num_grasps,
        )
        while True:
            print("Please provide the <name> of actions to start, or type end to end.")
            text = input("./actions/<name>.json: ")
            if text == "end":
                break
            actions_filepath = os.path.join(project_root_dir, "actions", text + ".json")
            actions = []
            try:
                with open(actions_filepath, "rb") as f:
                    actions = json.load(f)
                if not is_actions_format_valid(actions):
                    logger.error("bad actions file format")
                    continue
            except Exception as e:
                logger.exception(e)
                logger.error("failed reading file, try again.")
                continue
            # start to generate pointcloud
            scene_data = None
            target_names = [action["target_name"] for action in actions]
            try:
                scene_data = pc_generator.silent_mode_multiple_grounding(target_names)
            except Exception:
                continue
            logger.info(scene_data)
            # transform
            scene_data = silent_transform_multiple_obj_with_name(
                scene_data, args.transform_config
            )
            # objects_pointcloud = [pcs["pc"] for pcs in transformed_pointcloud["objects_info"]]
            # target_objects = [obj for obj in scene_data["object_infos"]]
            # GraspGen
            for obj, action in zip(scene_data["object_infos"], actions, strict=False):
                try:
                    grasp = grasp_generator.flexible_auto_select_valid_grasp(
                        obj, action["qualifier"]
                    )
                except Exception:
                    logger.error(
                        f"Error while generating grasp for {obj['name']}, stopping."
                    )
                    break
                moves = act(action["action"], grasp, action["args"])
                # send the grasp to gripper
                try:
                    send_moves_to_robot(moves)
                except KeyboardInterrupt:
                    logger.info("Manual stopping gripper.")
                    break
    finally:
        logger.info("turning off zed camera")
        pc_generator.close()
        logger.info("terminating process")


if __name__ == "__main__":
    main()
