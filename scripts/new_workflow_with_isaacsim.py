import os
import argparse
import logging
import json
import time
from pathlib import Path
from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from PointCloud_Generation.PC_transform import (
    silent_transform_multiple_obj_with_name_dict,
)
from common_utils import config, network_config
from common_utils.graspgen_utils import GraspGeneratorUI
from common_utils.actions_format_checker import TaskConfig, ObstacleBound
from common_utils.movesets import act_with_name
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from common_utils.custom_logger import CustomFormatter
from common_utils.common_utils import save_json, create_obstacle_info

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
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="decide if we need confirm for groundingDINO detect and grasp Generation",
    )
    parser.add_argument(
        "--save-fullact",
        action="store_true",
        help="save the fullact",
    )
    parser.add_argument(
        "--use-png",
        type=str,
        default="",
        help="Use exisiting images at sample_data/zed_images instead of the real zed camera",
    )
    return parser.parse_args()

class GraspGenController:
    def __init__(self, args) -> None:
        self.args = args
        self.sender = NonBlockingJSONSender(port=network_config.GRASPGEN_TO_ISAACSIM_PORT)
        self.receiver = NonBlockingJSONReceiver(
            port=network_config.ISAACSIM_TO_GRASPGEN_PORT
        )
        self.pc_generator = PointCloudGenerator(self.args)
        self.grasp_generator = GraspGeneratorUI(
            args.gripper_config,
            args.grasp_threshold,
            args.num_grasps,
            args.topk_num_grasps,
            not args.no_confirm,
        )
        logger.info("======Successfully initialized======")
    def __enter__(self):
        """Allows the use of 'with GraspGenController(args) as controller:'"""
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Automatically called when the 'with' block ends.
        Even if an exception occurs, this method runs.
        """
        if exc_type:
            logger.error(f"Exiting due to error: {exc_val}")
        
        self._close()
        # Returning False allows the exception to continue propagating 
        # so you still see the traceback after cleanup.
        return False
    def handle_task_command(self):
        task_type, task_name = self._capture_task_name()
        # eat abort if there is any
        for _ in range(5):
            self.receiver.capture_data()
        if task_type == "GraspGen" and task_name == "Grasp_and_Dump":
            GraspGen_filepath = Path(__file__).resolve().parents[1] / "actions" / "Grasp_and_Dump.json"
            

    def _capture_task_name(self) -> tuple[str, str]:
        print("Please provide the command, or type 'end' to end.")
        text = input("Command: ")
        if text == "end":
            raise KeyboardInterrupt
        if text == "Grasp_and_Dump":
            return "GraspGen", text
        # check if it's in csv
        base_dir = Path("~").expanduser() / "RobotSnackServing-csv"
        if not base_dir.exists():
            raise FileNotFoundError(f"~/RobotSnackServing is missing. Is https://github.com/hongalicia/RobotSnackServing-csv.git cloned into ~/ ?")
        traj_dir = base_dir / "trajectories"
        csv_paths = list(traj_dir.glob("*.csv"))
        filenames = [path.stem for path in csv_paths]
        if text in filenames:
            return "csv", text
        else:
            avail_commands = ["Grasp_and_Dump"] + filenames
            print(f"There's no such command called {text}. \nAvailable commands are {avail_commands}")
            return self._capture_task_name()
    def _close(self):
        logger.info("Cleaning up resources...")
        try:
            self.pc_generator.close()
        except Exception as e:
            logger.exception(f"Error during pc_generator cleanup: {e}")
def main():
    args = parse_args()
    with GraspGenController(args) as controller:
        while True:
            controller.handle_task_command()

if __name__ == "__main__":
    main()