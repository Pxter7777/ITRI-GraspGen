import argparse
import logging
import numpy as np
import torch
import json
import trimesh
import trimesh.transformations as tra
from pathlib import Path
from common_utils import network_config
from common_utils.custom_logger import CustomFormatter
from common_utils.workflow_control import BaseWorkflowController
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from common_utils import config
from common_utils.graspgen_utils import GraspGeneratorUI, flip_upside_down_grasps
from common_utils.order_task_config import OrderTaskConfig
from grasp_gen.grasp_server import GraspGenSampler
from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format
from common_utils.grasp_data_format import GraspPack, GraspData
# root logger setup
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)

# Project root dir
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


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
        "--need-confirm",
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




class ExperimentWorkflowController:
    def __init__(self, args) -> None:
        self.args = args
        self.grasp_generator = GraspGeneratorUI(
            args.gripper_config,
            args.grasp_threshold,
            args.num_grasps,
            args.topk_num_grasps,
            args.need_confirm,
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
    def _close(self):
        logger.info("Cleaning up resources...")
        try:
            pass # nothing to do, actually.
        except Exception as e:
            logger.exception(f"Error during pc_generator cleanup: {e}")
    
    def run_experiment(self):
        for dir in ["in_basket", "on_shelf", "on_table"]:
            task_config_path = PROJECT_ROOT_DIR / "data" / "order_experiment_data" / dir / "task_config"
            json_paths = list(task_config_path.glob("*.json"))
            for json_path in json_paths:
                grasp_data_pack = self._generate_grasp_datas(json_path)
    def _generate_grasp_datas(self, json_path: Path) -> GraspPack:
        with open(json_path, "rb") as f:
            task_json = json.load(f)
        task = OrderTaskConfig(**task_json)

        # Load target mesh and sample a point cloud from its surface
        mesh_file = Path(task.target.obj_dir) / "mesh.obj"
        logger.info(f"Loading mesh: {mesh_file}")
        obj = trimesh.load(str(mesh_file), force="mesh", process=False)
        obj.apply_scale(task.target.scale)

        xyz, _ = trimesh.sample.sample_surface(obj, 2000)
        xyz = np.array(xyz, dtype=np.float32)

        # Center the point cloud (GraspGen expects centered input)
        T_subtract_mean = tra.translation_matrix(-xyz.mean(axis=0))
        xyz = tra.transform_points(xyz, T_subtract_mean)

        # Run grasp inference
        grasps, grasp_conf = GraspGenSampler.run_inference(
            xyz,
            self.grasp_generator.grasp_sampler,
            grasp_threshold=0.0,
            num_grasps=200,
            # topk_num_grasps=5,
            min_grasps=80,
            max_tries=20,
        )
        if len(grasps) == 0:
            logger.warning(f"No grasps found for {json_path.name}")
            return None

        grasp_conf = grasp_conf.cpu().numpy()
        grasps = grasps.cpu().numpy()
        grasps[:, 3, 3] = 1
        grasps = flip_upside_down_grasps(grasps)
        logger.info(f"Generated {len(grasps)} grasps (scores {grasp_conf.min():.3f}–{grasp_conf.max():.3f})")

        # Transform grasps back from centered mesh frame to original mesh frame
        T_restore = tra.inverse_matrix(T_subtract_mean)
        grasps = np.array([T_restore @ g for g in grasps])

        # Save results next to the task config
        return GraspPack(
            num_grasps=len(grasps),
            grasps=[GraspData(grasp_pose=grasp, curobo_success="Unknown", collision_detected_by_graspgen=False) for grasp in grasps]
        )


def main():
    args = parse_args()
    set_seed(42)
    with ExperimentWorkflowController(args) as controller:
        controller.run_experiment()


if __name__ == "__main__":
    main()
