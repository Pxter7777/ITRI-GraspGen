import argparse
import logging
import numpy as np
import json
import trimesh
import trimesh.transformations as tra
from pathlib import Path
from common_utils.custom_logger import CustomFormatter
from common_utils import config
from common_utils.graspgen_utils import start_meshcat_server, open_meshcat_url
from common_utils.order_task_config import OrderTaskConfig
from grasp_gen.grasp_server import load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_mesh,
)

# root logger setup
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
logger = logging.getLogger(__name__)

# Project root dir
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grasp data for a given scene."
    )
    parser.add_argument(
        "--class",
        type=str,
        required=True,
        help="choose one: on_table, on_shelf, in_basket",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="provide the name, for example, `1`",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default=str(config.GRIPPER_CFG),
        help="Path to gripper configuration YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scene_class = getattr(args, "class")
    name = args.name

    task_config_path = (
        PROJECT_ROOT_DIR
        / "data"
        / "order_experiment_data"
        / scene_class
        / "task_config"
        / f"{name}.json"
    )
    grasp_data_path = (
        PROJECT_ROOT_DIR
        / "data"
        / "order_experiment_data"
        / scene_class
        / "grasp_data"
        / f"{name}.json"
    )

    # Load task config
    with open(task_config_path) as f:
        task = OrderTaskConfig(**json.load(f))

    # Load grasp data (grasp_pose stored as list-of-lists, convert back to ndarray)
    with open(grasp_data_path) as f:
        grasp_pack_dict = json.load(f)
    grasps = [np.array(g["grasp_pose"]) for g in grasp_pack_dict["grasps"]]

    # Get gripper name from config
    gripper_name = load_grasp_cfg(args.gripper_config).data.gripper_name

    # Start meshcat
    start_meshcat_server()
    open_meshcat_url("http://127.0.0.1:7000/static/")
    vis = create_visualizer()

    # Visualize all object meshes (target + obstacles) in world frame
    for obj in [task.target] + task.obstacles:
        mesh_file = Path(obj.obj_dir) / "mesh.obj"
        obj_mesh = trimesh.load(str(mesh_file), force="mesh", process=False)
        obj_mesh.apply_scale(obj.scale)
        qx, qy, qz, qw = obj.pose_meter_quat[3:]
        T_world = tra.quaternion_matrix([qw, qx, qy, qz])
        T_world[:3, 3] = obj.pose_meter_quat[:3]
        obj_mesh.apply_transform(T_world)
        visualize_mesh(vis, f"scene/{obj.instance_id}", obj_mesh)

    # Visualize all grasps
    for i, grasp in enumerate(grasps):
        visualize_grasp(vis, f"grasps/{i:03d}", grasp, gripper_name=gripper_name)

    logger.info(f"Visualizing {len(grasps)} grasps. Open meshcat to inspect.")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
