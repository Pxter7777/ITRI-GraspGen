import os
import argparse
import logging
import numpy as np
import cv2
import torch
import pyzed.sl as sl
import open3d as o3d
import json
import datetime
import pye57
import sys

sys.path.insert(0, os.path.expanduser("~/Third_Party"))

from scipy.spatial.transform import Rotation as R
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from src.stereo_utils2 import FoundationStereoModel
from src.yolo_inference import YOLOv5Detector
from src import (
    config,
    mouse_handler,
    sam_utils,
    visualization,
)
from src.zed_utils import ZedCamera

zed = None
###### Models
stereo_model = None
sam_predictor = None
yolo_detector = None
grasp_cfg = None
gripper_name = None
grasp_sampler = None
def set_logging_format():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def depth2xyzmap(depth, K):
    vy, vx = np.meshgrid(
        np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij"
    )
    x_map = (vx - K[0, 2]) * depth / K[0, 0]
    y_map = (vy - K[1, 2]) * depth / K[1, 1]
    z_map = depth
    xyz_map = np.stack([x_map, y_map, z_map], axis=-1)
    return xyz_map


def save_json_object_and_scene(
    out_dir,
    output_tag,
    object_points,
    object_colors,
    scene_points,
    scene_colors,
    timestamp,
):
    """Saves the scene and object data to a JSON file."""
    object_colors_arr = np.array(object_colors)
    if object_colors_arr.size > 0:
        object_colors_arr = object_colors_arr[:, ::-1]

    scene_colors_arr = np.array(scene_colors)
    if scene_colors_arr.size > 0:
        scene_colors_arr = scene_colors_arr[:, ::-1]

    scene_data = {
        "object_info": {
            "pc": np.array(object_points).tolist(),
            "pc_color": object_colors_arr.tolist(),
        },
        "scene_info": {
            "pc_color": [np.array(scene_points).tolist()],
            "img_color": [scene_colors_arr.tolist()],
        },
        "grasp_info": {"grasp_poses": [], "grasp_conf": []},
    }

    json_filename = f"scene_{timestamp}.json"
    if output_tag != "":
        json_filename = f"scene_{output_tag}.json"
    json_filepath = os.path.join(out_dir, json_filename)

    with open(json_filepath, "w") as f:
        json.dump(scene_data, f, indent=4)
    logging.info(f"Scene saved to {json_filepath}")



def generate_zed_point_cloud(
    args,
    K_cam,
    depth,
    color_np_org,
    mask,
    captured_vis,
):
    logging.info("Saving scene and object...")

    K_scaled_cam = K_cam.copy()
    K_scaled_cam[:2, :] *= args.scale

    xyz_map = depth2xyzmap(depth, K_scaled_cam)
    points_in_cam_view = xyz_map.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_in_cam_view)
    pcd = pcd.remove_non_finite_points()

    points_post_filter = np.asarray(pcd.points)

    if not points_post_filter.size:
        logging.warning("No valid points in the disparity map to save.")
        return

    # Filter out points with zero or negative depth to prevent division by zero
    z_coords = points_post_filter[:, 2]
    valid_depth_mask = z_coords > 1e-6
    points_post_filter = points_post_filter[valid_depth_mask]

    if not points_post_filter.size:
        logging.warning("No valid points after filtering for depth > 0.")
        return

    if args.max_depth is not None:
        valid_depth_mask = points_post_filter[:, 2] < args.max_depth
        points_post_filter = points_post_filter[valid_depth_mask]
        if not points_post_filter.size:
            logging.warning(
                f"No points remaining after filtering with max_depth={args.max_depth}"
            )
            return

    # Since the depth map is already in the color camera's frame, no transformation is needed.
    # We just need to map points to pixels to get their color and check the mask.
    projected_points_uv = (K_cam @ points_post_filter.T).T
    projected_points_uv[:, :2] /= projected_points_uv[:, 2:]

    scene_points = []
    scene_colors = []
    object_points = []
    object_colors = []

    H_color, W_color = color_np_org.shape[:2]

    for i in range(len(projected_points_uv)):
        u, v = int(projected_points_uv[i, 0]), int(projected_points_uv[i, 1])
        if 0 <= u < W_color and 0 <= v < H_color:
            point = points_post_filter[i]
            color = color_np_org[v, u]

            if mask[v, u]:
                object_points.append(point)
                object_colors.append(color)
            else:
                scene_points.append(point)
                scene_colors.append(color)

    if not object_points:
        logging.warning(
            "The selected mask contains no points from the point cloud. Nothing to save."
        )
    scene_points = [[z, -x, -y] for x, y, z in scene_points]
    object_points = [[z, -x, -y] for x, y, z in object_points]

    """Saves the scene and object data to a JSON file."""
    object_colors_arr = np.array(object_colors)
    if object_colors_arr.size > 0:
        object_colors_arr = object_colors_arr[:, ::-1]

    scene_colors_arr = np.array(scene_colors)
    if scene_colors_arr.size > 0:
        scene_colors_arr = scene_colors_arr[:, ::-1]

    scene_data = {
        "object_info": {
            "pc": np.array(object_points),
            "pc_color": object_colors_arr,
        },
        "scene_info": {
            "pc_color": [np.array(scene_points)],
            "img_color": [scene_colors_arr],
        },
        "grasp_info": {"grasp_poses": [], "grasp_conf": []},
    }
    return scene_data



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
        "--save-e57",
        action="store_true",
        help="save e57",
    )
    parser.add_argument(
        "--exit-after-save",
        action="store_true",
        help="exit after first save",
    )
    parser.add_argument(
        "--use-yolo",
        action="store_true",
        help="use yolo to pick",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=3,
        help="Maximum depth in meters.",
    )
    ### GraspGen
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default=str(config.GRASPGEN_SCENE_DIR),
        help="Directory containing JSON files with point cloud data",
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
        "--transform-config",
        type=str,
        default="demo4.json",
        help="Transform config",
    )
    return parser.parse_args()


def gen_point_cloud(args):
    
    try:
        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        zed_status, left_image, right_image = zed.capture_images()
        left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
        right_gray = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2GRAY)
        color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
        color_np_org = color_np.copy()
        mask = np.zeros_like(color_np[:, :, 0], dtype=bool)

        df = yolo_detector.infer(color_np_org)

        cup_detections = df[df["name"] == "cup"]
        if cup_detections.empty:
            print("No Cup")
            raise ValueError
        cup_box = cup_detections.iloc[0]
        box = (
            int(cup_box["xmin"]),
            int(cup_box["ymin"]),
            int(cup_box["xmax"]),
            int(cup_box["ymax"]),
        )
        # mask
        color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        mask = sam_utils.run_sam2(
            sam_predictor,
            color_np_rgb,
            box,
            iterations=args.erosion_iterations,
        )
        # zed inference
        depth, (H_scaled, W_scaled) = stereo_model.run_inference(
            left_gray, right_gray, zed.K_left, zed.baseline
        )
        # gen pointcloud
        return generate_zed_point_cloud(
            args,
            zed.K_left,
            depth,
            color_np_org,
            mask,
            color_np_org,
        )


    except:
        print("failed during pointcloud")
        return None

def transform(original_pointcloud, transformation_matrix):
    original_pc_homogeneous = np.hstack(
        (
            original_pointcloud,
            np.ones((original_pointcloud.shape[0], 1)),
        )
    )
    transformed_object_pc_homogeneous = (
        transformation_matrix @ original_pc_homogeneous.T
    ).T
    transformed_pointcloud = transformed_object_pc_homogeneous[:, :3]
    return transformed_pointcloud

def quick_transform(original_pointcloud, transform_config):
    # check and load transform config
    if transform_config == "":
        print("Please provide transform config")
        exit(1)
    transform_filename = os.path.join("transform_config", transform_config)
    with open(transform_filename, "rb") as f:
        transform_data = json.load(f)

    # Build Transformation matrix
    translation_matrix = np.array(
        [
            [1, 0, 0, transform_data["tx"]],
            [0, 1, 0, transform_data["ty"]],
            [0, 0, 1, transform_data["tz"]],
            [0, 0, 0, 1],
        ]
    )
    rotation = R.from_euler(
        "xyz",
        [transform_data["rr"], transform_data["rp"], transform_data["ry"]],
        degrees=True,
    )
    rotation_matrix = np.identity(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    transformation = translation_matrix @ rotation_matrix

    # check and load pointcloud

    info = original_pointcloud
    info["object_info"]["pc"] = transform(info["object_info"]["pc"], transformation)
    info["scene_info"]["pc_color"] = [transform(np.array(info["scene_info"]["pc_color"][0]) , transformation)]



    return info


def main():
    args = parse_args()
    global zed, stereo_model, sam_predictor, yolo_detector, grasp_cfg, gripper_name, grasp_sampler
    zed = ZedCamera()
    stereo_model = FoundationStereoModel(args)
    sam_predictor = sam_utils.load_sam_model()
    yolo_detector = YOLOv5Detector(model_path="/home/j300/models/YOLOModels/yolov5x.pt",conf=0.4)

    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    grasp_sampler = GraspGenSampler(grasp_cfg)

    try:
        while True:
            text = input("text start to start")
            if text == "start":
                pointcloud = gen_point_cloud(args)
                if pointcloud is None:
                    continue
                transform_pointcloud = quick_transform(pointcloud, args.transform_config)
                print("what?")
            elif text == "end":
                break
            
    except KeyboardInterrupt:
        zed.close()
    finally:
        zed.close()


    

if __name__ == "__main__":
    main()
