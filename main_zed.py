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

from src.stereo_utils2 import FoundationStereoModel
from src import (
    config,
    mouse_handler,
    sam_utils,
    visualization,
)
from src.zed_utils import ZedCamera


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
    out_dir, output_tag, object_points, object_colors, scene_points, scene_colors, timestamp
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


def save_e57_object_and_scene(
    out_dir, output_tag, object_points, object_colors, scene_points, scene_colors, timestamp
):
    """Saves the scene points and colors to a .e57 file."""
    e57_object_filename = f"object_{timestamp}.e57"
    if output_tag!="":
        e57_object_filename = f"object_{output_tag}.e57"
    e57_object_filepath = os.path.join(out_dir, e57_object_filename)

    object_points = np.array(object_points)
    object_colors = np.array(object_colors)

    object_data = dict()
    object_data["cartesianX"] = object_points[:, 0]
    object_data["cartesianY"] = object_points[:, 1]
    object_data["cartesianZ"] = object_points[:, 2]
    object_data["colorRed"] = object_colors[:, 2]
    object_data["colorGreen"] = object_colors[:, 1]
    object_data["colorBlue"] = object_colors[:, 0]

    with pye57.E57(e57_object_filepath, mode="w") as e57_write:
        e57_write.write_scan_raw(object_data)

    e57_scene_filename = f"scene_{timestamp}.e57"
    if output_tag!="":
        e57_scene_filename = f"scene_{output_tag}.e57"
    e57_scene_filepath = os.path.join(out_dir, e57_scene_filename)

    scene_points = np.array(scene_points)
    scene_colors = np.array(scene_colors)

    scene_data = dict()
    scene_data["cartesianX"] = scene_points[:, 0]
    scene_data["cartesianY"] = scene_points[:, 1]
    scene_data["cartesianZ"] = scene_points[:, 2]
    scene_data["colorRed"] = scene_colors[:, 2]
    scene_data["colorGreen"] = scene_colors[:, 1]
    scene_data["colorBlue"] = scene_colors[:, 0]

    with pye57.E57(e57_scene_filepath, mode="w") as e57_write:
        e57_write.write_scan_raw(scene_data)

    logging.info("Scene saved to e57")


def save_mesh(
    out_dir,
    object_points,
    object_colors,
    captured_vis,
    timestamp,
    name,
    voxel_size=None,
):
    """Saves the segmented object as a mesh in an .obj file."""
    try:
        segmented_points_np = np.array(object_points)
        segmented_colors_np = np.array(object_colors)

        if segmented_colors_np.size > 0:
            segmented_colors_rgb = segmented_colors_np[:, ::-1]
        else:
            segmented_colors_rgb = segmented_colors_np
        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points_np)
        segmented_pcd.colors = o3d.utility.Vector3dVector(segmented_colors_rgb / 255.0)

        if voxel_size:
            segmented_pcd = segmented_pcd.voxel_down_sample(voxel_size)

        segmented_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        segmented_pcd.orient_normals_consistent_tangent_plane(k=10)

        dists = segmented_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)
        radii = [avg_dist * 2, avg_dist * 3, avg_dist * 5]

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            segmented_pcd, o3d.utility.DoubleVector(radii)
        )

        pcd_tree = o3d.geometry.KDTreeFlann(segmented_pcd)
        mesh_colors = np.asarray(segmented_pcd.colors)[
            [pcd_tree.search_knn_vector_3d(v, 1)[1][0] for v in mesh.vertices]
        ]
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

        mesh_filename = f"segmented_{name}_{timestamp}.obj"
        mesh_filepath = os.path.join(out_dir, mesh_filename)

        o3d.io.write_triangle_mesh(mesh_filepath, mesh, write_vertex_colors=True)
        logging.info(f"Reconstructed mesh saved to {mesh_filepath}")

        vis_filename = f"segmented_vis_{timestamp}.png"
        vis_filepath = os.path.join(out_dir, vis_filename)
        cv2.imwrite(vis_filepath, captured_vis)
        logging.info(f"Combined visualization saved to {vis_filepath}")

    except Exception as e:
        logging.error(f"An error occurred during mesh reconstruction or saving: {e}")
def save_capture_view(
        captured_vis,
        out_dir,
        output_tag,
        timestamp
    ):
    vis_filename = f"segmented_vis_{timestamp}.png"
    if output_tag != "":
        vis_filename = f"segmented_vis_{output_tag}.png"
    vis_filepath = os.path.join(out_dir, vis_filename)
    cv2.imwrite(vis_filepath, captured_vis)
    logging.info(f"Combined visualization saved to {vis_filepath}")

def save_zed_point_cloud(
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
        return

    scene_points = [[z, -x, -y] for x, y, z in scene_points]
    object_points = [[z, -x, -y] for x, y, z in object_points]

    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_json_object_and_scene(
        args.out_dir,
        args.output_tag,
        object_points,
        object_colors,
        scene_points,
        scene_colors,
        current_time_str,
    )
    save_capture_view(
        captured_vis,
        args.out_dir,
        args.output_tag,
        current_time_str
    )
    if args.save_e57:
        save_e57_object_and_scene(
            args.out_dir,
            args.output_tag,
            object_points,
            object_colors,
            scene_points,
            scene_colors,
            current_time_str,
        )
    if args.exit_after_save:
        print("saved and finish")
        exit(0)
    """
    save_mesh(
        args.out_dir,
        object_points,
        object_colors,
        captured_vis,
        current_time_str,
        "object",
    )
    save_mesh(
        args.out_dir,
        scene_points,
        scene_colors,
        captured_vis,
        current_time_str,
        "scene",
        voxel_size=0.01,
    )
    """
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
    return parser.parse_args()

def main():
    args = parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load Models ----------
    stereo_model = FoundationStereoModel(args)
    sam_predictor = sam_utils.load_sam_model()

    # ---------- ZED Init ----------
    zed = ZedCamera()

    # ---------- Window and Mouse Callback Setup ----------
    win_name = "RGB + Mask | Depth"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_handler.select_box)
    logging.info("Streaming... Draw a box with your mouse.")
    logging.info("Press SPACE to save, 'r' to reset box, ESC to exit.")
    try:
        while True:
            zed_status, left_image, right_image = zed.capture_images()
            if zed_status == sl.ERROR_CODE.SUCCESS:
                # Convert to numpy arrays
                left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
                right_gray = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2GRAY)
                color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
                color_np_org = color_np.copy()

                display_frame = color_np.copy()

                # ---------- Manual Box Selection + SAM2 Logic ----------
                mask = np.zeros_like(color_np[:, :, 0], dtype=bool)

                if mouse_handler.drawing_box:
                    visualization.draw_box(
                        display_frame,
                        mouse_handler.box_start_point,
                        mouse_handler.box_end_point,
                    )

                if mouse_handler.box_defined:
                    box = mouse_handler.get_box()
                    visualization.draw_box(
                        display_frame, (box[0], box[1]), (box[2], box[3])
                    )

                    color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
                    mask = sam_utils.run_sam2(
                        sam_predictor,
                        color_np_rgb,
                        box,
                        iterations=args.erosion_iterations,
                    )
                    display_frame = visualization.overlay_mask_on_frame(
                        display_frame, mask
                    )

                # ---------- FoundationStereo Inference ----------
                

                cv2.imshow(win_name, display_frame)
                key = cv2.waitKey(1)

                if key == ord("r"):
                    logging.info("Box reset. Draw a new one.")
                    mouse_handler.reset_box()

                if key == 32 and mouse_handler.box_defined:
                    depth, (H_scaled, W_scaled) = stereo_model.run_inference(
                        left_gray, right_gray, zed.K_left, zed.baseline
                    )
                    save_zed_point_cloud(
                        stereo_model.args,
                        zed.K_left,
                        depth,
                        color_np_org,
                        mask,
                        display_frame,
                    )

                if key == 27:
                    break
    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
