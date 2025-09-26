import numpy as np
import open3d as o3d
import json
import datetime
import os
import logging
import cv2
import pye57
from .realsense_utils import extrinsics_Rt

def depth2xyzmap(depth, K):
    vy, vx = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
    
    x_map = (vx - K[0, 2]) * depth / K[0, 0]
    y_map = (vy - K[1, 2]) * depth / K[1, 1]
    z_map = depth
    
    xyz_map = np.stack([x_map, y_map, z_map], axis=-1)
    return xyz_map

def save_scene_and_obj(args, K_ir1, baseline, disp, ext_ir1_to_color, K_color, color_np_org, mask, combined_vis):
    logging.info("Saving scene and object...")

    K_scaled_ir1 = K_ir1.copy()
    K_scaled_ir1[:2, :] *= args.scale
    depth = K_scaled_ir1[0, 0] * baseline / (disp + 1e-6)
    
    xyz_map = depth2xyzmap(depth, K_scaled_ir1)
    points_pre_filter = xyz_map.reshape(-1, 3)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_pre_filter)
    pcd = pcd.remove_non_finite_points()

    points_post_filter = np.asarray(pcd.points)

    if not points_post_filter.size:
        logging.warning("No valid points in the disparity map to save.")
        return

    R_ir1_to_color, t_ir1_to_color = extrinsics_Rt(ext_ir1_to_color)
    
    pcd_color_view = o3d.geometry.PointCloud()
    pcd_color_view.points = o3d.utility.Vector3dVector(points_post_filter @ R_ir1_to_color.T + t_ir1_to_color.T)

    points_color_view = np.asarray(pcd_color_view.points)
    projected_points_uv = (K_color @ points_color_view.T).T
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
        logging.warning("The selected mask contains no points from the point cloud. Nothing to save.")
        return
    # Apply transformation for MeshCat coordinate system (Y-up)
    scene_points = [[z, -x, -y] for x, y, z in scene_points];
    object_points = [[z, -x, -y] for x, y, z in object_points];
    
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_json_object_and_scene(args.out_dir, object_points, object_colors, scene_points, scene_colors, current_time_str)
    save_e57_object_and_scene(args.out_dir, object_points, object_colors, scene_points, scene_colors, current_time_str)
    save_obj_mesh(args.out_dir, object_points, object_colors, combined_vis, current_time_str)

def save_json_object_and_scene(out_dir, object_points, object_colors, scene_points, scene_colors, timestamp):
    """Saves the scene and object data to a JSON file."""
    object_colors_arr = np.array(object_colors)
    if object_colors_arr.size > 0:
        # BGR to RGB
        object_colors_arr = object_colors_arr[:, ::-1]

    scene_colors_arr = np.array(scene_colors)
    if scene_colors_arr.size > 0:
        # BGR to RGB
        scene_colors_arr = scene_colors_arr[:, ::-1]

    scene_data = {
        "object_info": {
            "pc": np.array(object_points).tolist(),
            "pc_color": object_colors_arr.tolist()
        },
        "scene_info": {
            "pc_color": [np.array(scene_points).tolist()],
            "img_color": [scene_colors_arr.tolist()]
        },
        "grasp_info": {
            "grasp_poses": [],
            "grasp_conf": []
        }
    }
    
    json_filename = f"scene_{timestamp}.json"
    json_filepath = os.path.join(out_dir, json_filename)
    
    with open(json_filepath, 'w') as f:
        json.dump(scene_data, f, indent=4)
    logging.info(f"Scene saved to {json_filepath}")

def save_e57_object_and_scene(out_dir, object_points, object_colors, scene_points, scene_colors, timestamp):
    """Saves the scene points and colors to a .e57 file."""
    e57_object_filename = f"object_{timestamp}.e57"
    e57_object_filepath = os.path.join(out_dir, e57_object_filename)

    object_points = np.array(object_points)
    object_colors = np.array(object_colors)

    object_data = dict()
    object_data['cartesianX'] = object_points[:, 0]
    object_data['cartesianY'] = object_points[:, 1]
    object_data['cartesianZ'] = object_points[:, 2]
    object_data['colorRed'] = object_colors[:, 2] # Assuming scene_colors is BGR
    object_data['colorGreen'] = object_colors[:, 1]
    object_data['colorBlue'] = object_colors[:, 0]

    with pye57.E57(e57_object_filepath, mode="w") as e57_write:
        e57_write.write_scan_raw(object_data)

    e57_scene_filename = f"scene_{timestamp}.e57"
    e57_scene_filepath = os.path.join(out_dir, e57_scene_filename)

    scene_points = np.array(scene_points)
    scene_colors = np.array(scene_colors)

    scene_data = dict()
    scene_data['cartesianX'] = scene_points[:, 0]
    scene_data['cartesianY'] = scene_points[:, 1]
    scene_data['cartesianZ'] = scene_points[:, 2]
    scene_data['colorRed'] = scene_colors[:, 2] # Assuming scene_colors is BGR
    scene_data['colorGreen'] = scene_colors[:, 1]
    scene_data['colorBlue'] = scene_colors[:, 0]

    with pye57.E57(e57_scene_filepath, mode="w") as e57_write:
        e57_write.write_scan_raw(scene_data)
    
    logging.info(f"Scene saved to e57")

def save_obj_mesh(out_dir, object_points, object_colors, combined_vis, timestamp):
    """Saves the segmented object as a mesh in an .obj file."""
    try:
        segmented_points_np = np.array(object_points)
        segmented_colors_np = np.array(object_colors)

        if segmented_colors_np.size > 0:
            segmented_colors_rgb = segmented_colors_np[:, ::-1]  # BGR to RGB
        else:
            segmented_colors_rgb = segmented_colors_np
        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = o3d.utility.Vector3dVector(segmented_points_np)
        segmented_pcd.colors = o3d.utility.Vector3dVector(segmented_colors_rgb / 255.0)

        segmented_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        segmented_pcd.orient_normals_consistent_tangent_plane(k=10)
        
        dists = segmented_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)
        radii = [avg_dist*2, avg_dist*3, avg_dist*5]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            segmented_pcd,
            o3d.utility.DoubleVector(radii)
        )
        
        pcd_tree = o3d.geometry.KDTreeFlann(segmented_pcd)
        mesh_colors = np.asarray(segmented_pcd.colors)[ \
            [pcd_tree.search_knn_vector_3d(v, 1)[1][0] for v in mesh.vertices]
        ]
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        
        mesh_filename = f"segmented_mesh_{timestamp}.obj"
        mesh_filepath = os.path.join(out_dir, mesh_filename)
        
        o3d.io.write_triangle_mesh(mesh_filepath, mesh, write_vertex_colors=True)
        logging.info(f"Reconstructed mesh saved to {mesh_filepath}")
        
        vis_filename = f"segmented_vis_{timestamp}.png"
        vis_filepath = os.path.join(out_dir, vis_filename)
        cv2.imwrite(vis_filepath, combined_vis)
        logging.info(f"Combined visualization saved to {vis_filepath}")
        
    except Exception as e:
        logging.error(f"An error occurred during mesh reconstruction or saving: {e}")