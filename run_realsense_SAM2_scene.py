import os, sys, argparse, logging, datetime, json
import numpy as np
import cv2, torch, open3d as o3d
import pyrealsense2 as rs
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from FoundationStereo.core.utils.utils import InputPadder
from FoundationStereo.Utils import *
from FoundationStereo.core.foundation_stereo import *

# ------------------- Config -------------------
SAM2_CHECKPOINT = "/home/j300/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAY_COLOR_CV = (0, 200, 0) # light yellow in BGR
OVERLAY_ALPHA = 0.6 # Transparency of the mask
BOX_COLOR = (0, 255, 0) # Green for the drawing box

# ------------------- Mouse Interaction State -------------------
drawing_box = False
box_start_point = (-1, -1)
box_end_point = (-1, -1)
box_defined = False

def select_box(event, x, y, flags, param):
    """Mouse callback function to select a bounding box."""
    global drawing_box, box_start_point, box_end_point, box_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_box = True
        box_defined = False
        box_start_point = (x, y)
        box_end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_box:
            box_end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        # Ensure the box has a non-zero area by checking width and height
        if abs(box_start_point[0] - x) > 1 and abs(box_start_point[1] - y) > 1:
            box_end_point = (x, y)
            box_defined = True
        else:
            # If the box is too small, reset it
            box_start_point = (-1, -1)
            box_end_point = (-1, -1)


# ------------------- Functions -------------------
def run_sam2(predictor, box, iterations=6):
    x1, y1, x2, y2 = box
    input_point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
    input_label = np.array([1])
    with torch.inference_mode():
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
    mask_uint8 = (masks[0] * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask_uint8, kernel, iterations=iterations)
    return eroded_mask.astype(bool)

def overlay_mask_on_frame(frame, mask):
    overlay = frame.copy()
    overlay[mask] = OVERLAY_COLOR_CV
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
    return frame

def intrinsics_from_profile(stream_profile: rs.video_stream_profile):
    intr = stream_profile.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0,      0,      1]], dtype=np.float32)
    return intr, K

def extrinsics_Rt(extrin: rs.extrinsics):
    R = np.array(extrin.rotation, dtype=np.float32).reshape(3, 3)
    t = np.array(extrin.translation, dtype=np.float32).reshape(3, )
    return R, t

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/pretrained_models/23-51-11/model_best_bp2.pth',
                        type=str, help='pretrained model path')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--out_dir', default='./output/', type=str, help='the directory to save results')
    args = parser.parse_args()
    
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ---------- Load Models ----------
    logging.info("Loading models...")
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda().eval()

    try:
        sam_model = build_sam2(SAM2_CFG, SAM2_CHECKPOINT).to(DEVICE)
        sam_predictor = SAM2ImagePredictor(sam_model)
    except Exception as e:
        logging.error(f"Error loading SAM2 model: {e}")
        logging.error(f"Please check your checkpoint ('{SAM2_CHECKPOINT}') and config ('{SAM2_CFG}') paths.")
        sys.exit(1)

    # ---------- RealSense Init ----------
    pipeline = rs.pipeline()
    config = rs.config()
    W, H, FPS = 848, 480, 15
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    
    align_to = rs.stream.infrared
    align = rs.align(align_to)

    ir1_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 1))
    ir2_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 2))
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    
    _, K_ir1 = intrinsics_from_profile(ir1_profile)
    _, K_color = intrinsics_from_profile(color_profile)
    ext_ir1_to_ir2 = ir1_profile.get_extrinsics_to(ir2_profile)
    ext_ir1_to_color = ir1_profile.get_extrinsics_to(color_profile)
    
    _, t_ir1_to_ir2 = extrinsics_Rt(ext_ir1_to_ir2)
    baseline = np.linalg.norm(t_ir1_to_ir2)
    K_ir1 = K_ir1.astype(np.float32)

    # ---------- Window and Mouse Callback Setup ----------
    win_name = "RGB + Mask | Disparity"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_box)
    logging.info("Streaming... Draw a box with your mouse.")
    logging.info("Press SPACE to save, 'r' to reset box, ESC to exit.")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            ir1_frame = aligned_frames.get_infrared_frame(1)
            ir2_frame = aligned_frames.get_infrared_frame(2)
            color_frame = aligned_frames.get_color_frame()
    
            if not ir1_frame or not ir2_frame or not color_frame:
                continue
    
            ir1_np = np.asanyarray(ir1_frame.get_data())
            ir2_np = np.asanyarray(ir2_frame.get_data())
            color_np = np.asanyarray(color_frame.get_data())
            color_np_org = color_np.copy()
            
            # Create a copy for display to draw on
            display_frame = color_np.copy()

            # ---------- Manual Box Selection + SAM2 Logic ----------
            mask = np.zeros_like(color_np[:,:,0], dtype=bool)
            
            if drawing_box:
                cv2.rectangle(display_frame, box_start_point, box_end_point, BOX_COLOR, 2)
            
            if box_defined:
                # Draw the final box
                x1 = min(box_start_point[0], box_end_point[0])
                y1 = min(box_start_point[1], box_end_point[1])
                x2 = max(box_start_point[0], box_end_point[0])
                y2 = max(box_start_point[1], box_end_point[1])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                
                # Run SAM2 on the defined box
                box = [x1, y1, x2, y2]
                color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
                sam_predictor.set_image(color_np_rgb)
                mask = run_sam2(sam_predictor, box)
                display_frame = overlay_mask_on_frame(display_frame, mask)

            # ---------- FoundationStereo Inference ----------
            ir1_np_bgr = cv2.cvtColor(ir1_np, cv2.COLOR_GRAY2BGR)
            ir2_np_bgr = cv2.cvtColor(ir2_np, cv2.COLOR_GRAY2BGR)
            img0 = cv2.resize(ir1_np_bgr, fx=args.scale, fy=args.scale, dsize=None)
            img1 = cv2.resize(ir2_np_bgr, fx=args.scale, fy=args.scale, dsize=None)
            H_scaled, W_scaled = img0.shape[:2]
    
            img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
            img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
            padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
            img0_t, img1_t = padder.pad(img0_t, img1_t)
    
            with torch.cuda.amp.autocast(True):
                disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
    
            disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H_scaled, W_scaled)
            
            xx, yy = np.meshgrid(np.arange(W_scaled), np.arange(H_scaled), indexing='xy')
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf
            
            vis_disp = vis_disparity(disp)
            vis_disp_resized = cv2.resize(vis_disp, fx=1/args.scale, fy=1/args.scale, dsize=None)
            
            combined_vis = np.concatenate([display_frame, vis_disp_resized], axis=1)
            cv2.imshow(win_name, combined_vis)
            key = cv2.waitKey(1)
            
            if key == ord('r'):
                logging.info("Box reset. Draw a new one.")
                box_defined = False
                drawing_box = False
                box_start_point = (-1, -1)
                box_end_point = (-1, -1)

            # ---------- Save Scene to JSON on Spacebar Press ----------
            if key == 32 and box_defined:
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
                    continue

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
                        
                        scene_points.append(point)
                        scene_colors.append(color)
                        
                        if mask[v, u]:
                            object_points.append(point)
                            object_colors.append(color)

                if not object_points:
                    logging.warning("The selected mask contains no points from the point cloud. Nothing to save.")
                    continue

                # --- Save Scene JSON ---
                scene_data = {
                    "object_info": {
                        "pc": np.array(object_points).tolist(),
                        "pc_color": np.array(object_colors).tolist()
                    },
                    "scene_info": {
                        "pc_color": [np.array(scene_points).tolist()],
                        "img_color": [np.array(scene_colors).tolist()]
                    },
                    "grasp_info": {
                        "grasp_poses": [],
                        "grasp_conf": []
                    }
                }
                
                current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"scene_{current_time_str}.json"
                json_filepath = os.path.join(args.out_dir, json_filename)
                
                with open(json_filepath, 'w') as f:
                    json.dump(scene_data, f, indent=4)
                logging.info(f"Scene saved to {json_filepath}")

                # --- Save Segmented Object Mesh ---
                try:
                    segmented_points_np = np.array(object_points)
                    segmented_colors_np = np.array(object_colors)

                    segmented_colors_rgb = cv2.cvtColor(segmented_colors_np.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
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
                    
                    mesh_filename = f"segmented_mesh_{current_time_str}.obj"
                    mesh_filepath = os.path.join(args.out_dir, mesh_filename)
                    
                    o3d.io.write_triangle_mesh(mesh_filepath, mesh, write_vertex_colors=True)
                    logging.info(f"Reconstructed mesh saved to {mesh_filepath}")
                    
                    vis_filename = f"segmented_vis_{current_time_str}.png"
                    vis_filepath = os.path.join(args.out_dir, vis_filename)
                    cv2.imwrite(vis_filepath, combined_vis)
                    logging.info(f"Combined visualization saved to {vis_filepath}")
                    
                except Exception as e:
                    logging.error(f"An error occurred during mesh reconstruction or saving: {e}")

            if key == 27:
                break
    
    finally:
                pipeline.stop()
                cv2.destroyAllWindows()