import sys
import torch
import cv2
import logging
import open3d as o3d
import numpy as np
import pyzed.sl as sl

from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal_with_color
from PointCloud_Generation.mouse_handlerv2 import MouseHandler
from PointCloud_Generation.grounding_dino_utils import GroundindDinoPredictor
from PointCloud_Generation.visualization import visualize_named_box, visualize_mask

# Add the project root to sys.path to enable relative imports when run as a script
# current_file_dir = os.path.dirname(os.path.abspath(__file__))
# project_root_dir = os.path.dirname(current_file_dir)
# if project_root_dir not in sys.path:
#     sys.path.insert(0, project_root_dir)
# THIRD_PARTY_DIR = os.path.expanduser("~/Third_Party")
# if THIRD_PARTY_DIR not in sys.path:
#     sys.path.insert(0, THIRD_PARTY_DIR)
from PointCloud_Generation import mouse_handler  # noqa: E402
from PointCloud_Generation import visualization  # noqa: E402
from PointCloud_Generation import sam_utils  # noqa: E402
from PointCloud_Generation.stereo_utils import FoundationStereoModel  # noqa: E402
from PointCloud_Generation.zed_utils import ZedCamera  # noqa: E402
# from PointCloud_Generation.yolo_inference import YOLOv5Detector  # noqa: E402

logger = logging.getLogger(__name__)


class NamedMask:
    def __init__(self, name, mask):
        self.name = name
        self.mask = mask


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


def generate_pointcloud(depth, color_np_org, mask, K_cam, scale, max_depth):
    logging.info("generating scene and object...")

    K_scaled_cam = K_cam.copy()
    K_scaled_cam[:2, :] *= scale

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

    valid_depth_mask = points_post_filter[:, 2] < max_depth
    points_post_filter = points_post_filter[valid_depth_mask]
    if not points_post_filter.size:
        logging.warning(
            f"No points remaining after filtering with max_depth={max_depth}"
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
        raise ValueError("The selected mask contains no points from the point cloud.")
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


def generate_pointcloud_multiple_obj(
    depth, color_np_org, masks, K_cam, scale, max_depth
):
    logging.info("generating scene and object...")

    K_scaled_cam = K_cam.copy()
    K_scaled_cam[:2, :] *= scale

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

    valid_depth_mask = points_post_filter[:, 2] < max_depth
    points_post_filter = points_post_filter[valid_depth_mask]
    if not points_post_filter.size:
        logging.warning(
            f"No points remaining after filtering with max_depth={max_depth}"
        )
        return

    # Since the depth map is already in the color camera's frame, no transformation is needed.
    # We just need to map points to pixels to get their color and check the mask.
    projected_points_uv = (K_cam @ points_post_filter.T).T
    projected_points_uv[:, :2] /= projected_points_uv[:, 2:]

    scene_points = []
    scene_colors = []
    objects_points = []
    objects_colors = []
    for _ in masks:
        objects_points.append([])
        objects_colors.append([])
    # num_objects = len(masks)

    H_color, W_color = color_np_org.shape[:2]

    for i in range(len(projected_points_uv)):
        u, v = int(projected_points_uv[i, 0]), int(projected_points_uv[i, 1])
        if 0 <= u < W_color and 0 <= v < H_color:
            point = points_post_filter[i]
            color = color_np_org[v, u]

            for i, mask in enumerate(masks):
                if mask[v, u]:
                    objects_points[i].append(point)
                    objects_colors[i].append(color)
                else:
                    scene_points.append(point)
                    scene_colors.append(color)
    # scene points
    scene_points = [[z, -x, -y] for x, y, z in scene_points]
    scene_colors_arr = np.array(scene_colors)
    if scene_colors_arr.size > 0:
        scene_colors_arr = scene_colors_arr[:, ::-1]

    scene_data = {
        "objects_info": [],
        "scene_info": {
            "pc_color": [np.array(scene_points)],
            "img_color": [scene_colors_arr],
        },
        "grasp_info": {"grasp_poses": [], "grasp_conf": []},
    }

    # objects points
    for object_points, object_colors in zip(
        objects_points, objects_colors, strict=False
    ):
        if not object_points:
            logging.warning(
                "The selected mask contains no points from the point cloud. Nothing to save."
            )
            raise ValueError(
                "The selected mask contains no points from the point cloud."
            )
        object_points = [[z, -x, -y] for x, y, z in object_points]

        """Saves the scene and object data to a JSON file."""
        object_colors_arr = np.array(object_colors)
        if object_colors_arr.size > 0:
            object_colors_arr = object_colors_arr[:, ::-1]
        scene_data["objects_info"].append(
            {"pc": np.array(object_points), "pc_color": object_colors_arr}
        )

    return scene_data


def generate_pointcloud_multiple_obj_with_name(
    depth, color_np_org, named_masks: list[NamedMask], K_cam, scale, max_depth
):
    logging.info("generating scene and object...")

    K_scaled_cam = K_cam.copy()
    K_scaled_cam[:2, :] *= scale

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

    valid_depth_mask = points_post_filter[:, 2] < max_depth
    points_post_filter = points_post_filter[valid_depth_mask]
    if not points_post_filter.size:
        logging.warning(
            f"No points remaining after filtering with max_depth={max_depth}"
        )
        return

    # Since the depth map is already in the color camera's frame, no transformation is needed.
    # We just need to map points to pixels to get their color and check the mask.
    projected_points_uv = (K_cam @ points_post_filter.T).T
    projected_points_uv[:, :2] /= projected_points_uv[:, 2:]

    # objects_name = [named_mask.name for named_mask in named_masks]
    objects_points = [[] for _ in named_masks]
    objects_colors = [[] for _ in named_masks]
    scene_points = []
    scene_colors = []

    H_color, W_color = color_np_org.shape[:2]

    for i in range(len(projected_points_uv)):
        u, v = int(projected_points_uv[i, 0]), int(projected_points_uv[i, 1])
        if 0 <= u < W_color and 0 <= v < H_color:
            point = points_post_filter[i]
            color = color_np_org[v, u]

            for object_points, object_colors, named_mask in zip(
                objects_points, objects_colors, named_masks, strict=False
            ):
                if named_mask.mask[v, u]:
                    object_points.append(point)
                    object_colors.append(color)
                    break
            else:
                scene_points.append(point)
                scene_colors.append(color)
    # here
    # objects points parse
    for i in range(len(objects_points)):
        if not objects_points[i]:
            logging.error(
                "The selected mask contains no points from the point cloud. Nothing to save."
            )
            raise ValueError(
                "The selected mask contains no points from the point cloud."
            )

        objects_points[i] = np.array([[z, -x, -y] for x, y, z in objects_points[i]])
        objects_colors[i] = np.array(objects_colors[i])
        if objects_colors[i].size > 0:
            objects_colors[i] = objects_colors[i][:, ::-1]

    # scene points
    scene_points = np.array([[z, -x, -y] for x, y, z in scene_points])
    scene_colors = np.array(object_colors)
    if scene_colors.size > 0:
        scene_colors = scene_colors[:, ::-1]

    # Final construct
    scene_data = {
        "object_infos": [
            {"name": named_mask.name, "points": object_points, "colors": object_colors}
            for object_points, object_colors, named_mask in zip(
                objects_points, objects_colors, named_masks, strict=False
            )
        ],
        "scene_info": {
            "pc_color": [scene_points],
            "img_color": [scene_colors],
        },
        "grasp_info": {"grasp_poses": [], "grasp_conf": []},
    }
    return scene_data


def generate_pointcloud_multiple_obj_with_name_dict(
    depth, color_np_org, named_masks: list[NamedMask], K_cam, scale, max_depth
):
    logging.info("generating scene and object...")

    K_scaled_cam = K_cam.copy()
    K_scaled_cam[:2, :] *= scale

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

    valid_depth_mask = points_post_filter[:, 2] < max_depth
    points_post_filter = points_post_filter[valid_depth_mask]
    if not points_post_filter.size:
        logging.warning(
            f"No points remaining after filtering with max_depth={max_depth}"
        )
        return

    # Since the depth map is already in the color camera's frame, no transformation is needed.
    # We just need to map points to pixels to get their color and check the mask.
    projected_points_uv = (K_cam @ points_post_filter.T).T
    projected_points_uv[:, :2] /= projected_points_uv[:, 2:]

    # objects_name = [named_mask.name for named_mask in named_masks]
    objects_points = [[] for _ in named_masks]
    objects_colors = [[] for _ in named_masks]
    scene_points = []
    scene_colors = []

    H_color, W_color = color_np_org.shape[:2]

    for i in range(len(projected_points_uv)):
        u, v = int(projected_points_uv[i, 0]), int(projected_points_uv[i, 1])
        if 0 <= u < W_color and 0 <= v < H_color:
            point = points_post_filter[i]
            color = color_np_org[v, u]

            for object_points, object_colors, named_mask in zip(
                objects_points, objects_colors, named_masks, strict=False
            ):
                if named_mask.mask[v, u]:
                    object_points.append(point)
                    object_colors.append(color)
                    break
            # add the point into scene no matter what
            scene_points.append(point)
            scene_colors.append(color)
    # here
    # objects points parse
    for i in range(len(objects_points)):
        if not objects_points[i]:
            logging.error(
                "The selected mask contains no points from the point cloud. Nothing to save."
            )
            raise ValueError(
                "The selected mask contains no points from the point cloud."
            )

        objects_points[i] = np.array([[z, -x, -y] for x, y, z in objects_points[i]])
        objects_colors[i] = np.array(objects_colors[i])
        if objects_colors[i].size > 0:
            objects_colors[i] = objects_colors[i][:, ::-1]

        ## remove outliers
        objects_points[i], _, objects_colors[i], _ = (
            point_cloud_outlier_removal_with_color(
                torch.from_numpy(objects_points[i]),
                torch.from_numpy(objects_colors[i].copy()),
            )
        )
        objects_points[i] = objects_points[i].cpu().numpy()
        objects_colors[i] = objects_colors[i].cpu().numpy()

    # scene points
    scene_points = np.array([[z, -x, -y] for x, y, z in scene_points])
    scene_colors = np.array(scene_colors)
    if scene_colors.size > 0:
        scene_colors = scene_colors[:, ::-1]

    # Final construct
    scene_data = {
        "object_infos": {
            named_mask.name: {"points": object_points, "colors": object_colors}
            for object_points, object_colors, named_mask in zip(
                objects_points, objects_colors, named_masks, strict=False
            )
        },
        "scene_info": {
            "pc_color": [scene_points],
            "img_color": [scene_colors],
        },
        "grasp_info": {"grasp_poses": [], "grasp_conf": []},
    }
    return scene_data


class PointCloudGenerator:
    def __init__(self, args):
        # Init
        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        # Init configs
        self.erosion_iterations = args.erosion_iterations
        self.max_depth = args.max_depth
        self.scale = args.scale
        # This part could take a while, to load all these three models
        # self.yolo_detector = YOLOv5Detector(model_path=config.YOLO_CHECKPOINT, conf=0.4)
        self.sam_predictor = sam_utils.load_sam_model()
        self.stereo_model = FoundationStereoModel(args)
        self.groundingdino_predictor = GroundindDinoPredictor()
        self.zed = ZedCamera(args.use_png)

    def generate_pointcloud(self, target_names: list[str], need_confirm=True):
        # Target objects
        prompt = ""
        target_boxes = dict()
        for target in target_names:
            prompt += target + " ."
            target_boxes[target] = None

        # Capture image
        try:
            zed_status, left_image, right_image = self.zed.capture_images()
        except Exception as e:
            logger.exception(f"error{e}")
            return RuntimeError

        color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
        color_np_org = color_np.copy()
        display_frame = color_np.copy()
        # GroundingDINO detection
        boxes = self.groundingdino_predictor.predict_boxes(color_np_org, prompt)
        for box in boxes:
            if box.phrase not in target_boxes:
                logger.error(
                    f"Unknown Object {box.phrase} generated by GroundingDINO, Failed"
                )
                raise ValueError
            if target_boxes[box.phrase] is not None:
                logger.error(f"More than one {box.phrase} detected")
                raise ValueError
            target_boxes[box.phrase] = box
            print(box.box, box.logits, box.phrase)
        for target_box in target_boxes:
            if target_boxes[target_box] is None:
                logger.error(f"Object {target_box} is not detected")
                raise ValueError
        boxes = [target_boxes[target_name] for target_name in target_names]

        # SAM2 inference
        named_masks = []
        for box in boxes:
            color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
            mask = sam_utils.run_sam2(
                self.sam_predictor,
                color_np_rgb,
                box.box,
                iterations=self.erosion_iterations,
            )
            named_masks.append(NamedMask(name=box.phrase, mask=mask))

        # stereo inference
        left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
        right_gray = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2GRAY)
        depth, (H_scaled, W_scaled) = self.stereo_model.run_inference(
            left_gray, right_gray, self.zed.K_left, self.zed.baseline
        )

        # GroundingDINO detection

        """
        Maybe, maybe, I shouldn't request GroundingDINO to detect all kinds of things at once.
        Maybe we can try to detect things one by one.
        But then it won't be able to know if that things has been detected and recorded before...
        """

        if need_confirm:
            win_name = "RGB + Mask | Depth"
            cv2.namedWindow(win_name)
            for box in boxes:
                display_frame = visualize_named_box(display_frame, box)
            for named_mask in named_masks:
                display_frame = visualize_mask(display_frame, named_mask.mask)
            cv2.imshow(win_name, display_frame)
            while True:
                logger.info("if satisfied, press space and continue.")
                key = cv2.waitKey(0)
                if key == 32:
                    break
                elif key == 27:
                    raise ValueError("Not satisfied with the generated result.")

        # gen pointcloud
        result_scene_data = generate_pointcloud_multiple_obj_with_name_dict(
            depth,
            color_np_org,
            named_masks,
            self.zed.K_left,
            self.scale,
            self.max_depth,
        )
        return result_scene_data

    def interactive_gui_mode(self, save_json=False):
        # ---------- Window and Mouse Callback Setup ----------
        win_name = "RGB + Mask | Depth"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, mouse_handler.select_box)
        logging.info("Streaming... Draw a box with your mouse.")
        logging.info("Press SPACE to save, 'r' to reset box, ESC to exit.")
        try:
            while True:
                zed_status, left_image, right_image = self.zed.capture_images()
                if zed_status == sl.ERROR_CODE.SUCCESS:
                    # Convert to numpy arrays
                    left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
                    right_gray = cv2.cvtColor(
                        right_image.get_data(), cv2.COLOR_BGRA2GRAY
                    )
                    color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
                    color_np_org = color_np.copy()

                    display_frame = color_np.copy()

                    # ---------- Manual Box Selection + SAM2 Logic ----------
                    mask = np.zeros_like(color_np[:, :, 0], dtype=bool)

                    key = cv2.waitKey(1)
                    if key == ord("y"):
                        df = self.yolo_detector.infer(display_frame)
                        if "name" in df.columns:
                            cup_detections = df[df["name"] == "cup"]
                            if not cup_detections.empty:
                                logging.info("Cup detected, selecting it.")
                                cup_box = cup_detections.iloc[0]
                                box = (
                                    int(cup_box["xmin"]),
                                    int(cup_box["ymin"]),
                                    int(cup_box["xmax"]),
                                    int(cup_box["ymax"]),
                                )
                                mouse_handler.box_start_point = (box[0], box[1])
                                mouse_handler.box_end_point = (box[2], box[3])
                                mouse_handler.box_defined = True
                                mouse_handler.drawing_box = False

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
                            self.sam_predictor,
                            color_np_rgb,
                            box,
                            iterations=self.erosion_iterations,
                        )
                        display_frame = visualization.overlay_mask_on_frame(
                            display_frame, mask
                        )

                    # ---------- FoundationStereo Inference ----------

                    cv2.imshow(win_name, display_frame)

                    if key == ord("r"):
                        logging.info("Box reset. Draw a new one.")
                        mouse_handler.reset_box()

                    if key == 32 and mouse_handler.box_defined:
                        depth, (H_scaled, W_scaled) = self.stereo_model.run_inference(
                            left_gray, right_gray, self.zed.K_left, self.zed.baseline
                        )
                        result = generate_pointcloud(
                            depth,
                            color_np_org,
                            mask,
                            self.zed.K_left,
                            self.scale,
                            self.max_depth,
                        )
                        return result

                    if key == 27:
                        return None
        except KeyboardInterrupt:
            self.close()
            sys.exit(0)
        except Exception as e:
            logging.error(
                f"An error occurred during mesh reconstruction or saving: {e}"
            )
            return None

    def interactive_gui_mode_multiple(self, save_json=False):
        # ---------- Window and Mouse Callback Setup ----------
        win_name = "RGB + Mask | Depth"
        cv2.namedWindow(win_name)
        mousehandler = MouseHandler()
        cv2.setMouseCallback(win_name, mousehandler.select_box)
        logging.info("Streaming... Draw a box with your mouse.")
        logging.info("Press SPACE to save, 'r' to reset box, ESC to exit.")
        try:
            while True:
                zed_status, left_image, right_image = self.zed.capture_images()
                if zed_status == sl.ERROR_CODE.SUCCESS:
                    # Convert to numpy arrays
                    left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
                    right_gray = cv2.cvtColor(
                        right_image.get_data(), cv2.COLOR_BGRA2GRAY
                    )
                    color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
                    color_np_org = color_np.copy()

                    display_frame = color_np.copy()

                    # ---------- Manual Box Selection + SAM2 Logic ----------
                    mask = np.zeros_like(color_np[:, :, 0], dtype=bool)

                    key = cv2.waitKey(1)
                    if key == ord("y"):
                        df = self.yolo_detector.infer(display_frame)
                        if "name" in df.columns:
                            cup_detections = df[df["name"] == "cup"]
                            if cup_detections.empty:
                                logging.warning("No Cup Detected!")
                            for i in range(len(cup_detections)):
                                cup_box = cup_detections.iloc[i]
                                box = (
                                    int(cup_box["xmin"]),
                                    int(cup_box["ymin"]),
                                    int(cup_box["xmax"]),
                                    int(cup_box["ymax"]),
                                )
                                mousehandler.box_start_points.append((box[0], box[1]))
                                mousehandler.box_end_points.append((box[2], box[3]))
                                mousehandler.num_boxes += 1
                                mousehandler.drawing_box = False

                    # if mousehandler.drawing_box:
                    #    for i in range (mousehandler.num_boxes):
                    #        visualization.draw_box(
                    #            display_frame,
                    #            mousehandler.box_start_points[i],
                    #            mousehandler.box_end_points[i],
                    #        )
                    masks = []
                    if mousehandler.num_boxes > 0:
                        color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
                        boxes = mousehandler.get_boxes()
                        for box in boxes:
                            visualization.draw_box(
                                display_frame, (box[0], box[1]), (box[2], box[3])
                            )
                            mask = sam_utils.run_sam2(
                                self.sam_predictor,
                                color_np_rgb,
                                box,
                                iterations=self.erosion_iterations,
                            )
                            display_frame = visualization.overlay_mask_on_frame(
                                display_frame, mask
                            )
                            masks.append(mask)

                    # ---------- FoundationStereo Inference ----------

                    cv2.imshow(win_name, display_frame)

                    if key == ord("r"):
                        logging.info("Box reset. Draw a new one.")
                        mousehandler.reset()

                    if key == 32 and mousehandler.num_boxes > 0:
                        depth, (H_scaled, W_scaled) = self.stereo_model.run_inference(
                            left_gray, right_gray, self.zed.K_left, self.zed.baseline
                        )
                        result = generate_pointcloud_multiple_obj(
                            depth,
                            color_np_org,
                            masks,
                            self.zed.K_left,
                            self.scale,
                            self.max_depth,
                        )
                        return result

                    if key == 27:
                        return None
        except KeyboardInterrupt:
            self.close()
            sys.exit(0)
        except Exception as e:
            logging.error(
                f"An error occurred during mesh reconstruction or saving: {e}"
            )
            return None

    def interactive_gui_mode_multiple_grounding(self, save_json=False):
        # ---------- Window and Mouse Callback Setup ----------
        win_name = "RGB + Mask | Depth"
        cv2.namedWindow(win_name)
        mousehandler = MouseHandler()
        cv2.setMouseCallback(win_name, mousehandler.select_box)
        logging.info("Streaming... Draw a box with your mouse.")
        logging.info("Press SPACE to save, 'r' to reset box, ESC to exit.")
        try:
            while True:
                zed_status, left_image, right_image = self.zed.capture_images()
                if zed_status == sl.ERROR_CODE.SUCCESS:
                    # Convert to numpy arrays
                    left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
                    right_gray = cv2.cvtColor(
                        right_image.get_data(), cv2.COLOR_BGRA2GRAY
                    )
                    color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
                    color_np_org = color_np.copy()

                    display_frame = color_np.copy()

                    # -- grounding dino test--
                    boxes = self.groundingdino_predictor.predict_boxes(
                        color_np_org, "purple cup . purple cube ."
                    )
                    for box in boxes:
                        print(box.box, box.logits, box.phrase)
                    # ---------- Manual Box Selection + SAM2 Logic ----------
                    mask = np.zeros_like(color_np[:, :, 0], dtype=bool)

                    key = cv2.waitKey(1)
                    if key == ord("y"):
                        df = self.yolo_detector.infer(display_frame)
                        if "name" in df.columns:
                            cup_detections = df[df["name"] == "cup"]
                            if cup_detections.empty:
                                logging.warning("No Cup Detected!")
                            for i in range(len(cup_detections)):
                                cup_box = cup_detections.iloc[i]
                                box = (
                                    int(cup_box["xmin"]),
                                    int(cup_box["ymin"]),
                                    int(cup_box["xmax"]),
                                    int(cup_box["ymax"]),
                                )
                                mousehandler.box_start_points.append((box[0], box[1]))
                                mousehandler.box_end_points.append((box[2], box[3]))
                                mousehandler.num_boxes += 1
                                mousehandler.drawing_box = False

                    # if mousehandler.drawing_box:
                    #    for i in range (mousehandler.num_boxes):
                    #        visualization.draw_box(
                    #            display_frame,
                    #            mousehandler.box_start_points[i],
                    #            mousehandler.box_end_points[i],
                    #        )
                    masks = []
                    if mousehandler.num_boxes > 0:
                        color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
                        boxes = mousehandler.get_boxes()
                        for box in boxes:
                            visualization.draw_box(
                                display_frame, (box[0], box[1]), (box[2], box[3])
                            )
                            mask = sam_utils.run_sam2(
                                self.sam_predictor,
                                color_np_rgb,
                                box,
                                iterations=self.erosion_iterations,
                            )
                            display_frame = visualization.overlay_mask_on_frame(
                                display_frame, mask
                            )
                            masks.append(mask)

                    # ---------- FoundationStereo Inference ----------

                    cv2.imshow(win_name, display_frame)

                    if key == ord("r"):
                        logging.info("Box reset. Draw a new one.")
                        mousehandler.reset()

                    if key == 32 and mousehandler.num_boxes > 0:
                        depth, (H_scaled, W_scaled) = self.stereo_model.run_inference(
                            left_gray, right_gray, self.zed.K_left, self.zed.baseline
                        )
                        result = generate_pointcloud_multiple_obj(
                            depth,
                            color_np_org,
                            masks,
                            self.zed.K_left,
                            self.scale,
                            self.max_depth,
                        )
                        return result

                    if key == 27:
                        return None
        except KeyboardInterrupt:
            self.close()
            sys.exit(0)
        except Exception as e:
            logging.error(
                f"An error occurred during mesh reconstruction or saving: {e}"
            )
            return None

    def interactive_grounding_test(self, target_names: list[str]):
        prompt = ""
        target_boxes = dict()

        for target in target_names:
            prompt += target + " ."
            target_boxes[target] = None

        # cv2 init
        win_name = "image with mask"
        cv2.namedWindow(win_name)
        mousehandler = MouseHandler()
        cv2.setMouseCallback(win_name, mousehandler.select_box)
        logger.info("Streaming... Draw a box with your mouse.")

        method = "all_in_one"
        box_threshold = 0.4
        text_threshold = 0.4
        while True:
            try:
                zed_status, left_image, right_image = self.zed.capture_images()
            except Exception as e:
                logger.error("Something went wrong when capturing the image.")
                raise e
            if zed_status != sl.ERROR_CODE.SUCCESS:
                continue  # try again
            # image captured

            # Convert to numpy arrays
            # left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
            # right_gray = cv2.cvtColor(
            #    right_image.get_data(), cv2.COLOR_BGRA2GRAY
            # )
            color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
            color_np_org = color_np.copy()

            display_frame = color_np.copy()

            # -- grounding dino reference--
            if method == "all_in_one":
                boxes = self.groundingdino_predictor.predict_boxes(
                    color_np_org,
                    prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            elif method == "one_by_one":
                boxes = []
                for name in target_names:
                    boxes.extend(
                        self.groundingdino_predictor.predict_boxes(
                            color_np_org,
                            name + " .",
                            box_threshold=box_threshold,
                            text_threshold=text_threshold,
                        )
                    )
            else:
                logger.critical("unknown method")
                raise ValueError

            # annotate
            for box in boxes:
                start_point = (int(box.box[0]), int(box.box[1]))
                end_point = (int(box.box[2]), int(box.box[3]))
                cv2.rectangle(display_frame, start_point, end_point, (0, 255, 0), 2)
                label = f"{box.phrase}: {box.logits:.2f}"
                cv2.putText(
                    display_frame,
                    label,
                    (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # show image
            cv2.imshow(win_name, display_frame)
            logger.info(
                f"box threshold = {box_threshold:.2f}, text threshold = {text_threshold:.2f}, method = {method}"
            )
            # block
            key = cv2.waitKey(0)
            if key == ord("u"):
                box_threshold -= 0.05
            elif key == ord("i"):
                box_threshold += 0.05
            elif key == ord("o"):
                text_threshold -= 0.05
            elif key == ord("p"):
                text_threshold += 0.05
            elif key == ord("a"):
                method = "all_in_one"
            elif key == ord("s"):
                method = "one_by_one"
            elif key == 27:
                return None
            # anykey except for ese will let it do it again.

    def silent_mode_multiple_grounding(self, target_names: list[str]):
        # Target objects
        prompt = ""
        target_boxes = dict()

        for target in target_names:
            prompt += target + " ."
            target_boxes[target] = None
        # Capture image
        try:
            zed_status, left_image, right_image = self.zed.capture_images()
        except Exception as e:
            logger.exception(f"error{e}")
            return RuntimeError

        color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
        color_np_org = color_np.copy()

        # stereo inference
        left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
        right_gray = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2GRAY)
        depth, (H_scaled, W_scaled) = self.stereo_model.run_inference(
            left_gray, right_gray, self.zed.K_left, self.zed.baseline
        )

        named_masks = []
        # GroundingDINO detection

        """
        Maybe, maybe, I shouldn't request GroundingDINO to detect all kinds of things at once.
        Maybe we can try to detect things one by one.
        But then it won't be able to know if that things has been detected and recorded before...
        """

        boxes = self.groundingdino_predictor.predict_boxes(color_np_org, prompt)
        for box in boxes:
            if box.phrase not in target_boxes:
                logger.error(
                    f"Unknown Object {box.phrase} generated by GroundingDINO, Failed"
                )
                raise ValueError
            if target_boxes[box.phrase] is not None:
                logger.error(f"More than one {box.phrase} detected")
                raise ValueError
            target_boxes[box.phrase] = box
            print(box.box, box.logits, box.phrase)
        for target_box in target_boxes:
            if target_boxes[target_box] is None:
                logger.error(f"Object {target_box} is not detected")
                raise ValueError
        boxes = [target_boxes[target_name] for target_name in target_names]
        # SAM2 inference
        for box in boxes:
            color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
            mask = sam_utils.run_sam2(
                self.sam_predictor,
                color_np_rgb,
                box.box,
                iterations=self.erosion_iterations,
            )
            named_masks.append(NamedMask(name=box.phrase, mask=mask))

        # gen pointcloud
        result_scene_data = generate_pointcloud_multiple_obj_with_name(
            depth,
            color_np_org,
            named_masks,
            self.zed.K_left,
            self.scale,
            self.max_depth,
        )
        return result_scene_data

    def silent_mode_multiple_grounding_dict(self, target_names: list[str]):
        # Target objects
        prompt = ""
        target_boxes = dict()

        for target in target_names:
            prompt += target + " ."
            target_boxes[target] = None
        # Capture image
        try:
            zed_status, left_image, right_image = self.zed.capture_images()
        except Exception as e:
            logger.exception(f"error{e}")
            return RuntimeError

        color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
        color_np_org = color_np.copy()

        # stereo inference
        left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
        right_gray = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2GRAY)
        depth, (H_scaled, W_scaled) = self.stereo_model.run_inference(
            left_gray, right_gray, self.zed.K_left, self.zed.baseline
        )

        named_masks = []
        # GroundingDINO detection

        """
        Maybe, maybe, I shouldn't request GroundingDINO to detect all kinds of things at once.
        Maybe we can try to detect things one by one.
        But then it won't be able to know if that things has been detected and recorded before...
        """

        boxes = self.groundingdino_predictor.predict_boxes(color_np_org, prompt)
        for box in boxes:
            if box.phrase not in target_boxes:
                logger.error(
                    f"Unknown Object {box.phrase} generated by GroundingDINO, Failed"
                )
                raise ValueError
            if target_boxes[box.phrase] is not None:
                logger.error(f"More than one {box.phrase} detected")
                raise ValueError
            target_boxes[box.phrase] = box
            print(box.box, box.logits, box.phrase)
        for target_box in target_boxes:
            if target_boxes[target_box] is None:
                logger.error(f"Object {target_box} is not detected")
                raise ValueError
        boxes = [target_boxes[target_name] for target_name in target_names]
        # SAM2 inference
        for box in boxes:
            color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
            mask = sam_utils.run_sam2(
                self.sam_predictor,
                color_np_rgb,
                box.box,
                iterations=self.erosion_iterations,
            )
            named_masks.append(NamedMask(name=box.phrase, mask=mask))

        # gen pointcloud
        result_scene_data = generate_pointcloud_multiple_obj_with_name_dict(
            depth,
            color_np_org,
            named_masks,
            self.zed.K_left,
            self.scale,
            self.max_depth,
        )
        return result_scene_data

    def silent_mode(self):
        try:
            # Capture image
            zed_status, left_image, right_image = self.zed.capture_images()
            color_np = left_image.get_data()[:, :, :3]  # Drop alpha channel
            color_np_org = color_np.copy()

            # Yolo detection
            df = self.yolo_detector.infer(color_np_org)
            cup_detections = df[df["name"] == "cup"]

            # Check detection result
            if cup_detections.empty:
                logging.error("No Cup")
                return None
            if len(cup_detections) > 1:
                logging.error(
                    f"Warning: Multiple cups ({len(cup_detections)}) detected, please keep only one cup in sight and retry."
                )
                return None

            # collect the box returned by yolo
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
                self.sam_predictor,
                color_np_rgb,
                box,
                iterations=self.erosion_iterations,
            )

            # zed inference
            left_gray = cv2.cvtColor(left_image.get_data(), cv2.COLOR_BGRA2GRAY)
            right_gray = cv2.cvtColor(right_image.get_data(), cv2.COLOR_BGRA2GRAY)
            depth, (H_scaled, W_scaled) = self.stereo_model.run_inference(
                left_gray, right_gray, self.zed.K_left, self.zed.baseline
            )

            # gen pointcloud
            result = generate_pointcloud(
                depth, color_np_org, mask, self.zed.K_left, self.scale, self.max_depth
            )
            return result

        except Exception:
            print("failed during pointcloud generation")
            return None

    def close(self):
        self.zed.close()
        cv2.destroyAllWindows()
