import pyzed.sl as sl
import numpy as np
import os
import cv2
import json


class ZedCamera:
    """A class to interface with a ZED camera."""

    def __init__(self, use_png="") -> None:
        """Initializes the ZedCamera object."""
        self.baseline: float
        self.camera: sl.Camera
        self.ext_ir1_to_color: np.ndarray
        self.K_left: np.ndarray
        self.W: int
        self.H: int
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.png_dir = None
        if use_png == "":
            print("use_png is empty-*+--------------")
            self.initialize_zed()
        else:
            self.initialize_zed_using_existing_png(use_png)

    def initialize_zed(self) -> None:
        """Initializes the ZED camera and sets camera parameters."""
        self.camera = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use VGA video mode
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        info = self.camera.get_camera_information()
        cam_params = info.camera_configuration.calibration_parameters

        self.K_left = np.array(
            [
                [cam_params.left_cam.fx, 0, cam_params.left_cam.cx],
                [0, cam_params.left_cam.fy, cam_params.left_cam.cy],
                [0, 0, 1],
            ]
        )

        self.baseline = (
            info.camera_configuration.calibration_parameters.get_camera_baseline()
            / 1000.0
        )

        self.W = (
            self.camera.get_camera_information().camera_configuration.resolution.width
        )
        self.H = (
            self.camera.get_camera_information().camera_configuration.resolution.height
        )

        self.ext_ir1_to_color = np.identity(4)

    def initialize_zed_using_existing_png(self, use_png) -> None:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_file_dir)
        self.png_dir = os.path.join(
            project_root_dir, "sample_data/zed_images/", use_png
        )
        self.baseline = 0
        self.K_left = 0
        left_image_path = os.path.join(self.png_dir, "left.png")
        right_image_path = os.path.join(self.png_dir, "right.png")
        left_image_np = cv2.imread(left_image_path)
        right_image_np = cv2.imread(right_image_path)
        h, w = left_image_np.shape[:2]
        self.left_image = sl.Mat(w, h, sl.MAT_TYPE.U8_C3, sl.MEM.CPU)
        self.right_image = sl.Mat(w, h, sl.MAT_TYPE.U8_C3, sl.MEM.CPU)
        np.copyto(self.left_image.get_data(), left_image_np)
        np.copyto(self.right_image.get_data(), right_image_np)
        # ZED INFO
        with open(os.path.join(self.png_dir, "zed_info.json"), "rb") as f:
            camera_data = json.load(f)
        self.K_left = np.array(camera_data["K_left"])
        self.baseline = camera_data["baseline"]

    def capture_images(self) -> tuple[sl.ERROR_CODE, sl.Mat, sl.Mat]:
        if (
            self.png_dir is not None
        ):  # use existing png instead of the actual camera, for test purpose
            return sl.ERROR_CODE.SUCCESS, self.left_image, self.right_image

        """Captures left and right images from the ZED camera."""
        status = self.camera.grab()
        if status == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.left_image, sl.VIEW.LEFT)
            self.camera.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        return status, self.left_image, self.right_image

    def close(self) -> None:
        """Closes the ZED camera."""
        self.camera.close()
