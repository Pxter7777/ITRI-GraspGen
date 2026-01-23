import pyzed.sl as sl
import numpy as np
import os
import cv2
import json
import zmq
import logging

try:
    from common_utils import port_config

    STREAM_TO_ZED = port_config.STREAM_TO_ZED
except ModuleNotFoundError:
    STREAM_TO_ZED = 9091
logger = logging.getLogger(__name__)

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)


class ZedCamera:
    """A class to interface with a ZED camera."""

    def __init__(self, use_png="", from_stream=False) -> None:
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
        self.from_stream = from_stream
        if self.from_stream:
            self.initialize_zed_using_stream()
            return
        if use_png != "":
            self.initialize_zed_using_existing_png(use_png)
            return
        ### not from stream nor from png, try to actually use the zed camera
        try:
            self.initialize_zed()
        except RuntimeError:
            logger.warning("Initializing Zed Camera failed, trying streaming source.")
            self.initialize_zed_using_stream()
        return

    def initialize_zed_using_stream(self):
        self.initialize_zed_using_existing_png("demo6")
        self.from_stream = True

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

    def capture_images_from_stream(
        self, port=STREAM_TO_ZED
    ) -> tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]:
        """
        To capture image from stream, here we immediately connect, try to grab info, and disconnect + close the socket completely.
        If it keeps the socket connect, but not actually grabbing info, the info will pile up and eventually cause out of memory.
        """
        ctx = zmq.Context()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"zed_raw")
        self.sub.setsockopt(zmq.LINGER, 0)  # <--- CRITICAL: Don't wait to close
        self.sub.setsockopt(zmq.RCVTIMEO, 500)  # timeout 500ms
        try:
            self.sub.connect(f"tcp://127.0.0.1:{port}")
            (topic, ts, l_shape, l_dtype, l_buf, r_shape, r_dtype, r_buf) = (
                self.sub.recv_multipart()
            )
        except zmq.Again as e:
            raise ValueError(
                "Failed to capture images from stream. Is try_stream.py running?"
            ) from e
        # Grab images from stream successful
        left_image = np.frombuffer(l_buf, dtype=np.dtype(l_dtype.decode())).reshape(
            eval(l_shape.decode())
        )
        right_image = np.frombuffer(r_buf, dtype=np.dtype(r_dtype.decode())).reshape(
            eval(r_shape.decode())
        )
        ts = int(ts.decode())
        self.sub.disconnect(f"tcp://127.0.0.1:{port}")
        self.sub.close()
        return (sl.ERROR_CODE.SUCCESS, left_image, right_image)

    def capture_images_from_exsisting_png(
        self,
    ) -> tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]:
        return (
            sl.ERROR_CODE.SUCCESS,
            self.left_image.get_data(),
            self.right_image.get_data(),
        )

    def capture_images(self) -> tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]:
        if self.from_stream:
            return self.capture_images_from_stream()
        if (
            self.png_dir is not None
        ):  # use existing png instead of the actual camera, for test purpose
            return self.capture_images_from_exsisting_png()

        """Captures left and right images from the ZED camera."""
        status = self.camera.grab()
        if status == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.left_image, sl.VIEW.LEFT)
            self.camera.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        return status, self.left_image.get_data(), self.right_image.get_data()

    def close(self) -> None:
        """Closes the ZED camera."""
        self.camera.close()
