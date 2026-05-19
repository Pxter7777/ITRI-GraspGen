"""ZED camera interface and image capture utilities."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import zmq

try:
    from common_utils import network_config

    _stream_port = network_config.STREAM_TO_ZED_PORT
except ModuleNotFoundError:
    _stream_port = 9091

STREAM_TO_ZED_PORT: int = _stream_port
logger = logging.getLogger(__name__)

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]


class ZedCamera:
    """A class to interface with a ZED camera.

    Args:
        use_png (str): Name of a saved PNG dataset to load instead of
            a live camera.
        from_stream (bool): Whether to receive frames via ZMQ stream.

    Attributes:
        baseline (float): Camera stereo baseline in meters.
        camera (sl.Camera): Underlying ZED SDK camera handle.
        ext_ir1_to_color (np.ndarray): Extrinsic transform from IR1 to color.
        K_left (np.ndarray): Left camera intrinsic matrix.
        W (int): Image width in pixels.
        H (int): Image height in pixels.
        left_image (sl.Mat): Left image buffer.
        right_image (sl.Mat): Right image buffer.
        png_dir (Path | None): Directory of pre-loaded PNG images.
        from_stream (bool): Whether frames come from a ZMQ stream.
        sub (zmq.Socket[bytes]): ZMQ subscriber socket.
    """

    baseline: float
    camera: sl.Camera
    ext_ir1_to_color: np.ndarray
    K_left: np.ndarray
    W: int
    H: int
    left_image: sl.Mat
    right_image: sl.Mat
    png_dir: Path | None
    from_stream: bool
    sub: zmq.Socket[bytes]

    def __init__(self, use_png: str = "", from_stream: bool = False) -> None:
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

    def initialize_zed_using_stream(self) -> None:
        """Initialize the camera from a streaming source."""
        self.initialize_zed_using_existing_png("demo6")
        self.from_stream = True

    def initialize_zed(self) -> None:
        """Initialize the ZED camera and set camera parameters.

        Raises:
            RuntimeError: If the camera fails to open.
        """
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

        self.W = (  # type: ignore[reportConstantRedefinition]
            self.camera.get_camera_information().camera_configuration.resolution.width
        )
        self.H = (  # type: ignore[reportConstantRedefinition]
            self.camera.get_camera_information().camera_configuration.resolution.height
        )

        self.ext_ir1_to_color = np.identity(4)

    def initialize_zed_using_existing_png(self, use_png: str) -> None:
        """Initialize camera parameters from saved PNG images and calibration data.

        Args:
            use_png (str): Name of the PNG dataset directory under ``data/zed_images/``.

        Raises:
            FileNotFoundError: If the provided --use-png dir name doesn't exist.
        """
        self.png_dir = PROJECT_ROOT_DIR / "data" / "zed_images" / use_png
        if not self.png_dir.exists():
            raise FileNotFoundError(
                f"There is no pre-captured images at {self.png_dir}"
            )
        self.baseline = 0.0
        self.K_left = np.zeros((3, 3))
        left_image_np = cv2.imread(str(self.png_dir / "left.png"))
        right_image_np = cv2.imread(str(self.png_dir / "right.png"))
        h, w = left_image_np.shape[:2]
        self.left_image = sl.Mat(w, h, sl.MAT_TYPE.U8_C3, sl.MEM.CPU)
        self.right_image = sl.Mat(w, h, sl.MAT_TYPE.U8_C3, sl.MEM.CPU)
        np.copyto(self.left_image.get_data(), left_image_np)
        np.copyto(self.right_image.get_data(), right_image_np)
        # ZED INFO
        with open(self.png_dir / "zed_info.json", "rb") as f:
            camera_data = json.load(f)
        self.K_left = np.array(camera_data["K_left"])
        self.baseline = camera_data["baseline"]

    def capture_images_from_stream(
        self, port: int = STREAM_TO_ZED_PORT
    ) -> tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]:
        """Capture images from a ZMQ stream.

        Connect, grab one frame, and disconnect immediately to avoid memory buildup.

        Args:
            port (int): TCP port to connect to.

        Returns:
            tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]: Status code, left image,
                and right image.

        Raises:
            ValueError: If the stream times out.
        """
        ctx = zmq.Context()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"zed_raw")
        self.sub.setsockopt(zmq.LINGER, 0)  # <--- CRITICAL: Don't wait to close
        self.sub.setsockopt(zmq.RCVTIMEO, 500)  # timeout 500ms
        try:
            self.sub.connect(f"tcp://127.0.0.1:{port}")
            (_topic, ts, l_shape, l_dtype, l_buf, r_shape, r_dtype, r_buf) = (
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
        """Return the pre-loaded PNG images.

        Returns:
            tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]: Status code, left image,
                and right image.
        """
        return (
            sl.ERROR_CODE.SUCCESS,
            self.left_image.get_data(),
            self.right_image.get_data(),
        )

    def capture_images(self) -> tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]:
        """Capture left and right images from the active source.

        Returns:
            tuple[sl.ERROR_CODE, np.ndarray, np.ndarray]: Status code, left image,
                and right image.
        """
        if self.from_stream:
            return self.capture_images_from_stream()
        if (
            self.png_dir is not None
        ):  # use existing png instead of the actual camera, for test purpose
            return self.capture_images_from_exsisting_png()

        status = self.camera.grab()
        if status == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.left_image, sl.VIEW.LEFT)
            self.camera.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        return status, self.left_image.get_data(), self.right_image.get_data()

    def close(self) -> None:
        """Closes the ZED camera."""
        if hasattr(self, "camera"):
            self.camera.close()
