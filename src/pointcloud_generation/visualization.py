"""Visualization helpers for masks, boxes, and depth maps."""

import cv2
import numpy as np

from common_utils import config
from pointcloud_generation.grounding_dino_utils import DetectedBoxInfo


def overlay_mask_on_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay a semi-transparent mask on a frame.

    Args:
        frame (np.ndarray): BGR image to overlay onto.
        mask (np.ndarray): Boolean mask selecting overlay pixels.

    Returns:
        np.ndarray: The blended frame.
    """
    overlay = frame.copy()
    overlay[mask] = config.OVERLAY_COLOR_CV
    cv2.addWeighted(
        overlay, config.OVERLAY_ALPHA, frame, 1 - config.OVERLAY_ALPHA, 0, frame
    )
    return frame


def draw_box(
    frame: np.ndarray,
    start_point: tuple[int, int],
    end_point: tuple[int, int],
) -> np.ndarray:
    """Draw a bounding box rectangle on a frame.

    Args:
        frame (np.ndarray): Image to draw on.
        start_point (tuple[int, int]): Top-left corner of the box.
        end_point (tuple[int, int]): Bottom-right corner of the box.

    Returns:
        np.ndarray: The frame with the box drawn.
    """
    cv2.rectangle(frame, start_point, end_point, config.BOX_COLOR, 2)
    return frame


def vis_disparity(disp: np.ndarray, vmax_percent: int = 95) -> np.ndarray:
    """Visualize a disparity map as a color-mapped image.

    Args:
        disp (np.ndarray): Raw disparity map.
        vmax_percent (int): Percentile for clamping max disparity.

    Returns:
        np.ndarray: Color-mapped disparity visualization.
    """
    disp_vis = disp.copy()
    disp_vis[disp_vis == np.inf] = 0

    vmax = np.percentile(disp_vis, vmax_percent)
    disp_vis[disp_vis > vmax] = vmax
    disp_vis = disp_vis / (vmax + 1e-6)
    disp_vis = (disp_vis * 255).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    disp_vis[disp == np.inf] = [0, 0, 0]
    return disp_vis


def vis_depth(depth: np.ndarray, vmax_percent: int = 95) -> np.ndarray:
    """Visualize a depth map as a color-mapped image.

    Args:
        depth (np.ndarray): Raw depth map.
        vmax_percent (int): Percentile for clamping max depth.

    Returns:
        np.ndarray: Color-mapped depth visualization.
    """
    depth_vis = depth.copy()
    depth_vis[depth_vis == np.inf] = 0

    vmax = np.percentile(depth_vis, vmax_percent)
    depth_vis[depth_vis > vmax] = vmax
    depth_vis = depth_vis / (vmax + 1e-6)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_vis[depth == np.inf] = [0, 0, 0]
    return depth_vis


def visualize_named_box(display_frame: np.ndarray, box: DetectedBoxInfo) -> np.ndarray:
    """Draw a labeled detection box on a display frame.

    Args:
        display_frame (np.ndarray): Image to draw on.
        box (DetectedBoxInfo): Detection box with label and confidence.

    Returns:
        np.ndarray: The frame with the labeled box drawn.
    """
    overlay = display_frame.copy()
    start_point = (int(box.box.x_min), int(box.box.y_min))
    end_point = (int(box.box.x_max), int(box.box.y_max))
    cv2.rectangle(overlay, start_point, end_point, (0, 255, 0), 2)
    label = f"{box.phrase}: {box.logits:.2f}"
    cv2.putText(
        overlay,
        label,
        (start_point[0], start_point[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    return overlay


def visualize_mask(display_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a segmentation mask onto a display frame.

    Args:
        display_frame (np.ndarray): Image to blend onto.
        mask (np.ndarray): Boolean mask selecting overlay pixels.

    Returns:
        np.ndarray: The blended display frame.
    """
    overlay = display_frame.copy()
    overlay[mask] = config.OVERLAY_COLOR_CV
    cv2.addWeighted(
        overlay,
        config.OVERLAY_ALPHA,
        display_frame,
        1 - config.OVERLAY_ALPHA,
        0,
        display_frame,
    )
    return display_frame
