import cv2
from . import config
import numpy as np


def overlay_mask_on_frame(frame, mask):
    overlay = frame.copy()
    overlay[mask] = config.OVERLAY_COLOR_CV
    cv2.addWeighted(
        overlay, config.OVERLAY_ALPHA, frame, 1 - config.OVERLAY_ALPHA, 0, frame
    )
    return frame


def draw_box(frame, start_point, end_point):
    cv2.rectangle(frame, start_point, end_point, config.BOX_COLOR, 2)
    return frame


def vis_disparity(disp, vmax_percent=95):
    """
    vmax: max value of disparity
    """
    disp_vis = disp.copy()
    disp_vis[disp_vis == np.inf] = 0

    if vmax_percent != 100:
        vmax = np.percentile(disp_vis, vmax_percent)
        disp_vis[disp_vis > vmax] = vmax

    disp_vis = disp_vis / (vmax + 1e-6)
    disp_vis = (disp_vis * 255).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    disp_vis[disp == np.inf] = [0, 0, 0]
    return disp_vis


def vis_depth(depth, vmax_percent=95):
    """
    vmax: max value of depth
    """
    depth_vis = depth.copy()
    depth_vis[depth_vis == np.inf] = 0

    if vmax_percent != 100:
        vmax = np.percentile(depth_vis, vmax_percent)
        depth_vis[depth_vis > vmax] = vmax

    depth_vis = depth_vis / (vmax + 1e-6)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_vis[depth == np.inf] = [0, 0, 0]
    return depth_vis
