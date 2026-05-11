import logging
import sys

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from common_utils import config
from pointcloud_generation.mouse_handlerv2 import BoundingBox


def load_sam_model():
    try:
        sam_model = build_sam2(config.SAM2_CFG, config.SAM2_CHECKPOINT).to(
            config.DEVICE
        )
        sam_predictor = SAM2ImagePredictor(sam_model)
        return sam_predictor
    except Exception as e:
        logging.error(f"Error loading SAM2 model: {e}")
        logging.error(
            f"Please check your checkpoint ('{config.SAM2_CHECKPOINT}') and config ('{config.SAM2_CFG}') paths."
        )
        sys.exit(1)


def run_sam2(predictor, image_rgb, box: BoundingBox, iterations=6):
    predictor.set_image(image_rgb)
    input_point = np.array([[(box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2]])
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
