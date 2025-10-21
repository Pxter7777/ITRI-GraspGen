import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import logging
import sys
from common_utils import config


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


def run_sam2(predictor, image_rgb, box, iterations=6):
    predictor.set_image(image_rgb)
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
