"""Grounding DINO detection utilities."""

import cv2
import groundingdino.datasets.transforms as transforms
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict
from PIL import Image
from torchvision.ops import box_convert

from common_utils import config
from pointcloud_generation.mouse_handlerv2 import BoundingBox


class DetectedBoxInfo:
    """Store a detected bounding box with its phrase and confidence.

    Args:
        box (BoundingBox): Bounding box coordinates.
        phrase (str): Detected phrase label.
        logits (float): Confidence score.

    Attributes:
        box (BoundingBox): Bounding box coordinates.
        phrase (str): Detected phrase label.
        logits (float): Confidence score.
    """

    box: BoundingBox
    phrase: str
    logits: float

    def __init__(self, box: BoundingBox, phrase: str, logits: float) -> None:
        self.box = box
        self.phrase = phrase
        self.logits = logits


class GroundindDinoPredictor:
    """Wrap the Grounding DINO model for bounding box prediction.

    Attributes:
        model (object): Loaded Grounding DINO model.
    """

    model: object

    def __init__(self) -> None:
        self.model = load_model(
            str(config.GROUNDINGDINO_CFG),
            str(config.GROUNDINGDINO_CKPT),
        )

    def predict_boxes(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.4,
    ) -> list[DetectedBoxInfo]:
        """Predict bounding boxes for the given text prompt in an image.

        Args:
            image (np.ndarray): BGR input image.
            text_prompt (str): Text prompt describing objects to detect.
            box_threshold (float): Minimum box confidence threshold.
            text_threshold (float): Minimum text confidence threshold.

        Returns:
            list[DetectedBoxInfo]: Detected boxes with phrases and scores.
        """
        transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        h, w, _ = image.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

        results = []
        for i in range(boxes.size(0)):
            coords = xyxy_boxes[i].numpy().astype(int).tolist()
            results.append(
                DetectedBoxInfo(
                    box=BoundingBox(
                        x_min=coords[0],
                        y_min=coords[1],
                        x_max=coords[2],
                        y_max=coords[3],
                    ),
                    phrase=phrases[i],
                    logits=logits[i].item(),
                )
            )

        return results


def main() -> None:
    """Run a quick smoke test of the predictor."""
    predictor = GroundindDinoPredictor()
    predictor.predict_boxes(np.zeros((100, 100, 3), dtype=np.uint8), "")
