from groundingdino.util.inference import load_model, predict
import cv2
import numpy as np
from PIL import Image
import torch
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert


class DetectedBoxInfo:
    def __init__(self, box, phrase, logits):
        self.box = box  # match pixel format like in pointcloud_generation.py
        self.phrase = phrase
        self.logits = logits


class GroundindDinoPredictor:
    def __init__(self):
        self.model = load_model(
            "/home/j300/Third_Party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "/home/j300/models/GroundingDinoModels/groundingdino_swint_ogc.pth",
        )

    def predict_boxes(
        self, image: np.array, text_prompt: str, box_threshold=0.4, text_threshold=0.4
    ) -> list[DetectedBoxInfo]:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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
            results.append(
                DetectedBoxInfo(
                    box=tuple(xyxy_boxes[i].numpy().astype(int).tolist()),
                    phrase=phrases[i],
                    logits=logits[i].item(),
                )
            )

        return results


def main():
    predictor = GroundindDinoPredictor()
    predictor.predict_boxes("")
