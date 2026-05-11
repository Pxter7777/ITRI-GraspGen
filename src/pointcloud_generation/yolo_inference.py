"""YOLOv5 object detection inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    import pandas as pd


class YOLOv5Detector:
    """Wrap a YOLOv5 model for cup detection.

    Args:
        model_path (str): Path to the YOLOv5 weights file.
        device (str): Device to run inference on.
        conf (float): Confidence threshold.

    Attributes:
        device (torch.device): Torch device used for inference.
        model: Loaded YOLOv5 model.
    """

    def __init__(
        self,
        model_path: str = "yolov5s.pt",
        device: str = "cuda",
        conf: float = 0.6,
    ) -> None:
        print("🔹 Loading model, please wait...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.conf = conf  # confidence threshold setting
        print("✅ Model loaded successfully!")

    def infer(
        self, image: np.ndarray | str
    ) -> pd.DataFrame:
        """Run inference on an image path or numpy array.

        Return detection results as a pandas DataFrame.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        results = self.model(image)
        df = results.pandas().xyxy[0]
        # 只取 class == 41 的資料
        df = df[df["class"] == 41]
        return df

    def infer_and_show(
        self, image: np.ndarray | str
    ) -> None:
        """Run inference and display results in an OpenCV window."""
        if isinstance(image, str):
            image = cv2.imread(image)
        results = self.model(image)
        df = results.pandas().xyxy[0]

        # 只取 class == 41 的資料
        df = df[df["class"] == 41]

        # OpenCV畫框
        for _, row in df.iterrows():
            x1, y1, x2, y2 = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )
            label = f"{row['name']} {row['confidence']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Cup Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = YOLOv5Detector(model_path="yolov5x.pt")

    print("\nEnter image path for inference (type 'q' to quit):")
    while True:
        path = input("Image path: ").strip()
        if path.lower() == "q":
            print("👋 Exit.")
            break
        if not Path(path).exists():
            print("⚠️ File not found, try again.")
            continue

        # df = detector.infer(path)
        # print(df)
        df = detector.infer_and_show(path)
