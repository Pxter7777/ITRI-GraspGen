import torch
import cv2
from pathlib import Path


class YOLOv5Detector:
    def __init__(self, model_path="yolov5s.pt", device="cuda", conf=0.6):
        print("🔹 Loading model, please wait...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.conf = conf  # confidence threshold setting
        print("✅ Model loaded successfully!")

    def infer(self, image):
        """
        image: 可為影像檔路徑(str) 或 OpenCV 影像(numpy array)
        回傳推論結果 (pandas DataFrame)
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        results = self.model(image)
        df = results.pandas().xyxy[0]
        # 只取 class == 41 的資料
        df = df[df["class"] == 41]
        return df

    def infer_and_show(self, image):
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
