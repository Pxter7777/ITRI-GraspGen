import os
import cv2
import logging
import json
from pathlib import Path
from pointcloud_generation.zed_utils import ZedCamera

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def main():
    zed = ZedCamera()
    try:
        while True:
            print("Please provide the <name> of actions to start, or type end to end.")
            text = input("./data/zed_images/<name>/: ")
            if text == "end":
                break
            zed_status, left_image, right_image = zed.capture_images()
            # mkdir and save the two images
            save_dir = PROJECT_ROOT_DIR / "data/zed_images/" / text
            os.makedirs(str(save_dir), exist_ok=True)
            cv2.imwrite(str(save_dir / "left.png"), left_image.get_data())
            cv2.imwrite(str(save_dir / "right.png"), right_image.get_data())

            # ZED INFO
            camera_data = {"K_left": zed.K_left.tolist(), "baseline": zed.baseline}
            json_path = save_dir / "zed_info.json"

            with open(json_path, "w") as f:
                json.dump(camera_data, f, indent=4)

            logger.info(f"Images saved to {save_dir}")
    finally:
        logger.info("Closing ZED...")
        zed.close()


if __name__ == "__main__":
    main()
