import os
import cv2
from PointCloud_Generation.zed_utils import ZedCamera

def main():
    zed = ZedCamera()
    while True:
        print("Please provide the <name> of actions to start, or type end to end.")
        text = input("./sample_data/zed_images/<name>/: ")
        if text == "end":
            return
        zed_status, left_image, right_image = zed.capture_images()
        # mkdir and save the two images
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_file_dir)
        save_dir = os.path.join(project_root_dir, "sample_data/zed_images/", text)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "left.png"), left_image.get_data())
        cv2.imwrite(os.path.join(save_dir, "right.png"), right_image.get_data())
        print(f"Images saved to {save_dir}")
        

if __name__ == "__main__":
    main()