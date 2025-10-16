import os
import argparse
import logging
import numpy as np
import cv2
import torch

import sys

sys.path.insert(0, os.path.expanduser("~/Third_Party"))

from src import (
    config,
    mouse_handler,
    sam_utils,
    realsense_utils,
    stereo_utils,
    visualization,
    pointcloud_utils,
)


def set_logging_format():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default=str(config.FOUNDATIONSTEREO_CHECKPOINT),
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--scale",
        default=1,
        type=float,
        help="downsize the image by scale, must be <=1",
    )
    parser.add_argument("--hiera", default=0, type=int, help="hierarchical inference")
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--out_dir", default="./output/", type=str, help="the directory to save results"
    )
    parser.add_argument(
        "--erosion_iterations",
        type=int,
        default=0,  # can be 6
        help="Number of erosion iterations for the SAM mask.",
    )
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Load Models ----------
    stereo_model, stereo_args = stereo_utils.load_stereo_model(args)
    sam_predictor = sam_utils.load_sam_model()

    # ---------- RealSense Init ----------
    pipeline, align, K_ir1, K_color, ext_ir1_to_color, baseline, (W, H) = (
        realsense_utils.initialize_realsense()
    )

    # ---------- Window and Mouse Callback Setup ----------
    win_name = "RGB + Mask | Disparity"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_handler.select_box)
    logging.info("Streaming... Draw a box with your mouse.")
    logging.info("Press SPACE to save, 'r' to reset box, ESC to exit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            ir1_frame = aligned_frames.get_infrared_frame(1)
            ir2_frame = aligned_frames.get_infrared_frame(2)
            color_frame = aligned_frames.get_color_frame()

            if not ir1_frame or not ir2_frame or not color_frame:
                continue

            ir1_np = np.asanyarray(ir1_frame.get_data())
            ir2_np = np.asanyarray(ir2_frame.get_data())
            color_np = np.asanyarray(color_frame.get_data())
            color_np_org = color_np.copy()

            display_frame = color_np.copy()

            # ---------- Manual Box Selection + SAM2 Logic ----------
            mask = np.zeros_like(color_np[:, :, 0], dtype=bool)

            if mouse_handler.drawing_box:
                visualization.draw_box(
                    display_frame,
                    mouse_handler.box_start_point,
                    mouse_handler.box_end_point,
                )

            if mouse_handler.box_defined:
                box = mouse_handler.get_box()
                visualization.draw_box(
                    display_frame, (box[0], box[1]), (box[2], box[3])
                )

                color_np_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
                mask = sam_utils.run_sam2(
                    sam_predictor, color_np_rgb, box, iterations=args.erosion_iterations
                )
                display_frame = visualization.overlay_mask_on_frame(display_frame, mask)

            # ---------- FoundationStereo Inference ----------
            disp, (H_scaled, W_scaled) = stereo_utils.run_stereo_inference(
                stereo_model, ir1_np, ir2_np, stereo_args
            )

            vis_disp = visualization.vis_disparity(disp)
            vis_disp_resized = cv2.resize(
                vis_disp, fx=1 / stereo_args.scale, fy=1 / stereo_args.scale, dsize=None
            )

            combined_vis = np.concatenate([display_frame, vis_disp_resized], axis=1)
            cv2.imshow(win_name, combined_vis)
            key = cv2.waitKey(1)

            if key == ord("r"):
                logging.info("Box reset. Draw a new one.")
                mouse_handler.reset_box()

            if key == 32 and mouse_handler.box_defined:
                pointcloud_utils.save_scene_and_obj(
                    stereo_args,
                    K_ir1,
                    baseline,
                    disp,
                    ext_ir1_to_color,
                    K_color,
                    color_np_org,
                    mask,
                    combined_vis,
                )

            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
