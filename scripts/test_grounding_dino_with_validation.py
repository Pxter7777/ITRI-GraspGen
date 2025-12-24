#!/usr/bin/env python
"""
Demo: Test GroundingDINO detection with sample validation.

This script:
1. Loads an image from sample_data/zed_images/
2. Runs GroundingDINO to detect objects
3. Validates detections using SampleMatcher
4. Displays results with bounding boxes

Usage:
    python scripts/test_grounding_dino_with_validation.py --camera
    python scripts/test_grounding_dino_with_validation.py --demo demo7 --targets "green cup" "blue cup"
    python scripts/test_grounding_dino_with_validation.py --image sample_data/zed_images/demo7/left.png
"""

import sys
import argparse
import cv2
import logging
from pathlib import Path

from PointCloud_Generation.grounding_dino_utils import GroundindDinoPredictor
from PointCloud_Generation.sample_validation import SampleMatcher
from PointCloud_Generation.zed_utils import ZedCamera

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_boxes_with_scores(image, boxes, target_names, scores, kept_indices=None):
    """
    Draw bounding boxes and validation scores on the image.

    Args:
        image: BGR image to draw on
        boxes: List of box objects with .box attribute [x1, y1, x2, y2]
        target_names: List of object names
        scores: Dict of {name: score}
        kept_indices: Set of indices that passed validation (green), others red
    """
    result = image.copy()

    for idx, (box, name) in enumerate(zip(boxes, target_names, strict=False)):
        x1, y1, x2, y2 = map(int, box.box)
        score = scores.get(name, 0.0)

        # Color: green if kept, red if filtered
        if kept_indices is not None:
            color = (0, 255, 0) if idx in kept_indices else (0, 0, 255)
        else:
            color = (0, 255, 0)

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Draw label with score (multi-line handling)
        lines = [f"{name}", f"conf={box.logits:.2f}", f"val={score:.3f}"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_gap = 2
        pad = 2

        # Measure each line
        line_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
        max_w = max((sz[0] for sz in line_sizes), default=0)
        total_h = sum((sz[1] for sz in line_sizes), 0) + line_gap * (
            len(lines) - 1 if lines else 0
        )

        # Position the label: prefer above the box; if not enough space, place below
        text_x = x1
        preferred_top = y1 - 5 - total_h - pad
        if preferred_top < 0:
            # place below the box
            y_start = y2 + 5
        else:
            y_start = preferred_top

        # Background rectangle (semi-transparent)
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (text_x, max(0, y_start - pad)),
            (
                min(result.shape[1] - 1, text_x + max_w + 2 * pad),
                min(result.shape[0] - 1, y_start + total_h + pad),
            ),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

        # Render each line stacked
        y_cursor = y_start + (line_sizes[0][1] if line_sizes else 0)
        for t, (_w_i, h_i) in zip(lines, line_sizes, strict=False):
            cv2.putText(
                result,
                t,
                (text_x + pad, y_cursor),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            y_cursor += h_i + line_gap

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test GroundingDINO + sample validation"
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use live ZED camera instead of static image",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file (overrides --demo)",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="demo7",
        help="Demo folder name in sample_data/zed_images/ (default: demo7)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["green cup", "blue cup"],
        help="Objects to detect (default: 'green cup' 'blue cup')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Validation threshold (default: 0.60)",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default="sample_data/refs",
        help="Directory with reference samples",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display the result image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save result image to this path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second for live camera (default: 1, set to 0 for all)",
    )

    args = parser.parse_args()

    # Initialize GroundingDINO
    logger.info("Initializing GroundingDINO...")
    try:
        gd_predictor = GroundindDinoPredictor()
    except Exception as e:
        logger.error(f"Failed to initialize GroundingDINO: {e}")
        return 1

    # Initialize SampleMatcher
    logger.info(f"Initializing SampleMatcher (threshold={args.threshold})...")
    try:
        matcher = SampleMatcher(sample_dir=args.sample_dir, threshold=args.threshold)
    except Exception as e:
        logger.error(f"Failed to initialize SampleMatcher: {e}")
        return 1

    logger.info(
        f"Loaded {len(matcher.samples)} reference samples: {list(matcher.samples.keys())}"
    )

    # Camera or static image?
    if args.camera:
        return run_camera_mode(gd_predictor, matcher, args)
    else:
        return run_static_mode(gd_predictor, matcher, args)


def process_frame(image, gd_predictor, matcher, args, frame_num=None):
    """Detect and validate objects in a single frame."""
    prompt = " . ".join(args.targets) + " ."

    # Run detection
    try:
        boxes = gd_predictor.predict_boxes(image, prompt)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return None

    if not boxes:
        logger.warning(
            f"Frame {frame_num}: No boxes detected"
            if frame_num
            else "No boxes detected"
        )
        boxes_by_name = {name: [] for name in args.targets}
    else:
        if frame_num:
            logger.info(f"Frame {frame_num}: Detected {len(boxes)} boxes")
        else:
            logger.info(f"Detected {len(boxes)} boxes")
        for box in boxes:
            logger.info(f"  - {box.phrase}: confidence={box.logits:.3f}")

        # Group boxes by name
        boxes_by_name = {name: [] for name in args.targets}
        for box in boxes:
            if box.phrase in boxes_by_name:
                boxes_by_name[box.phrase].append(box)

    # Validate boxes
    try:
        filtered_boxes, scores = matcher.filter_boxes(image, boxes_by_name)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return None

    logger.info(f"  Validation: {len(filtered_boxes)} passed, scores={scores}")

    # Build list of all original boxes with kept status
    all_boxes = []
    all_names = []
    kept_indices = set()
    idx = 0

    for box in boxes:
        all_boxes.append(box)
        all_names.append(box.phrase)
        if box in filtered_boxes:
            kept_indices.add(idx)
        idx += 1

    # Draw results
    result_image = draw_boxes_with_scores(
        image,
        all_boxes,
        all_names,
        scores,
        kept_indices=kept_indices if filtered_boxes else None,
    )

    return {
        "image": result_image,
        "boxes": all_boxes,
        "names": all_names,
        "scores": scores,
        "validated": len(filtered_boxes),
        "total": len(boxes),
    }


def run_static_mode(gd_predictor, matcher, args):
    """Run on a single static image."""
    # Resolve image path
    if args.image:
        image_path = Path(args.image)
    else:
        image_path = repo_root / "sample_data" / "zed_images" / args.demo / "left.png"

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return 1

    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return 1

    logger.info(f"Image shape: {image.shape}")
    logger.info(f"Targets: {args.targets}")

    result = process_frame(image, gd_predictor, matcher, args)
    if result is None:
        return 1

    result_image = result["image"]

    # Display or save
    if args.show:
        cv2.namedWindow("GroundingDINO + Validation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GroundingDINO + Validation", 1280, 720)
        cv2.imshow("GroundingDINO + Validation", result_image)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_image)
        logger.info(f"Saved result to: {output_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Image: {image_path}")
    logger.info(f"  Detections: {result['total']}")
    logger.info(f"  Validated: {result['validated']}")
    logger.info(f"  Threshold: {args.threshold}")
    logger.info(f"  Scores: {result['scores']}")
    logger.info("=" * 60)

    return 0


def run_camera_mode(gd_predictor, matcher, args):
    """Run on live ZED camera stream."""
    logger.info("Initializing ZED camera...")
    try:
        zed = ZedCamera(use_png="")
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        logger.info("Tip: If ZED camera is not available, use --demo flag instead")
        return 1

    logger.info(
        "Camera initialized. Press 'q' to quit, 's' to save frame, 'p' to pause"
    )
    logger.info(f"Targets: {args.targets}")

    frame_count = 0
    paused = False
    camera_fps = 30  # Assuming camera runs at 30 FPS
    frame_interval = max(1, camera_fps // args.fps) if args.fps > 0 else 1

    try:
        while True:
            frame_count += 1

            # Capture frame
            try:
                status, left_img, right_img = zed.capture_images()
                if status != 0:  # sl.ERROR_CODE.SUCCESS
                    logger.error(f"Failed to capture image: {status}")
                    continue
            except Exception as e:
                logger.error(f"Capture error: {e}")
                break

            # Convert to numpy
            try:
                image = left_img.get_data()[:, :, :3]  # BGR
            except Exception as e:
                logger.error(f"Failed to convert image: {e}")
                continue

            # Process frame (at desired FPS)
            if frame_count % frame_interval == 0 or paused:
                logger.info(f"Processing frame {frame_count}...")
                result = process_frame(image, gd_predictor, matcher, args, frame_count)
                if result is None:
                    continue
                result_image = result["image"]
            else:
                result_image = image

            # Display
            cv2.namedWindow("Camera - GroundingDINO + Validation", cv2.WINDOW_NORMAL)
            cv2.imshow("Camera - GroundingDINO + Validation", result_image)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quitting...")
                break
            elif key == ord("s"):
                output_path = Path("output") / f"camera_frame_{frame_count}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), result_image)
                logger.info(f"Saved frame to: {output_path}")
            elif key == ord("p"):
                paused = not paused
                logger.info(f"Paused: {paused}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        cv2.destroyAllWindows()
        try:
            zed.close()
        except Exception as e:
            logger.warning(f"Error closing camera: {e}")

    logger.info(f"Processed {frame_count} frames total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
