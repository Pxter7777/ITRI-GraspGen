"""
Sample-based validation for detected objects.
Compares detected bounding boxes against reference images using:
- Global HSV color histograms (fast overall color signature)
- Spatial HSV histograms (grid-based, preserves where colors appear)
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SampleMatcher:
    def __init__(self, sample_dir: str = "sample_data/refs", threshold: float = 0.65):
        repo_root = Path(__file__).resolve().parents[1]
        default_dir = repo_root / "sample_data/refs"
        self.sample_dir = Path(sample_dir) if sample_dir else default_dir
        self.threshold = threshold
        self.samples = {}
        self.spatial_samples = {}
        # Spatial histogram settings: 3x3 grid, reuse existing bin counts
        self.grid_size = (3, 3)
        self._load_samples()

    def _load_samples(self):
        """Load all reference images and compute their HSV and spatial HSV histograms."""
        if not self.sample_dir.exists():
            logger.warning(f"Sample directory {self.sample_dir} does not exist.")
            return

        for img_path in self.sample_dir.glob("*.[jp][pn]g"):
            name = img_path.stem
            img = cv2.imread(str(img_path))

            if img is None:
                logger.warning(f"Failed to read image {img_path}. Skipping.")
                continue

            # Compute global HSV histogram for the sample
            hist = self._compute_hsv_histogram(img)
            self.samples[name] = hist

            # Compute spatial HSV histograms (grid-based)
            spatial_hists = self._compute_spatial_hsv_histograms(img, self.grid_size)
            self.spatial_samples[name] = spatial_hists

            logger.info(f"Loaded sample: {name} (spatial cells={len(spatial_hists)})")

        logger.info(f"Loaded {len(self.samples)} sample references")

    def _compute_hsv_histogram(self, img: np.ndarray) -> np.ndarray:
        """
        Compute normalized HSV histogram for color comparison.

        Args:
            img: BGR image (H, W, 3)

        Returns:
            Normalized histogram
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Compute 3D histogram: Hue (180), Saturation (256), Value (256)
        # Using reduced bins for efficiency: [30, 32, 32]
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [30, 32, 32], [0, 180, 0, 256, 0, 256]
        )

        # Normalize
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _compute_spatial_hsv_histograms(
        self, img: np.ndarray, grid_size: tuple[int, int] = (3, 3)
    ) -> list[np.ndarray]:
        """
        Compute HSV histograms per grid cell to retain coarse spatial layout.

        Args:
            img: BGR image
            grid_size: (rows, cols) for the grid

        Returns:
            List of normalized histograms, one per cell in row-major order
        """
        rows, cols = grid_size
        h, w = img.shape[:2]
        cell_h = max(1, h // rows)
        cell_w = max(1, w // cols)

        hists: list[np.ndarray] = []
        for r in range(rows):
            for c in range(cols):
                y1 = r * cell_h
                x1 = c * cell_w
                # Last row/col consume remainder to cover full image
                y2 = h if r == rows - 1 else (r + 1) * cell_h
                x2 = w if c == cols - 1 else (c + 1) * cell_w
                cell = img[y1:y2, x1:x2]
                if cell.size == 0:
                    # Fallback to zero histogram if unexpected empty cell
                    hists.append(np.zeros((30 * 32 * 32,), dtype=np.float32))
                    continue
                hists.append(self._compute_hsv_histogram(cell))
        return hists

    def _compare_spatial_histograms(
        self, hists_a: list[np.ndarray], hists_b: list[np.ndarray]
    ) -> float:
        """
        Compare spatial histograms cell-by-cell and average the scores.
        """
        if not hists_a or not hists_b or len(hists_a) != len(hists_b):
            return 0.0
        scores = []
        for ha, hb in zip(hists_a, hists_b, strict=False):
            s = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
            s = (s + 1) / 2  # map [-1,1] -> [0,1]
            scores.append(max(0.0, min(1.0, float(s))))
        return float(sum(scores) / len(scores))

    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compare two histograms using correlation method.

        Returns:
            Similarity score between 0 (different) and 1 (identical)
        """
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        # Correlation returns values in [-1, 1], map to [0, 1]
        return (score + 1) / 2

    def validate_boxes(
        self,
        image: np.ndarray,
        boxes: list,
        target_names: list[str],
    ) -> tuple[bool, dict]:
        """
        Validate detected boxes against reference samples.

        Args:
            image: Captured BGR image (H, W, 3).
            boxes: List of detected box objects with .box attribute [x1, y1, x2, y2]
            target_names: List of expected object names (must match sample filenames)

        Returns:
            (passed, scores): Boolean pass/fail and dict of {name: score} for each detection
        """

        if len(self.samples) == 0:
            logger.warning("No samples loaded, skipping validation")
            return True, {}

        scores = {}
        all_passed = True

        for box, target_name in zip(boxes, target_names, strict=False):
            # Normalize target name (handle variations)
            sample_key = target_name.replace(" ", "_")

            if sample_key not in self.samples:
                logger.warning(
                    f"No reference sample for '{target_name}', skipping validation"
                )
                scores[target_name] = 1.0  # Pass by default if no reference
                continue

            # Crop the detected region
            x1, y1, x2, y2 = map(int, box.box)
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                logger.error(f"Empty crop for {target_name} at box {box.box}")
                scores[target_name] = 0.0
                all_passed = False
                continue

            # Compute histogram for detected region
            detected_hist = self._compute_hsv_histogram(cropped)

            # Compare with reference
            score = self._compare_histograms(detected_hist, self.samples[sample_key])
            scores[target_name] = score

            if score < self.threshold:
                logger.warning(
                    f"Validation failed for '{target_name}': "
                    f"score={score:.3f} < threshold={self.threshold:.3f}"
                )
                all_passed = False
            else:
                logger.info(f"Validation passed for '{target_name}': score={score:.3f}")

        return all_passed, scores

    def filter_boxes(self, image: np.ndarray, target_names: dict) -> tuple[list, dict]:
        """
        Filter detected boxes based on combined global + spatial histogram scores.
        """

        if len(self.samples) == 0:
            logger.error(
                f"No samples loaded from {self.sample_dir}; failing validation."
            )
            return [], {}

        kept = []
        scores = {}
        for target_name, boxes_list in target_names.items():
            sample_key = target_name.replace(" ", "_")

            if sample_key not in self.samples:
                logger.warning(
                    f"No reference sample for '{target_name}', skipping validation"
                )
                scores[target_name] = 1.0
                continue

            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box.box)
                cropped = image[y1:y2, x1:x2]

                if cropped.size == 0:
                    logger.error(f"Empty crop for {target_name} at box {box.box}")
                    scores[target_name] = 0.0
                    continue

                # Global histogram score
                detected_hist = self._compute_hsv_histogram(cropped)
                global_score = self._compare_histograms(
                    detected_hist, self.samples[sample_key]
                )

                # Spatial histogram score
                det_spatial = self._compute_spatial_hsv_histograms(
                    cropped, self.grid_size
                )
                ref_spatial = self.spatial_samples.get(sample_key)
                spatial_score = self._compare_spatial_histograms(
                    det_spatial, ref_spatial
                )

                # Combine: emphasize spatial layout to combat pixel shuffling
                combined = 0.3 * global_score + 0.7 * spatial_score
                scores[target_name] = combined

                if combined >= self.threshold:
                    kept.append(box)
                else:
                    logger.warning(
                        f"Filtering out '{target_name}': score={combined:.3f} < threshold={self.threshold:.3f}"
                    )

        return kept, scores
