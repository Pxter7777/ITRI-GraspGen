"""
Sample-based validation for detected objects.
Compares detected bounding boxes against reference images using color histograms.
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
        self._load_samples()

    def _load_samples(self):
        """Load all reference images and compute their HSV histograms."""
        if not self.sample_dir.exists():
            logger.warning(f"Sample directory {self.sample_dir} does not exist.")
            return
        
        for img_path in self.sample_dir.glob("*.[jp][pn]g"):
            name = img_path.stem
            img = cv2.imread(str(img_path))

            if img is None:
                logger.warning(f"Failed to read image {img_path}. Skipping.")
                continue
            
            # Compute HSV histogram for the sample
            hist = self._compute_hsv_histogram(img)
            self.samples[name] = hist
            logger.info(f"Loaded sample: {name}")
        
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
            [hsv],
            [0, 1, 2],
            None,
            [30, 32, 32],
            [0, 180, 0, 256, 0, 256]
        )
        
        # Normalize
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
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
        
        for box, target_name in zip(boxes, target_names):
            # Normalize target name (handle variations)
            sample_key = target_name.replace(" ", "_")
            
            if sample_key not in self.samples:
                logger.warning(f"No reference sample for '{target_name}', skipping validation")
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
        Filter detected boxes based on validation scores.
        """

        if len(self.samples) == 0:
            logger.error(f"No samples loaded from {self.sample_dir}; failing validation.")
            return [], {}
        
        kept = []
        scores = {}
        for target_name, boxes_list in target_names.items():
            sample_key = target_name.replace(" ", "_")
            
            if sample_key not in self.samples:
                logger.warning(f"No reference sample for '{target_name}', skipping validation")
                scores[target_name] = 1.0
                continue
            
            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box.box)
                cropped = image[y1:y2, x1:x2]
            
                if cropped.size == 0:
                    logger.error(f"Empty crop for {target_name} at box {box.box}")
                    scores[target_name] = 0.0
                    continue
                
                detected_hist = self._compute_hsv_histogram(cropped)
                score = self._compare_histograms(detected_hist, self.samples[sample_key])
                scores[target_name] = score
                
                if score >= self.threshold:
                    kept.append(box)
                else:
                    logger.warning(f"Filtering out '{target_name}': score={score:.3f} < threshold={self.threshold:.3f}")
            
        return kept, scores