"""Cell counting pipeline using ROI extraction + MobileNetV3 classification.

Comprehensive pipeline for counting and classifying cells in microscopy images:
1. Loads image and detects smear type
2. Preprocesses with CLAHE contrast normalization
3. Segments using watershed (thick) or morphological opening (thin)
4. Extracts ROI proposals from contours
5. Runs batch MobileNetV3 inference on ROIs
6. Applies threshold filtering (infected vs uninfected)
7. De-duplicates with Non-Maximum Suppression
8. Returns per-image cell counts and statistics

Example:
    counter = CellCounter(verbose=True)
    result = counter.process_image("image.jpg")
    print(f"Infected: {result.infected_cells}/{result.total_cells}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import logging
from collections import Counter

import cv2
import numpy as np
import tensorflow as tf

from .inference import load_model_with_weights, load_decision_threshold
from .roi_detection.roi_grabber import (
    preprocess_for_segmentation,
    segment_thick_smear_watershed,
    segment_thin_smear,
    extract_roi_proposals,
    batch_predict_rois,
    non_max_suppression,
    RoiProposal,
)


logger = logging.getLogger(__name__)


@dataclass
class CellCountResult:
    """Results from processing a single image.

    Attributes:
        image_path: Path to the processed image
        total_cells: Total number of detected cells
        infected_cells: Number of cells classified as infected (above threshold)
        uninfected_cells: Number of cells classified as uninfected
        parasitemia_percent: Percentage of infected cells
        raw_proposal_count: Total ROI proposals before NMS
        smear_type: Detected or provided smear type (thick/thin)
        proposals: List of RoiProposal objects
        probabilities: Raw model output probabilities [0, 1] for each proposal
        kept_indices: Indices of cells retained after NMS (infected cells only)
        metadata: Optional additional statistics
    """

    image_path: str
    total_cells: int
    infected_cells: int
    uninfected_cells: int
    parasitemia_percent: float
    raw_proposal_count: int
    smear_type: str
    proposals: Sequence[RoiProposal] = field(default_factory=list)
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    kept_indices: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Format results as human-readable string."""
        filename = Path(self.image_path).name
        return (
            f"{filename} ({self.smear_type}): "
            f"{self.infected_cells}/{self.total_cells} infected "
            f"({self.parasitemia_percent:.1f}%)"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export or JSON serialization."""
        return {
            "image_path": self.image_path,
            "total_cells": self.total_cells,
            "infected_cells": self.infected_cells,
            "uninfected_cells": self.uninfected_cells,
            "parasitemia_percent": self.parasitemia_percent,
            "raw_proposal_count": self.raw_proposal_count,
            "smear_type": self.smear_type,
            "kept_indices_count": len(self.kept_indices),
        }


class CellCounter:
    """Pipeline for counting and classifying cells in microscopy images.

    Orchestrates the complete workflow from image loading through cell
    classification and NMS deduplication. Designed for efficient batch
    processing with configurable parameters.
    """

    def __init__(
        self,
        weights_path: str | Path = "models/best_mobilenetv3_small.weights.h5",
        threshold_path: str | Path | None = "models/decision_threshold.json",
        model_input_size: tuple[int, int] = (224, 224),
        roi_size: int = 128,
        batch_size: int = 32,
        nms_iou_threshold: float = 0.5,
        min_blob_area: float = 80.0,
        max_blob_area: float = 50_000.0,
        verbose: bool = False,
    ) -> None:
        """Initialize the cell counting pipeline.

        Args:
            weights_path: Path to MobileNetV3 best weights (h5 or keras format)
            threshold_path: Path to decision_threshold.json for loading calibrated
                threshold. If None, defaults to 0.3
            model_input_size: Input resolution for MobileNetV3 (H, W)
            roi_size: Size of square ROI crops extracted around cell centroids
            batch_size: Batch size for MobileNetV3 inference
            nms_iou_threshold: IOU threshold for Non-Maximum Suppression
            min_blob_area: Minimum cell area in pixels (filters noise)
            max_blob_area: Maximum cell area in pixels (filters artifacts)
            verbose: Enable DEBUG-level logging

        Raises:
            FileNotFoundError: If weights_path does not exist
            ValueError: If model fails to load
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        if verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            )

        logger.debug(f"Loading model from {weights_path}")
        self.model = load_model_with_weights(
            str(weights_path),
            input_shape=(model_input_size[0], model_input_size[1], 3),
            compile_model=False,
        )
        logger.debug("Model loaded successfully")

        self.threshold = load_decision_threshold(threshold_path, default=0.3)
        self.model_input_size = model_input_size
        self.roi_size = roi_size
        self.batch_size = batch_size
        self.nms_iou_threshold = nms_iou_threshold
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area

        logger.info(
            f"CellCounter initialized | threshold={self.threshold:.3f} | "
            f"roi_size={roi_size} | batch_size={batch_size} | "
            f"nms_iou={nms_iou_threshold}"
        )

    def process_image(
        self,
        image_path: str | Path,
        smear_type: str = "auto",
    ) -> CellCountResult:
        """Process a single image and count cells.

        Orchestrates the complete pipeline:
        1. Load image from disk
        2. Auto-detect smear type if needed (based on dimensions)
        3. Preprocess with CLAHE
        4. Segment (watershed for thick, morphological for thin)
        5. Extract ROI proposals from contours
        6. Run batch MobileNetV3 inference
        7. Filter by threshold
        8. Apply NMS deduplication
        9. Calculate statistics

        Args:
            image_path: Path to image file (any OpenCV-supported format)
            smear_type: "thin", "thick", or "auto" (default: infer from dimensions)

        Returns:
            CellCountResult with all counts, proposals, and probabilities

        Raises:
            ValueError: If image cannot be read
            RuntimeError: If segmentation or inference fails
        """
        image_path = Path(image_path)
        logger.debug(f"Processing image: {image_path}")

        # Load image
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = image_bgr.shape[:2]
        logger.debug(f"Image loaded: {width}x{height}")

        # Auto-detect smear type if needed
        if smear_type == "auto":
            smear_type = "thick" if min(height, width) >= 512 else "thin"
            logger.debug(f"Auto-detected smear type: {smear_type}")
        elif smear_type not in ("thick", "thin"):
            raise ValueError(f"Invalid smear_type: {smear_type}. Must be 'thin', 'thick', or 'auto'")

        # Preprocess for segmentation
        try:
            preprocessed = preprocess_for_segmentation(image_bgr)
            logger.debug("Preprocessing complete")
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}") from e

        # Segment image
        try:
            if smear_type == "thick":
                mask = segment_thick_smear_watershed(image_bgr, preprocessed)
                logger.debug("Thick smear watershed segmentation complete")
            else:
                mask = segment_thin_smear(preprocessed)
                logger.debug("Thin smear morphological segmentation complete")
        except Exception as e:
            raise RuntimeError(f"Segmentation failed: {e}") from e

        # Extract ROI proposals
        try:
            proposals = extract_roi_proposals(
                mask,
                image_shape=image_bgr.shape,
                roi_size=self.roi_size,
                min_blob_area=self.min_blob_area,
                max_blob_area=self.max_blob_area,
            )
            logger.debug(f"Extracted {len(proposals)} ROI proposals")
        except Exception as e:
            raise RuntimeError(f"ROI extraction failed: {e}") from e

        # If no proposals found, return early
        if not proposals:
            result = CellCountResult(
                image_path=str(image_path),
                total_cells=0,
                infected_cells=0,
                uninfected_cells=0,
                parasitemia_percent=0.0,
                raw_proposal_count=0,
                smear_type=smear_type,
                proposals=[],
                probabilities=np.array([], dtype=np.float32),
                kept_indices=[],
                metadata={"status": "no_cells_detected"},
            )
            logger.info(f"✓ {result}")
            return result

        # Run batch prediction on all ROIs
        try:
            probabilities = batch_predict_rois(
                model=self.model,
                image_bgr=image_bgr,
                proposals=proposals,
                model_input_size=self.model_input_size,
                batch_size=self.batch_size,
            )
            logger.debug(
                f"Batch prediction complete | "
                f"prob_min={probabilities.min():.3f} | "
                f"prob_max={probabilities.max():.3f} | "
                f"prob_mean={probabilities.mean():.3f}"
            )
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}") from e

        # Filter positive predictions (above threshold)
        positive_mask = probabilities >= self.threshold
        positive_indices = set(np.where(positive_mask)[0])
        logger.debug(f"Positive predictions: {len(positive_indices)}/{len(proposals)}")

        # Apply NMS to de-duplicate overlapping infected cells
        kept_indices: list[int] = []
        if positive_indices:
            try:
                # Extract boxes and scores for positive predictions only
                positive_list = sorted(positive_indices)
                positive_boxes = np.array(
                    [proposals[i].box for i in positive_list],
                    dtype=np.float32,
                )
                positive_scores = np.array(
                    [probabilities[i] for i in positive_list],
                    dtype=np.float32,
                )

                # NMS returns indices relative to positive_list
                kept_relative_indices = non_max_suppression(
                    positive_boxes,
                    positive_scores,
                    self.nms_iou_threshold,
                )

                # Convert back to original proposal indices
                kept_indices = [positive_list[i] for i in kept_relative_indices]
                logger.debug(
                    f"NMS de-duplication: {len(positive_indices)} → {len(kept_indices)} "
                    f"(IOU thresh={self.nms_iou_threshold})"
                )
            except Exception as e:
                logger.warning(f"NMS failed, using all positive detections: {e}")
                kept_indices = positive_list

        # Calculate statistics
        total_count = len(proposals)
        infected_count = len(kept_indices)
        uninfected_count = total_count - infected_count
        parasitemia = (infected_count / total_count * 100) if total_count > 0 else 0.0

        result = CellCountResult(
            image_path=str(image_path),
            total_cells=total_count,
            infected_cells=infected_count,
            uninfected_cells=uninfected_count,
            parasitemia_percent=parasitemia,
            raw_proposal_count=len(proposals),
            smear_type=smear_type,
            proposals=proposals,
            probabilities=probabilities,
            kept_indices=kept_indices,
            metadata={
                "threshold_applied": self.threshold,
                "positive_before_nms": len(positive_indices),
            },
        )

        logger.info(f"✓ {result}")
        return result

    def process_batch(
        self,
        image_paths: Sequence[str | Path],
        smear_type: str = "auto",
        skip_errors: bool = True,
    ) -> list[CellCountResult]:
        """Process multiple images efficiently.

        Args:
            image_paths: List of image file paths
            smear_type: "thin", "thick", or "auto" (applied to all images)
            skip_errors: If True, log errors and continue; if False, raise

        Returns:
            List of CellCountResult objects (in same order as input)

        Raises:
            RuntimeError: If skip_errors=False and any image fails
        """
        results: list[CellCountResult] = []
        failed_count = 0

        logger.info(f"Processing batch of {len(image_paths)} images")

        for i, image_path in enumerate(image_paths, 1):
            try:
                result = self.process_image(image_path, smear_type=smear_type)
                results.append(result)
            except Exception as e:
                failed_count += 1
                error_msg = f"[{i}/{len(image_paths)}] Failed to process {image_path}: {e}"
                if skip_errors:
                    logger.error(error_msg)
                else:
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

        logger.info(
            f"Batch processing complete: {len(results)} successful, {failed_count} failed"
        )
        return results

    def process_directory(
        self,
        directory: str | Path,
        pattern: str = "*.png",
        recursive: bool = True,
        smear_type: str = "auto",
        skip_errors: bool = True,
    ) -> list[CellCountResult]:
        """Process all images in a directory matching a glob pattern.

        Args:
            directory: Root directory to search
            pattern: Glob pattern for image files (default: *.png)
            recursive: If True, search subdirectories (default: True)
            smear_type: "thin", "thick", or "auto"
            skip_errors: If True, log errors and continue

        Returns:
            List of CellCountResult objects

        Raises:
            FileNotFoundError: If directory does not exist
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find matching images
        if recursive:
            image_paths = list(directory.rglob(pattern))
        else:
            image_paths = list(directory.glob(pattern))

        logger.info(f"Found {len(image_paths)} images in {directory}")

        if not image_paths:
            logger.warning(f"No images matching pattern '{pattern}' in {directory}")
            return []

        return self.process_batch(image_paths, smear_type=smear_type, skip_errors=skip_errors)

    def get_summary_statistics(self, results: Sequence[CellCountResult]) -> dict:
        """Compute summary statistics across multiple results.

        Args:
            results: List of CellCountResult objects

        Returns:
            Dictionary with aggregated statistics
        """
        if not results:
            return {}

        total_cells_all = sum(r.total_cells for r in results)
        infected_cells_all = sum(r.infected_cells for r in results)
        parasitemia_values = [r.parasitemia_percent for r in results if r.total_cells > 0]

        smear_counts = Counter(r.smear_type for r in results)

        return {
            "num_images": len(results),
            "total_cells": total_cells_all,
            "total_infected": infected_cells_all,
            "total_uninfected": total_cells_all - infected_cells_all,
            "mean_parasitemia_percent": (
                np.mean(parasitemia_values) if parasitemia_values else 0.0
            ),
            "std_parasitemia_percent": (
                np.std(parasitemia_values) if parasitemia_values else 0.0
            ),
            "min_parasitemia_percent": (
                float(np.min(parasitemia_values)) if parasitemia_values else 0.0
            ),
            "max_parasitemia_percent": (
                float(np.max(parasitemia_values)) if parasitemia_values else 0.0
            ),
            "smear_type_distribution": dict(smear_counts),
        }
