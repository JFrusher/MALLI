"""Tests for the ROI detection and cell-counting pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.roi_detection.roi_grabber import (
    preprocess_for_segmentation,
    extract_roi_proposals,
    non_max_suppression,
)


def test_preprocess_for_segmentation_returns_same_shape(fake_smear_image_bgr):
    result = preprocess_for_segmentation(fake_smear_image_bgr)
    assert result.shape == fake_smear_image_bgr.shape[:2], (
        "preprocess_for_segmentation should return a 2D grayscale array "
        "matching the input HxW"
    )


def test_preprocess_for_segmentation_output_dtype(fake_smear_image_bgr):
    result = preprocess_for_segmentation(fake_smear_image_bgr)
    assert result.dtype == np.uint8


def test_extract_roi_proposals_returns_list():
    """A synthetic mask with one clear blob should produce at least one proposal."""
    import cv2
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (128, 128), 20, 255, -1)
    image_shape = (256, 256, 3)
    proposals = extract_roi_proposals(mask, image_shape, roi_size=64)
    assert isinstance(proposals, list)
    assert len(proposals) >= 1, "Should detect at least one proposal for a circle blob"


def test_extract_roi_proposals_filters_tiny_blobs():
    """Blobs smaller than min_blob_area should be filtered out."""
    import cv2
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Draw a 2px circle — too small
    cv2.circle(mask, (64, 64), 2, 255, -1)
    image_shape = (256, 256, 3)
    proposals = extract_roi_proposals(mask, image_shape, roi_size=64, min_blob_area=80.0)
    assert len(proposals) == 0, "Tiny blobs below min_blob_area should be rejected"


def test_non_max_suppression_removes_overlapping_boxes():
    """Two heavily overlapping boxes of different scores — only top one kept."""
    boxes = np.array([
        [10, 10, 60, 60],  # high confidence
        [12, 12, 62, 62],  # heavily overlapping
    ], dtype=np.float32)
    scores = np.array([0.95, 0.70])
    kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0] == 0, "Should keep the higher-confidence box"


def test_non_max_suppression_keeps_non_overlapping_boxes():
    """Two non-overlapping boxes should both be kept."""
    boxes = np.array([
        [0, 0, 50, 50],
        [200, 200, 250, 250],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8])
    kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
    assert len(kept) == 2


def test_non_max_suppression_empty_input():
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.array([])
    kept = non_max_suppression(boxes, scores, iou_threshold=0.5)
    assert kept == []


def test_cell_counter_initializes(tmp_path):
    """CellCounter should initialize with default parameters."""
    import shutil
    import cv2

    weights_path = tmp_path / "best.weights.h5"
    threshold_path = tmp_path / "decision_threshold.json"
    threshold_path.write_text('{"threshold": 0.3}', encoding="utf-8")

    from src.models.factory import build_mobilenetv3_small
    model = build_mobilenetv3_small()
    model.save_weights(str(weights_path))

    from src.models.cell_counter import CellCounter
    counter = CellCounter(
        weights_path=str(weights_path),
        threshold_path=str(threshold_path),
        verbose=False,
    )
    assert counter.threshold == pytest.approx(0.3, abs=1e-5)
    assert counter.roi_size == 128
