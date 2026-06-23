"""Shared pytest fixtures for M.A.L.L.I. test suite."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Alias for pytest's tmp_path fixture."""
    return tmp_path


@pytest.fixture
def fake_cell_image_rgb() -> np.ndarray:
    """Random uint8 RGB image simulating a single cell crop."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def fake_smear_image_bgr() -> np.ndarray:
    """Random uint8 BGR image simulating a whole blood smear slide (1024×1024)."""
    rng = np.random.default_rng(1)
    return rng.integers(100, 200, (1024, 1024, 3), dtype=np.uint8)


@pytest.fixture
def nih_fixture_dir(tmp_path: Path) -> Path:
    """Minimal NIH-like dataset with 4 images per class."""
    rng = np.random.default_rng(42)
    for cls in ("Parasitized", "Uninfected"):
        cls_dir = tmp_path / "cell_images" / cls
        cls_dir.mkdir(parents=True)
        for i in range(4):
            img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(cls_dir / f"img_{i:03d}.png"), img)
    return tmp_path


@pytest.fixture
def smear_fixture_dir(tmp_path: Path) -> Path:
    """Minimal blood smear dataset: 2 slides + labels.csv."""
    rng = np.random.default_rng(7)
    smear_dir = tmp_path / "smear"
    smear_dir.mkdir()
    import cv2
    for i, label in enumerate([0, 1]):
        img = rng.integers(100, 200, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(smear_dir / f"slide_{i:03d}.png"), img)
    # Write labels.csv
    with (smear_dir / "labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerow({"filename": "slide_000.png", "label": 0})
        writer.writerow({"filename": "slide_001.png", "label": 1})
    return smear_dir
