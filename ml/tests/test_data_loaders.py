"""Tests for data loader modules."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import tensorflow as tf


class TestMalariaDataset:
    def test_creates_train_and_val_datasets(self, nih_fixture_dir):
        from src.data.loaders.nih_loader import MalariaDataset
        ds = MalariaDataset(
            dataset_root=nih_fixture_dir,
            image_size=(64, 64),
            batch_size=2,
            test_split=0.5,
            seed=0,
            use_tfrecord_cache=False,
        )
        train_ds, val_ds = ds.create_datasets()
        assert train_ds is not None
        assert val_ds is not None

    def test_dataset_yields_image_and_label(self, nih_fixture_dir):
        from src.data.loaders.nih_loader import MalariaDataset
        ds = MalariaDataset(
            dataset_root=nih_fixture_dir,
            image_size=(64, 64),
            batch_size=2,
            test_split=0.5,
            seed=0,
            use_tfrecord_cache=False,
        )
        train_ds, _ = ds.create_datasets()
        for batch in train_ds.take(1):
            images, labels = batch[0], batch[1]
            assert images.shape[-1] == 3
            assert labels.shape[0] == images.shape[0]

    def test_cache_manifest_written(self, nih_fixture_dir, tmp_path):
        from src.data.loaders.nih_loader import MalariaDataset
        cache_dir = tmp_path / "tfrecord"
        ds = MalariaDataset(
            dataset_root=nih_fixture_dir,
            image_size=(64, 64),
            batch_size=2,
            test_split=0.5,
            seed=0,
            use_tfrecord_cache=True,
            cache_dir=cache_dir,
        )
        ds.create_datasets()
        manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
        assert "schema_version" in manifest
        assert manifest["schema_version"] == 2

    def test_stratified_split_preserves_class_ratio(self, nih_fixture_dir):
        from src.data.loaders.nih_loader import MalariaDataset
        ds = MalariaDataset(
            dataset_root=nih_fixture_dir,
            image_size=(64, 64),
            batch_size=2,
            test_split=0.5,
            seed=42,
            use_tfrecord_cache=False,
        )
        samples = ds._collect_source_samples()
        train, val = ds._stratified_split(samples)
        train_labels = [s[1] for s in train]
        val_labels = [s[1] for s in val]
        # Each split should contain both classes
        assert 0 in train_labels
        assert 1 in train_labels
        assert 0 in val_labels
        assert 1 in val_labels


class TestSyntheticFieldReadyDataset:
    def _write_fixture(self, tmp_path: Path) -> Path:
        import cv2
        rng = np.random.default_rng(5)
        synth_dir = tmp_path / "synth"
        synth_dir.mkdir()
        rows = []
        for i in range(4):
            img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            fname = f"cell_{i:03d}.png"
            cv2.imwrite(str(synth_dir / fname), img)
            rows.append({"filename": fname, "soft_label": float(i % 2), "hard_label": i % 2})
        with (synth_dir / "labels.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "soft_label", "hard_label"])
            writer.writeheader()
            writer.writerows(rows)
        return synth_dir

    def test_creates_datasets(self, tmp_path):
        from src.data.loaders.synthetic_loader import SyntheticFieldReadyDataset
        synth_dir = self._write_fixture(tmp_path)
        ds = SyntheticFieldReadyDataset(
            dataset_root=synth_dir,
            labels_csv="labels.csv",
            image_size=(64, 64),
            batch_size=2,
            test_split=0.5,
            seed=0,
        )
        train_ds, val_ds = ds.create_datasets()
        assert train_ds is not None
        assert val_ds is not None

    def test_reads_csv_labels(self, tmp_path):
        from src.data.loaders.synthetic_loader import SyntheticFieldReadyDataset
        synth_dir = self._write_fixture(tmp_path)
        ds = SyntheticFieldReadyDataset(
            dataset_root=synth_dir,
            labels_csv="labels.csv",
            image_size=(64, 64),
            batch_size=4,
            test_split=0.5,
            seed=0,
        )
        samples = ds._collect_samples()
        assert len(samples) == 4
        labels = {s.hard_label for s in samples}
        assert labels == {0, 1}


class TestSmearROILoader:
    def test_loads_slide_records(self, smear_fixture_dir):
        from src.data.loaders.smear_roi_loader import SmearROILoader
        loader = SmearROILoader(
            smear_root=smear_fixture_dir,
            labels_csv=smear_fixture_dir / "labels.csv",
        )
        records = loader._load_slide_records()
        assert len(records) == 2

    def test_stratified_split_produces_train_val(self, smear_fixture_dir):
        from src.data.loaders.smear_roi_loader import SmearROILoader
        loader = SmearROILoader(
            smear_root=smear_fixture_dir,
            labels_csv=smear_fixture_dir / "labels.csv",
            test_split=0.5,
            seed=0,
        )
        records = loader._load_slide_records()
        train, val = loader._stratified_split(records)
        assert len(train) + len(val) == len(records)

    def test_raises_if_labels_csv_missing(self, tmp_path):
        from src.data.loaders.smear_roi_loader import SmearROILoader
        with pytest.raises(FileNotFoundError):
            loader = SmearROILoader(
                smear_root=tmp_path,
                labels_csv=tmp_path / "missing.csv",
            )
            loader._load_slide_records()

    def test_extract_patches_returns_list(self, smear_fixture_dir):
        """With a real (tiny) smear image, extraction should return a list (possibly empty)."""
        from src.data.loaders.smear_roi_loader import SmearROILoader, _SlideRecord
        loader = SmearROILoader(
            smear_root=smear_fixture_dir,
            labels_csv=smear_fixture_dir / "labels.csv",
            apply_stain_norm=False,
        )
        record = _SlideRecord(path=smear_fixture_dir / "slide_000.png", label=0)
        patches = loader._extract_patches_from_smear(record)
        assert isinstance(patches, list)
