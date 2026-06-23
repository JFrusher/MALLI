"""Tests for TFLite export and mobile asset synchronization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf


class TestExportTFLite:
    def test_output_file_is_created(self, tmp_path):
        """export_tflite_from_weights should write a .tflite file at the given path."""
        from src.models.factory import build_mobilenetv3_small
        from src.export.to_tflite import export_tflite_from_weights

        # Save a tiny set of weights
        weights_path = tmp_path / "weights.weights.h5"
        model = build_mobilenetv3_small()
        model.save_weights(str(weights_path))

        # Minimal calibration dataset: 2 batches of 1 image each
        dummy_images = np.zeros((2, 224, 224, 3), dtype=np.float32)
        dummy_labels = np.zeros((2,), dtype=np.float32)
        cal_ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(1)

        output_path = tmp_path / "subdir" / "model.tflite"

        export_tflite_from_weights(
            weights_path=weights_path,
            output_path=output_path,
            calibration_dataset=cal_ds,
            representative_batches=2,
        )

        assert output_path.exists(), "TFLite export must create the output file"
        assert output_path.stat().st_size > 0, "TFLite file must not be empty"

    def test_output_parent_dirs_created(self, tmp_path):
        """Exporter must create any missing parent directories automatically."""
        from src.models.factory import build_mobilenetv3_small
        from src.export.to_tflite import export_tflite_from_weights

        weights_path = tmp_path / "weights.weights.h5"
        model = build_mobilenetv3_small()
        model.save_weights(str(weights_path))

        dummy_images = np.zeros((1, 224, 224, 3), dtype=np.float32)
        dummy_labels = np.zeros((1,), dtype=np.float32)
        cal_ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(1)

        deep_path = tmp_path / "a" / "b" / "c" / "model.tflite"

        export_tflite_from_weights(
            weights_path=weights_path,
            output_path=deep_path,
            calibration_dataset=cal_ds,
            representative_batches=1,
        )

        assert deep_path.exists(), "Nested parent directories must be created automatically"

    def test_representative_dataset_generator_yields_float32(self):
        """representative_dataset_generator must yield float32 tensors for INT8 calibration."""
        from src.export.to_tflite import representative_dataset_generator

        dummy = np.zeros((4, 224, 224, 3), dtype=np.float32)
        labels = np.zeros((4,), dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((dummy, labels)).batch(2)

        batches = list(representative_dataset_generator(ds, batches=1))
        assert len(batches) == 2, "Should yield one item per image in the first batch"
        for item in batches:
            assert len(item) == 1
            assert item[0].dtype == np.float32


class TestMobileAssetManager:
    def test_sync_model_copies_file(self, tmp_path):
        """sync_model should copy source to the shared assets directory."""
        from src.export.mobile_assets import MobileAssetManager

        mobile_root = tmp_path / "mobile"
        mobile_root.mkdir()
        source = tmp_path / "model.tflite"
        source.write_bytes(b"fake-tflite-content")

        mgr = MobileAssetManager(mobile_root=mobile_root, app_framework="flutter")
        success = mgr.sync_model(source, platform="shared")

        assert success is True
        dest = mobile_root / "assets" / "models" / "malaria_detector.tflite"
        assert dest.exists(), "Synced file should exist in the shared assets dir"

    def test_verify_asset_sync_checksums_match(self, tmp_path):
        """verify_asset_sync should confirm matching checksums for identical files."""
        from src.export.mobile_assets import MobileAssetManager

        mobile_root = tmp_path / "mobile"
        mobile_root.mkdir()
        payload = b"fake-tflite-binary-content"

        source = tmp_path / "source.tflite"
        source.write_bytes(payload)
        dest = tmp_path / "dest.tflite"
        dest.write_bytes(payload)

        mgr = MobileAssetManager(mobile_root=mobile_root, app_framework="flutter")
        assert mgr.verify_asset_sync(source, dest) is True

    def test_verify_asset_sync_fails_on_mismatch(self, tmp_path):
        """verify_asset_sync should return False when file contents differ."""
        from src.export.mobile_assets import MobileAssetManager

        mobile_root = tmp_path / "mobile"
        mobile_root.mkdir()

        source = tmp_path / "source.tflite"
        source.write_bytes(b"content-A")
        dest = tmp_path / "dest.tflite"
        dest.write_bytes(b"content-B-different")

        mgr = MobileAssetManager(mobile_root=mobile_root, app_framework="flutter")
        assert mgr.verify_asset_sync(source, dest) is False

    def test_generate_asset_manifest_contains_checksum(self, tmp_path):
        """generate_asset_manifest should record SHA-256 and size for each model."""
        from src.export.mobile_assets import MobileAssetManager

        mobile_root = tmp_path / "mobile"
        mobile_root.mkdir()
        model_file = tmp_path / "model.tflite"
        content = b"model-binary"
        model_file.write_bytes(content)

        mgr = MobileAssetManager(mobile_root=mobile_root, app_framework="flutter")
        manifest_path = tmp_path / "manifest.json"
        mgr.generate_asset_manifest(manifest_path, {"tflite": model_file})

        manifest = json.loads(manifest_path.read_text())
        assert "tflite" in manifest["models"]
        entry = manifest["models"]["tflite"]
        assert "checksum_sha256" in entry
        expected = hashlib.sha256(content).hexdigest()
        assert entry["checksum_sha256"] == expected


class TestMobileAssetSyncPipeline:
    def test_sync_exports_copies_tflite(self, tmp_path):
        """MobileAssetSyncPipeline.sync_exports should copy tflite to shared and android."""
        from src.export.mobile_assets import MobileAssetSyncPipeline

        mobile_root = tmp_path / "mobile"
        mobile_root.mkdir()
        export_root = tmp_path / "exports"
        export_root.mkdir()

        model_path = export_root / "model.tflite"
        model_path.write_bytes(b"fake-model-data")

        pipeline = MobileAssetSyncPipeline(
            mobile_root=mobile_root,
            export_root=export_root,
            app_framework="flutter",
        )
        results = pipeline.sync_exports({"tflite": model_path}, verify=True)

        assert results["synced"].get("tflite_shared") is True
        assert results["synced"].get("tflite_android") is True
        assert results["verified"].get("tflite_shared") is True
