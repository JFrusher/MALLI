"""Blood smear ROI patch loader for Phase C training.

Runs the existing ROI extraction pipeline on whole blood smear images to produce
labeled cell-patch training examples. Each patch is a 224×224 crop centred on a
detected cell region.

Label assignment:
  - Primary: slide-level CSV (``smear_root/labels.csv`` or custom path).
    CSV columns: ``filename`` (image filename), ``label`` (0=uninfected, 1=parasitized).
    If a per-ROI CSV with ``roi_x``, ``roi_y`` columns is provided, those
    per-patch labels are used directly.
  - The label is applied to every ROI extracted from the corresponding slide.

TFRecord caching:
  Extracted patches are cached as compressed TFRecord shards per slide so that
  the costly segmentation step only runs once. Use ``rebuild_cache=True`` to force
  a fresh extraction.

Usage::

    loader = SmearROILoader(
        smear_root="datasets/blood_smear",
        labels_csv="datasets/blood_smear/labels.csv",
        batch_size=32,
    )
    train_ds, val_ds = loader.create_datasets()
"""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf

from ...models.roi_detection.roi_grabber import (
    preprocess_for_segmentation,
    segment_thick_smear_watershed,
    segment_thin_smear,
    extract_roi_proposals,
)
from ..preprocessing import normalize_stain

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.AUTOTUNE

# Supported image extensions
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# TFRecord schema version — bump this if the patch format changes
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _SlideRecord:
    path: Path
    label: int  # 0 = uninfected, 1 = parasitized


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class SmearROILoader:
    """Extract and cache ROI patches from whole blood smear images.

    Args:
        smear_root: Directory containing blood smear images.
        labels_csv: Path to CSV with columns ``filename`` and ``label``.
            Defaults to ``smear_root/labels.csv``.
        roi_size: Side length (pixels) of the square crop around each detected
            cell centroid before resizing to ``model_input_size``.
        model_input_size: Target size fed to MobileNetV3.
        image_size: Alias for model_input_size (kept for API consistency).
        batch_size: Dataset batch size.
        test_split: Fraction of slides reserved for validation.
        seed: Random seed for slide splitting.
        cache_dir: Where to write TFRecord shards. Defaults to
            ``smear_root/.cache/roi_patches/``.
        rebuild_cache: If True, ignore existing cache and re-extract.
        smear_type: ``"auto"``, ``"thick"``, or ``"thin"``.
        apply_stain_norm: Apply Macenko stain normalization to each patch.
        max_workers: Thread pool size for parallel slide extraction.
    """

    def __init__(
        self,
        smear_root: str | Path,
        labels_csv: str | Path | None = None,
        roi_size: int = 128,
        model_input_size: Tuple[int, int] = (224, 224),
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        test_split: float = 0.2,
        seed: int = 42,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        smear_type: str = "auto",
        apply_stain_norm: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.smear_root = Path(smear_root)
        self.labels_csv = Path(labels_csv) if labels_csv else self.smear_root / "labels.csv"
        self.roi_size = roi_size
        self.model_input_size = model_input_size or image_size
        self.batch_size = batch_size
        self.test_split = test_split
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else self.smear_root / ".cache" / "roi_patches"
        self.rebuild_cache = rebuild_cache
        self.smear_type = smear_type
        self.apply_stain_norm = apply_stain_norm
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Return ``(train_ds, val_ds)`` of stain-normalized 224×224 cell patches."""
        slides = self._load_slide_records()
        if not slides:
            raise RuntimeError(
                f"No labeled smear images found under '{self.smear_root}'. "
                "Check that the directory exists and labels.csv is present."
            )

        train_slides, val_slides = self._stratified_split(slides)
        logger.info(
            "SmearROILoader: %d train slides, %d val slides",
            len(train_slides),
            len(val_slides),
        )

        # Build / load TFRecord cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._maybe_build_cache(slides)

        train_ds = self._build_dataset(train_slides, training=True)
        val_ds = self._build_dataset(val_slides, training=False)
        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def _cache_is_ready(self, slides: Sequence[_SlideRecord]) -> bool:
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema_version") != _SCHEMA_VERSION:
                return False
            if manifest.get("num_slides") != len(slides):
                return False
            return True
        except Exception:
            return False

    def _maybe_build_cache(self, slides: Sequence[_SlideRecord]) -> None:
        if not self.rebuild_cache and self._cache_is_ready(slides):
            logger.info("SmearROILoader: TFRecord cache is ready at %s", self.cache_dir)
            return

        logger.info(
            "SmearROILoader: building TFRecord cache for %d slides (this may take a while)…",
            len(slides),
        )
        total_patches = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._extract_and_write_slide, slide): slide
                for slide in slides
            }
            for future in as_completed(futures):
                slide = futures[future]
                try:
                    n = future.result()
                    total_patches += n
                    logger.debug("  %s → %d patches", slide.path.name, n)
                except Exception as exc:
                    logger.warning("  Skipping %s: %s", slide.path.name, exc)

        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "num_slides": len(slides),
            "total_patches": total_patches,
            "roi_size": self.roi_size,
            "model_input_size": list(self.model_input_size),
        }
        self._manifest_path().write_text(json.dumps(manifest, indent=2))
        logger.info("SmearROILoader: cached %d patches total", total_patches)

    def _extract_and_write_slide(self, slide: _SlideRecord) -> int:
        """Extract ROI patches from one slide, write as a TFRecord shard. Returns patch count."""
        shard_path = self.cache_dir / f"{slide.path.stem}.tfrecord.gz"
        if not self.rebuild_cache and shard_path.exists():
            # Count records quickly
            count = sum(1 for _ in tf.data.TFRecordDataset([str(shard_path)], compression_type="GZIP"))
            return count

        patches = self._extract_patches_from_smear(slide)
        if not patches:
            return 0

        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(str(shard_path), options=options) as writer:
            for image_rgb_224, label in patches:
                success, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb_224, cv2.COLOR_RGB2BGR))
                if not success:
                    continue
                feature = {
                    "image/encoded": _bytes_feature(buf.tobytes()),
                    "label": _int64_feature(label),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        return len(patches)

    # ------------------------------------------------------------------
    # ROI extraction
    # ------------------------------------------------------------------

    def _extract_patches_from_smear(
        self,
        slide: _SlideRecord,
    ) -> list[tuple[np.ndarray, int]]:
        """Run ROI pipeline on one smear image. Returns list of (crop_224_rgb, label)."""
        image_bgr = cv2.imread(str(slide.path))
        if image_bgr is None:
            raise ValueError(f"Could not read image: {slide.path}")

        # Determine smear type
        h, w = image_bgr.shape[:2]
        if self.smear_type == "auto":
            is_thick = min(h, w) >= 512
        else:
            is_thick = self.smear_type == "thick"

        # Preprocess and segment
        preprocessed = preprocess_for_segmentation(image_bgr)
        if is_thick:
            mask = segment_thick_smear_watershed(image_bgr, preprocessed)
        else:
            mask = segment_thin_smear(preprocessed)

        # Extract ROI proposals
        proposals = extract_roi_proposals(
            mask,
            image_bgr.shape,
            roi_size=self.roi_size,
        )

        patches: list[tuple[np.ndarray, int]] = []
        for prop in proposals:
            x1, y1, x2, y2 = prop.box
            crop_bgr = image_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            # Resize to model input size
            crop_bgr = cv2.resize(
                crop_bgr,
                (self.model_input_size[1], self.model_input_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            # Optional stain normalization
            if self.apply_stain_norm:
                crop_rgb = normalize_stain(crop_rgb)

            patches.append((crop_rgb, slide.label))

        return patches

    # ------------------------------------------------------------------
    # tf.data pipeline
    # ------------------------------------------------------------------

    def _build_dataset(
        self,
        slides: Sequence[_SlideRecord],
        training: bool,
    ) -> tf.data.Dataset:
        """Build a tf.data.Dataset from cached TFRecord shards for these slides."""
        shard_paths = [
            str(self.cache_dir / f"{slide.path.stem}.tfrecord.gz")
            for slide in slides
            if (self.cache_dir / f"{slide.path.stem}.tfrecord.gz").exists()
        ]
        if not shard_paths:
            raise RuntimeError(
                "No TFRecord shards found. Run create_datasets() first to build the cache."
            )

        ds = tf.data.TFRecordDataset(shard_paths, compression_type="GZIP", num_parallel_reads=AUTOTUNE)

        if training:
            ds = ds.shuffle(buffer_size=2000, seed=self.seed)

        ds = ds.map(self._parse_record, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=training)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def _parse_record(self, serialized: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        features = tf.io.parse_single_example(
            serialized,
            features={
                "image/encoded": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
            },
        )
        image = tf.image.decode_png(features["image/encoded"], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.model_input_size)
        label = tf.cast(features["label"], tf.float32)
        return image, label

    # ------------------------------------------------------------------
    # Slide discovery and splitting
    # ------------------------------------------------------------------

    def _load_slide_records(self) -> list[_SlideRecord]:
        """Read labels.csv and return _SlideRecord for each image found on disk."""
        if not self.labels_csv.exists():
            raise FileNotFoundError(
                f"Labels CSV not found: {self.labels_csv}. "
                "Expected columns: 'filename' and 'label'."
            )

        label_map: dict[str, int] = {}
        with self.labels_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "").strip()
                raw_label = row.get("label", "").strip()
                if not fname or raw_label == "":
                    continue
                try:
                    label_map[fname] = int(float(raw_label))
                except ValueError:
                    logger.warning("Skipping malformed CSV row: %s", row)

        records: list[_SlideRecord] = []
        for fname, label in label_map.items():
            candidate = self.smear_root / fname
            if not candidate.exists():
                # Try finding by stem in case of extension mismatch
                matches = list(self.smear_root.glob(f"{Path(fname).stem}.*"))
                matches = [m for m in matches if m.suffix.lower() in _IMG_EXTS]
                if not matches:
                    logger.warning("Image not found on disk: %s", candidate)
                    continue
                candidate = matches[0]

            if candidate.suffix.lower() not in _IMG_EXTS:
                continue

            records.append(_SlideRecord(path=candidate, label=label))

        logger.info("SmearROILoader: found %d slides with labels", len(records))
        return records

    def _stratified_split(
        self,
        slides: Sequence[_SlideRecord],
    ) -> Tuple[list[_SlideRecord], list[_SlideRecord]]:
        """Split slides into train/val preserving class ratios."""
        rng = np.random.default_rng(self.seed)
        pos = [s for s in slides if s.label == 1]
        neg = [s for s in slides if s.label == 0]

        def _split_class(items: list[_SlideRecord]) -> Tuple[list, list]:
            indices = rng.permutation(len(items)).tolist()
            n_val = max(1, int(len(items) * self.test_split))
            val_idx = set(indices[:n_val])
            val = [items[i] for i in range(len(items)) if i in val_idx]
            train = [items[i] for i in range(len(items)) if i not in val_idx]
            return train, val

        train_pos, val_pos = _split_class(pos)
        train_neg, val_neg = _split_class(neg)
        return list(train_pos + train_neg), list(val_pos + val_neg)
