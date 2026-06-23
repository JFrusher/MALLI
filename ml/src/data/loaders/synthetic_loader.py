"""Data loading utilities for the generated synthetic field-ready dataset."""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import tensorflow as tf

from .nih_loader import AUTOTUNE, medical_augmentation
from ..preprocessing import tf_normalize_stain


@dataclass(frozen=True)
class SyntheticSample:
    """A single generated synthetic image and its labels."""

    path: Path
    soft_label: float
    hard_label: int


class SyntheticFieldReadyDataset:
    """Load the generated synthetic dataset and its labels.csv metadata."""

    def __init__(
        self,
        dataset_root: str | Path,
        labels_csv: str | Path = "labels.csv",
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        test_split: float = 0.2,
        seed: int = 42,
        augment_training: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.labels_csv = Path(labels_csv)
        if not self.labels_csv.is_absolute():
            self.labels_csv = self.dataset_root / self.labels_csv
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_split = test_split
        self.seed = seed
        self.augment_training = augment_training

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Build train and validation datasets from the generated synthetic set."""

        samples = self._collect_samples()
        train_samples, test_samples = self._stratified_split(samples)

        train_ds = self._build_dataset(train_samples, training=True, use_soft_labels=True)
        test_ds = self._build_dataset(test_samples, training=False, use_soft_labels=False)

        logging.info("Synthetic samples: train=%d | validation=%d", len(train_samples), len(test_samples))
        return train_ds, test_ds

    def _collect_samples(self) -> List[SyntheticSample]:
        """Read rows from labels.csv and resolve image paths."""

        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Synthetic labels CSV not found: {self.labels_csv}")

        samples: List[SyntheticSample] = []
        with self.labels_csv.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_fields = {"filename", "soft_label", "hard_label"}
            missing = required_fields.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Synthetic labels CSV is missing required columns: {sorted(missing)}")

            for row in reader:
                filename = row["filename"].strip()
                if not filename:
                    continue
                image_path = self.labels_csv.parent / filename
                soft_label = float(row["soft_label"])
                hard_label = int(row.get("hard_label", int(soft_label >= 0.5)))
                samples.append(SyntheticSample(path=image_path, soft_label=soft_label, hard_label=hard_label))

        if not samples:
            raise ValueError(f"No synthetic samples found in {self.labels_csv}")

        return samples

    def _stratified_split(
        self,
        samples: Sequence[SyntheticSample],
    ) -> Tuple[List[SyntheticSample], List[SyntheticSample]]:
        """Split by hard labels so validation metrics remain stable."""

        rng = random.Random(self.seed)
        by_label: dict[int, List[SyntheticSample]] = {0: [], 1: []}
        for sample in samples:
            by_label[int(sample.hard_label)].append(sample)

        train_samples: List[SyntheticSample] = []
        test_samples: List[SyntheticSample] = []
        for label_samples in by_label.values():
            rng.shuffle(label_samples)
            split_idx = int((1.0 - self.test_split) * len(label_samples))
            train_samples.extend(label_samples[:split_idx])
            test_samples.extend(label_samples[split_idx:])

        rng.shuffle(train_samples)
        rng.shuffle(test_samples)
        return train_samples, test_samples

    def _build_dataset(
        self,
        samples: Sequence[SyntheticSample],
        training: bool,
        use_soft_labels: bool,
    ) -> tf.data.Dataset:
        """Create the tf.data pipeline for the synthetic dataset."""

        file_paths = [str(sample.path) for sample in samples]
        if use_soft_labels:
            labels = [sample.soft_label for sample in samples]
        else:
            labels = [sample.hard_label for sample in samples]

        # Provide sample weights during training to compensate for class imbalance.
        if training:
            total = len(samples)
            counts = {0: 0, 1: 0}
            for s in samples:
                counts[int(s.hard_label)] += 1
            counts = {k: max(1, v) for k, v in counts.items()}
            weights = [float(total) / (2.0 * counts[int(s.hard_label)]) for s in samples]
            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels, weights))
        else:
            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        if training:
            buffer_size = max(1024, len(file_paths))
            ds = ds.shuffle(buffer_size=buffer_size, seed=None, reshuffle_each_iteration=True)

            def _map_with_weight(path, label, weight):
                image, lbl = self._load_and_preprocess(path, label, training)
                return image, lbl, weight

            ds = ds.map(_map_with_weight, num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(
                lambda path, label: self._load_and_preprocess(path, label, training),
                num_parallel_calls=AUTOTUNE,
            )

        ds = ds.batch(self.batch_size).prefetch(AUTOTUNE)
        return ds

    def _load_and_preprocess(
        self,
        image_path: tf.Tensor,
        label: tf.Tensor,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Decode an image and apply the same preprocessing used for NIH data."""

        image_bytes = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)

        image = tf_normalize_stain(image)
        if training and self.augment_training:
            image = medical_augmentation(image)

        label = tf.cast(label, tf.float32)
        return image, label
