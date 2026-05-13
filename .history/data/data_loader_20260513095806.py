"""Data loading and augmentation utilities for the NIH Malaria dataset."""

from __future__ import annotations

import logging
import random
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


class MalariaDataset:
    """Create train/validation datasets for binary malaria cell classification.

    The expected class folders are:
    - Parasitized
    - Uninfected
    """

    def __init__(
        self,
        dataset_root: str | Path,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        seed: int = 42,
        zip_path: Optional[str | Path] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.zip_path = Path(zip_path) if zip_path else None

        self._class_dirs = self._prepare_dataset_layout()

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Build train and validation `tf.data.Dataset` pipelines."""
        samples = self._collect_labeled_files()
        train_samples, val_samples = self._stratified_split(samples)

        train_ds = self._build_dataset(train_samples, training=True)
        val_ds = self._build_dataset(val_samples, training=False)

        logging.info("Train samples: %d", len(train_samples))
        logging.info("Validation samples: %d", len(val_samples))
        return train_ds, val_ds

    def _prepare_dataset_layout(self) -> Tuple[Path, Path]:
        """Ensure class directories exist and extract ZIP if needed."""
        class_dirs = self._resolve_class_dirs(self.dataset_root)
        if class_dirs:
            return class_dirs

        if self.zip_path is None or not self.zip_path.exists():
            raise FileNotFoundError(
                "Dataset folders not found and ZIP path is missing or invalid."
            )

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        logging.info("Extracting dataset from %s to %s", self.zip_path, self.dataset_root)
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            archive.extractall(self.dataset_root)

        class_dirs = self._resolve_class_dirs(self.dataset_root)
        if not class_dirs:
            raise FileNotFoundError(
                "Could not find 'Parasitized' and 'Uninfected' folders after extraction."
            )
        return class_dirs

    @staticmethod
    def _resolve_class_dirs(root: Path) -> Optional[Tuple[Path, Path]]:
        """Locate dataset class folders under a few known layouts."""
        candidate_roots = [root, root / "cell_images", root / "archive" / "cell_images"]
        for candidate in candidate_roots:
            parasitized = candidate / "Parasitized"
            uninfected = candidate / "Uninfected"
            if parasitized.is_dir() and uninfected.is_dir():
                return parasitized, uninfected
        return None

    def _collect_labeled_files(self) -> List[Tuple[str, int]]:
        """Collect image file paths and integer labels for both classes."""
        parasitized_dir, uninfected_dir = self._class_dirs
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        parasitized_files = [
            str(path)
            for path in parasitized_dir.rglob("*")
            if path.suffix.lower() in exts
        ]
        uninfected_files = [
            str(path)
            for path in uninfected_dir.rglob("*")
            if path.suffix.lower() in exts
        ]

        samples = [(path, 1) for path in parasitized_files] + [
            (path, 0) for path in uninfected_files
        ]
        if not samples:
            raise ValueError("No image files were found in the dataset directories.")
        return samples

    def _stratified_split(
        self, samples: Sequence[Tuple[str, int]]
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Split samples by class distribution for stable validation metrics."""
        rng = random.Random(self.seed)

        by_label: dict[int, List[Tuple[str, int]]] = {0: [], 1: []}
        for sample in samples:
            by_label[sample[1]].append(sample)

        train_samples: List[Tuple[str, int]] = []
        val_samples: List[Tuple[str, int]] = []

        for label_samples in by_label.values():
            rng.shuffle(label_samples)
            split_idx = int((1.0 - self.val_split) * len(label_samples))
            train_samples.extend(label_samples[:split_idx])
            val_samples.extend(label_samples[split_idx:])

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        return train_samples, val_samples

    def _build_dataset(
        self,
        samples: Sequence[Tuple[str, int]],
        training: bool,
    ) -> tf.data.Dataset:
        """Create a performant TensorFlow input pipeline from sample tuples."""
        file_paths = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]

        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        if training:
            ds = ds.shuffle(buffer_size=len(file_paths), seed=self.seed)

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
        """Decode image, apply stain placeholder and augmentations."""
        image_bytes = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)

        image = stain_normalization_placeholder(image)
        if training:
            image = medical_augmentation(image)

        label = tf.cast(label, tf.float32)
        return image, label


def stain_normalization_placeholder(image: tf.Tensor) -> tf.Tensor:
    """Placeholder for stain normalization.

    Replace this with a domain-specific method (for example, Macenko or Reinhard)
    when curated stain references are available.
    """
    return image


def medical_augmentation(image: tf.Tensor) -> tf.Tensor:
    """Apply augmentation patterns that mimic field microscopy conditions."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    image = _random_gaussian_noise(image, stddev_min=0.005, stddev_max=0.03)
    image = _random_gaussian_blur(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def _random_gaussian_noise(
    image: tf.Tensor,
    stddev_min: float,
    stddev_max: float,
) -> tf.Tensor:
    """Inject Gaussian sensor noise to emulate low-light smartphone acquisition."""
    apply_noise = tf.random.uniform([]) < 0.5
    stddev = tf.random.uniform([], minval=stddev_min, maxval=stddev_max)

    def with_noise() -> tf.Tensor:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=stddev)
        return image + noise

    return tf.cond(apply_noise, with_noise, lambda: image)


def _random_gaussian_blur(image: tf.Tensor) -> tf.Tensor:
    """Apply light blur that approximates defocus from low-cost optics."""
    apply_blur = tf.random.uniform([]) < 0.35
    sigma = tf.random.uniform([], minval=0.6, maxval=1.5)

    def with_blur() -> tf.Tensor:
        kernel = _gaussian_kernel(size=5, sigma=sigma)
        kernel = tf.repeat(kernel[:, :, tf.newaxis, tf.newaxis], repeats=3, axis=2)
        image_4d = image[tf.newaxis, ...]
        blurred = tf.nn.depthwise_conv2d(
            image_4d,
            kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        return tf.squeeze(blurred, axis=0)

    return tf.cond(apply_blur, with_blur, lambda: image)


def _gaussian_kernel(size: int, sigma: tf.Tensor) -> tf.Tensor:
    """Create a normalized 2D Gaussian kernel."""
    coords = tf.range(size, dtype=tf.float32) - float(size - 1) / 2.0
    g = tf.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / tf.reduce_sum(g)
    kernel_2d = tf.tensordot(g, g, axes=0)
    kernel_2d = kernel_2d / tf.reduce_sum(kernel_2d)
    return kernel_2d
