"""Data loading and augmentation utilities for the NIH Malaria dataset."""

from __future__ import annotations

import json
import logging
import random
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import tensorflow as tf
from ..preprocessing import tf_normalize_stain

# Defensive check: if TensorFlow is incorrectly installed (or a mismatched
# package shadowing the name), accessing tf.data may fail with a confusing
# AttributeError. Detect this early and raise a clear, actionable message.
if not hasattr(tf, "data"):
    try:
        tf_version = getattr(tf, "__version__", "<unknown>")
        tf_file = getattr(tf, "__file__", "<unknown path>")
    except Exception:
        tf_version = "<unknown>"
        tf_file = "<unknown path>"
    raise ImportError(
        "TensorFlow appears to be improperly installed or incompatible with this environment. "
        f"Detected object: tf (version={tf_version}, path={tf_file}).\n"
        "Expected a full TensorFlow package exposing `tf.data`.\n"
        "Common fixes: install a supported combination of Python + TensorFlow and numpy, "
        "for example using Python 3.11 and: `python -m pip install 'tensorflow>=2.11,<2.14' 'numpy>=1.23,<1.25' tensorboard`.\n"
        "If you have both `tensorflow` and `tensorflow-intel` installed, uninstall both and install just one.\n"
    )


AUTOTUNE = tf.data.AUTOTUNE


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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
        test_split: float = 0.2,
        seed: int = 42,
        zip_path: Optional[str | Path] = None,
        extract_zip: bool = False,
        cache_dir: Optional[str | Path] = None,
        use_tfrecord_cache: bool = True,
        tfrecord_shards: int = 16,
        rebuild_cache: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_split = test_split
        self.seed = seed
        self.zip_path = Path(zip_path) if zip_path else None
        self.extract_zip = extract_zip
        self.cache_dir = Path(cache_dir) if cache_dir else self.dataset_root / ".cache" / "tfrecord"
        self.use_tfrecord_cache = use_tfrecord_cache
        self.tfrecord_shards = max(1, int(tfrecord_shards))
        self.rebuild_cache = rebuild_cache

        self._class_dirs = self._resolve_class_dirs(self.dataset_root)

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Build train and test `tf.data.Dataset` pipelines."""
        if self.use_tfrecord_cache:
            return self._create_tfrecord_datasets()

        if self._class_dirs is None:
            self._class_dirs = self._prepare_dataset_layout()

        samples = self._collect_labeled_files()
        train_samples, test_samples = self._stratified_split(samples)

        train_ds = self._build_dataset(train_samples, training=True)
        test_ds = self._build_dataset(test_samples, training=False)

        total_samples = len(samples)
        logging.info("Train samples: %d", len(train_samples))
        logging.info("Test samples: %d", len(test_samples))
        logging.info("Total discovered samples: %d", total_samples)
        if total_samples < 25000:
            logging.warning(
                "Discovered fewer images than expected for NIH malaria (~27k). "
                "Check that `nih_data/cell_images/Parasitized` and "
                "`nih_data/cell_images/Uninfected` are complete."
            )
        return train_ds, test_ds

    def _create_tfrecord_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create or reuse cached TFRecord shards for streaming training."""

        cache_ready, metadata = self._cache_is_ready()
        if not cache_ready or self.rebuild_cache:
            self._build_tfrecord_cache()
            _, metadata = self._cache_is_ready()

        train_records = sorted((self.cache_dir / "train").glob("*.tfrecord.gz"))
        validation_records = sorted((self.cache_dir / "validation").glob("*.tfrecord.gz"))
        if not train_records or not validation_records:
            raise FileNotFoundError(
                f"TFRecord cache is incomplete under {self.cache_dir}. "
                "Rebuild the cache or disable `use_tfrecord_cache`."
            )

        train_ds = self._build_tfrecord_dataset(
            train_records,
            training=True,
            expected_samples=int(metadata.get("train_count", 0)),
        )
        validation_ds = self._build_tfrecord_dataset(
            validation_records,
            training=False,
            expected_samples=int(metadata.get("validation_count", 0)),
        )
        return train_ds, validation_ds

    def _cache_metadata_path(self) -> Path:
        return self.cache_dir / "cache_manifest.json"

    # Bump this when the TFRecord schema or preprocessing changes so stale
    # caches are automatically invalidated.
    _CACHE_SCHEMA_VERSION = 2

    def _cache_is_ready(self) -> Tuple[bool, dict]:
        metadata_path = self._cache_metadata_path()
        if not metadata_path.exists():
            return False, {}

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False, {}

        # Reject caches written by older schema versions
        if metadata.get("schema_version") != self._CACHE_SCHEMA_VERSION:
            logging.info(
                "TFRecord cache schema mismatch (found %s, expected %s) — rebuilding.",
                metadata.get("schema_version"),
                self._CACHE_SCHEMA_VERSION,
            )
            return False, {}

        train_dir = self.cache_dir / "train"
        validation_dir = self.cache_dir / "validation"
        train_files = list(train_dir.glob("*.tfrecord.gz"))
        validation_files = list(validation_dir.glob("*.tfrecord.gz"))
        return bool(metadata) and bool(train_files) and bool(validation_files), metadata

    def _build_tfrecord_cache(self) -> None:
        """Materialize a compressed TFRecord cache from folders or a ZIP."""

        samples = self._collect_source_samples()
        train_samples, validation_samples = self._stratified_split(samples)
        train_weights = self._class_weights(train_samples)
        validation_weights = self._class_weights(validation_samples)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "validation").mkdir(parents=True, exist_ok=True)

        self._write_tfrecord_shards(
            split_name="train",
            samples=train_samples,
            weights=train_weights,
        )
        self._write_tfrecord_shards(
            split_name="validation",
            samples=validation_samples,
            weights=validation_weights,
        )

        metadata = {
            "schema_version": self._CACHE_SCHEMA_VERSION,
            "source": str(self.zip_path) if self.zip_path else str(self.dataset_root),
            "source_type": "zip" if self.zip_path and self.zip_path.exists() else "folders",
            "image_size": list(self.image_size),
            "test_split": self.test_split,
            "tfrecord_shards": self.tfrecord_shards,
            "train_count": len(train_samples),
            "validation_count": len(validation_samples),
        }
        self._cache_metadata_path().write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logging.info("Built TFRecord cache at %s", self.cache_dir)

    def _class_weights(self, samples: Sequence[Tuple[str, int]]) -> dict[int, float]:
        counts = {0: 0, 1: 0}
        for _, label in samples:
            counts[int(label)] += 1

        total = max(1, len(samples))
        counts = {label: max(1, count) for label, count in counts.items()}
        return {label: float(total) / (2.0 * count) for label, count in counts.items()}

    def _write_tfrecord_shards(
        self,
        split_name: str,
        samples: Sequence[Tuple[str, int]],
        weights: dict[int, float],
    ) -> None:
        output_dir = self.cache_dir / split_name
        shard_count = min(self.tfrecord_shards, max(1, len(samples)))
        shard_size = max(1, (len(samples) + shard_count - 1) // shard_count)
        options = tf.io.TFRecordOptions(compression_type="GZIP")

        for shard_index in range(shard_count):
            start = shard_index * shard_size
            end = min(len(samples), start + shard_size)
            if start >= end:
                break

            shard_path = output_dir / f"part-{shard_index:05d}-of-{shard_count:05d}.tfrecord.gz"
            with tf.io.TFRecordWriter(str(shard_path), options=options) as writer:
                for source_path, label in samples[start:end]:
                    image_bytes, rel_path = self._read_source_bytes(source_path)
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "image/encoded": _bytes_feature(image_bytes),
                                "label": _int64_feature(label),
                                "weight": _float_feature(weights[int(label)]),
                                "source_path": _bytes_feature(rel_path.encode("utf-8")),
                            }
                        )
                    )
                    writer.write(example.SerializeToString())

    def _read_source_bytes(self, source_path: str) -> Tuple[bytes, str]:
        path = Path(source_path)
        if path.exists():
            try:
                rel_path = str(path.relative_to(self.dataset_root))
            except ValueError:
                rel_path = str(path)
            return path.read_bytes(), rel_path

        if self.zip_path is None or not self.zip_path.exists():
            raise FileNotFoundError(f"Could not read source image bytes for {source_path}")

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            with archive.open(source_path, "r") as file_handle:
                return file_handle.read(), source_path

    def _collect_source_samples(self) -> List[Tuple[str, int]]:
        if self._class_dirs is not None:
            return self._collect_labeled_files()

        if self.zip_path is not None and self.zip_path.exists():
            return self._collect_labeled_zip_entries()

        if self.extract_zip:
            self._class_dirs = self._prepare_dataset_layout()
            return self._collect_labeled_files()

        raise FileNotFoundError(
            f"Could not locate class folders under {self.dataset_root} and no valid ZIP was provided."
        )

    def _collect_labeled_zip_entries(self) -> List[Tuple[str, int]]:
        if self.zip_path is None:
            raise FileNotFoundError("ZIP path was not configured.")

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        samples: List[Tuple[str, int]] = []

        with zipfile.ZipFile(self.zip_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                path = Path(info.filename)
                if path.suffix.lower() not in exts:
                    continue
                label = self._infer_label_from_path(path)
                if label is None:
                    continue
                samples.append((info.filename, label))

        if not samples:
            raise ValueError(f"No image files were found in ZIP archive {self.zip_path}")

        logging.info("Collected %d image entries from ZIP archive %s", len(samples), self.zip_path)
        return samples

    @staticmethod
    def _infer_label_from_path(path: Path) -> Optional[int]:
        parts = [part.lower() for part in path.parts]
        if "parasitized" in parts:
            return 1
        if "uninfected" in parts:
            return 0
        return None

    def _build_tfrecord_dataset(
        self,
        record_files: Sequence[Path],
        training: bool,
        expected_samples: int,
    ) -> tf.data.Dataset:
        filenames = [str(path) for path in record_files]
        ds = tf.data.TFRecordDataset(
            filenames,
            compression_type="GZIP",
            num_parallel_reads=AUTOTUNE,
        )

        if training:
            buffer_size = max(1024, len(filenames) * 32)
            ds = ds.shuffle(buffer_size=buffer_size, seed=None, reshuffle_each_iteration=True)
            ds = ds.map(self._parse_training_tfrecord, num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(self._parse_validation_tfrecord, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(self.batch_size)

        if expected_samples > 0:
            expected_batches = (expected_samples + self.batch_size - 1) // self.batch_size
            ds = ds.apply(tf.data.experimental.assert_cardinality(expected_batches))

        return ds.prefetch(AUTOTUNE)

    def _parse_training_tfrecord(self, serialized_example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        features = self._parse_tfrecord_features(serialized_example)
        image = self._decode_and_preprocess(features["image/encoded"], training=True)
        label = tf.cast(features["label"], tf.float32)
        weight = tf.cast(features["weight"], tf.float32)
        return image, label, weight

    def _parse_validation_tfrecord(self, serialized_example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        features = self._parse_tfrecord_features(serialized_example)
        image = self._decode_and_preprocess(features["image/encoded"], training=False)
        label = tf.cast(features["label"], tf.float32)
        return image, label

    @staticmethod
    def _parse_tfrecord_features(serialized_example: tf.Tensor) -> dict[str, tf.Tensor]:
        feature_description = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "weight": tf.io.FixedLenFeature([], tf.float32),
            "source_path": tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(serialized_example, feature_description)

    def _decode_and_preprocess(self, image_bytes: tf.Tensor, training: bool) -> tf.Tensor:
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)

        image = tf_normalize_stain(image)
        if training:
            image = medical_augmentation(image)
        return image

    def _prepare_dataset_layout(self) -> Tuple[Path, Path]:
        """Ensure class directories exist. Optionally extract a provided ZIP.

        By default this function assumes you have already extracted the NIH dataset
        into `dataset_root` (e.g., `nih_data/`). If the dataset folders are not
        present and `extract_zip` is True and a valid `zip_path` is provided, the
        ZIP will be extracted. Otherwise an error will be raised so the user is
        explicitly aware that the images must be present.
        """
        class_dirs = self._resolve_class_dirs(self.dataset_root)
        if class_dirs:
            logging.info(
                "Using dataset folders: Parasitized=%s | Uninfected=%s",
                class_dirs[0],
                class_dirs[1],
            )
            return class_dirs

        if not self.extract_zip:
            raise FileNotFoundError(
                f"Dataset folders not found under {self.dataset_root}. "
                "Set `extract_zip=True` and provide `zip_path` to extract automatically, "
                "or unzip the archive manually into the dataset directory (e.g., nih_data/)."
            )

        if self.zip_path is None or not self.zip_path.exists():
            raise FileNotFoundError(
                "extract_zip=True was set but ZIP path is missing or invalid."
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
        logging.info(
            "Using dataset folders after extraction: Parasitized=%s | Uninfected=%s",
            class_dirs[0],
            class_dirs[1],
        )
        return class_dirs

    @staticmethod
    def _resolve_class_dirs(root: Path) -> Optional[Tuple[Path, Path]]:
        """Locate dataset class folders under a few known layouts."""
        candidate_roots = [
            root,
            root / "nih_data",
            root / "cell_images",
            root / "nih_data" / "cell_images",
            root / "archive" / "cell_images",
        ]
        for candidate in candidate_roots:
            parasitized = candidate / "Parasitized"
            uninfected = candidate / "Uninfected"
            if parasitized.is_dir() and uninfected.is_dir():
                return parasitized, uninfected
        return None

    def _collect_labeled_files(self) -> List[Tuple[str, int]]:
        """Collect image file paths and integer labels for both classes."""
        if self._class_dirs is None:
            raise FileNotFoundError(
                f"Dataset folders not found under {self.dataset_root}. "
                "Enable TFRecord caching with a ZIP source or extract the dataset first."
            )

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
        logging.info(
            "Class counts | Parasitized=%d | Uninfected=%d",
            len(parasitized_files),
            len(uninfected_files),
        )
        if not samples:
            raise ValueError("No image files were found in the dataset directories.")
        return samples

    def _stratified_split(
        self, samples: Sequence[Tuple[str, int]]
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Split samples by class distribution for stable test metrics."""
        rng = random.Random(self.seed)

        by_label: dict[int, List[Tuple[str, int]]] = {0: [], 1: []}
        for sample in samples:
            by_label[sample[1]].append(sample)

        train_samples: List[Tuple[str, int]] = []
        test_samples: List[Tuple[str, int]] = []

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
        samples: Sequence[Tuple[str, int]],
        training: bool,
    ) -> tf.data.Dataset:
        """Create a performant TensorFlow input pipeline from sample tuples."""
        file_paths = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]

        # When training, provide per-sample weights to rebalance classes so
        # the model sees a more even contribution from each class.
        if training:
            total = len(labels)
            counts = {0: 0, 1: 0}
            for l in labels:
                counts[int(l)] += 1
            # avoid div-by-zero
            counts = {k: max(1, v) for k, v in counts.items()}
            weights = [float(total) / (2.0 * counts[int(l)]) for l in labels]
            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels, weights))
        else:
            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        if training:
            # Use a large buffer and allow reshuffling each iteration so
            # the training order changes every epoch. Avoid a fixed seed
            # here because a constant seed produces the same shuffled
            # order on each epoch which can cause class-ordering issues.
            buffer_size = max(1024, len(file_paths))
            ds = ds.shuffle(buffer_size=buffer_size, seed=None, reshuffle_each_iteration=True)

        if training:
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
        """Decode image, apply stain placeholder and augmentations."""
        image_bytes = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)

        image = tf_normalize_stain(image)
        if training:
            image = medical_augmentation(image)

        label = tf.cast(label, tf.float32)
        return image, label


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
