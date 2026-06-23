"""End-to-end training pipeline for Project M.A.L.L.I. (malaria detection).

This is the main entry point for training the binary classification model.
Supports multi-stage training with automatic export to mobile formats.

Usage:
    python train.py                                    # Use default config
    python train.py --config configs/custom.json      # Use custom config
    python train.py --stages 1 2 3 --epochs 10 20 30  # Override stages
    python train.py --export-formats tflite onnx      # Export to multiple formats
    python train.py --sync-mobile mobile/ --framework flutter  # Auto-sync to app
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from src.data.loaders.nih_loader import MalariaDataset
from src.data.loaders.synthetic_loader import SyntheticFieldReadyDataset
from src.data.loaders.smear_roi_loader import SmearROILoader
from src.models.factory import build_mobilenetv3_small, compile_binary_model
from src.utils.visualization import LiveDashboardCallback, launch_tensorboard
from src.export.pipeline import ExportPipeline
from src.export.mobile_assets import MobileAssetSyncPipeline


# Configure structured logging
def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """Configure structured logging with file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # File handler - verbose
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s:%(lineno)d | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler - warnings and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING if not verbose else logging.INFO)
    console_formatter = logging.Formatter(
        "%(levelname)-8s | %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress TensorFlow logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized. Details written to: %s", log_file)


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "dataset_root": "datasets/nih",
        "zip_path": "shared/data/dataverse_files.zip",
        "cache_dir": "datasets/blood_smear/processed/tfrecord",
        "use_tfrecord_cache": True,
        "tfrecord_shards": 16,
        "rebuild_cache": False,
        "image_size": [224, 224],
        "test_split": 0.2,
        "batch_size": 16,
        "seed": 42,
        "synthetic_dataset_root": "datasets/synthetic_field_ready",
        "synthetic_labels_csv": "labels.csv",
        "smear_dataset_root": "datasets/blood_smear",
        "smear_labels_csv": "datasets/blood_smear/labels.csv",
        "smear_type": "auto",
        "smear_apply_stain_norm": True,
    },
    "model": {
        "dropout_rate": 0.2,
        "train_backbone": False,
    },
    "train": {
        "epochs": 20,
        "learning_rate": 1e-3,
        "early_stopping_patience": 6,
    },
    "paths": {
        "models_dir": "experiments/logs/checkpoints",
        "logs_dir": "experiments/logs",
        "best_model_name": "best_mobilenetv3_small.weights.h5",
        "last_model_name": "last_mobilenetv3_small.keras",
        "tflite_name": "mobilenetv3_small_int8.tflite",
    },
    "export": {
        "enabled": True,
        "representative_batches": 100,
    },
    "stages": [
        {
            "name": "stage_1_nih_warmup",
            "dataset": "nih",
            "epochs": 5,
            "learning_rate": 1e-3,
            "train_backbone": False,
            "early_stopping_patience": 3,
            "description": "Phase A: head-only warmup on clean NIH single-cell images",
        },
        {
            "name": "stage_2_nih_finetune",
            "dataset": "nih",
            "epochs": 10,
            "learning_rate": 2e-4,
            "train_backbone": True,
            "early_stopping_patience": 5,
            "description": "Phase A: full fine-tune on NIH — unfreezes backbone",
        },
        {
            "name": "stage_3_obscure_warmup",
            "dataset": "nih",
            "epochs": 8,
            "learning_rate": 5e-5,
            "train_backbone": True,
            "early_stopping_patience": 4,
            "augmentation_curriculum": True,
            "description": "Phase B: NIH cells with ramping augmentation (0%→100% intensity)",
        },
        {
            "name": "stage_4_synthetic_field",
            "dataset": "synthetic",
            "epochs": 8,
            "learning_rate": 2e-5,
            "train_backbone": True,
            "early_stopping_patience": 4,
            "description": "Phase B: synthetic field-ready images with full augmentation",
        },
        {
            "name": "stage_5_smear_roi_adapt",
            "dataset": "smear_roi",
            "epochs": 10,
            "learning_rate": 1e-5,
            "train_backbone": False,
            "early_stopping_patience": 5,
            "description": "Phase C: head-only domain adaptation on blood smear ROI patches",
        },
        {
            "name": "stage_6_smear_roi_finetune",
            "dataset": "smear_roi",
            "epochs": 8,
            "learning_rate": 5e-6,
            "train_backbone": True,
            "early_stopping_patience": 4,
            "description": "Phase C: full fine-tune on smear ROI patches with field augmentation",
        },
        {
            "name": "stage_7_joint_consolidation",
            "dataset": "nih",
            "epochs": 5,
            "learning_rate": 1e-6,
            "train_backbone": True,
            "early_stopping_patience": 3,
            "description": "Phase C: joint NIH+smear consolidation polish",
        },
    ],
    "dashboard": {
        "enabled": True,
        "launch_tensorboard": False,
        "port": 6006,
        "update_freq": "batch",
        "histogram_freq": 0,
        "prediction_threshold": 0.5,
        "val_monitor_batches": 20,
    },
}



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        prog="M.A.L.L.I. Training Pipeline",
        description="Train binary malaria parasite detection model with multi-stage workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Train with default config
  python train.py

  # Use custom config file
  python train.py --config configs/prod.json

  # Override specific parameters
  python train.py --batch-size 32 --learning-rate 1e-4 --epochs 50

  # Export to multiple formats and sync to mobile app
  python train.py --export-formats tflite coreml \\
                  --sync-mobile ../mobile/ --framework flutter

  # Verbose logging with dashboard
  python train.py -v --dashboard --dashboard-port 6007

  # Run only specific training stages
  python train.py --stages 1 3 5
        """,
    )
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file to override defaults",
        metavar="FILE",
    )
    config_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    # Data options
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/nih",
        help="Path to NIH malaria dataset root",
        metavar="PATH",
    )
    data_group.add_argument(
        "--synthetic-root",
        type=str,
        default="datasets/synthetic_field_ready",
        help="Path to synthetic dataset root",
        metavar="PATH",
    )
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    data_group.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size H W (default: 224 224)",
        metavar=("H", "W"),
    )
    data_group.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    data_group.add_argument(
        "--cache-tfrecords",
        action="store_true",
        default=True,
        help="Cache dataset as TFRecords for faster loading",
    )
    data_group.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild TFRecord cache even if it exists",
    )
    
    # Model options
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for MobileNetV3 (default: 0.2)",
    )
    
    # Training options
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=None,
        help="Which stages to train (1-5, default: all)",
        metavar="STAGE_NUM",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Override stage epochs (must match # of stages if specified)",
        metavar="EPOCHS",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate for all stages",
        metavar="LR",
    )
    train_group.add_argument(
        "--early-stopping",
        type=int,
        default=6,
        help="Early stopping patience in epochs (default: 6)",
    )
    
    # Logging & Monitoring
    logging_group = parser.add_argument_group("Logging & Monitoring")
    logging_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging to console",
    )
    logging_group.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch TensorBoard dashboard automatically",
    )
    logging_group.add_argument(
        "--dashboard-port",
        type=int,
        default=6006,
        help="TensorBoard port (default: 6006)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="experiments/logs",
        help="Root directory for logs and checkpoints",
        metavar="PATH",
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for organizing runs (auto-generated if not set)",
        metavar="NAME",
    )
    
    # Export options
    export_group = parser.add_argument_group("Export")
    export_group.add_argument(
        "--export-formats",
        choices=["tflite", "onnx", "coreml"],
        nargs="+",
        default=["tflite"],
        help="Export model to these formats (default: tflite)",
    )
    export_group.add_argument(
        "--export-disabled",
        action="store_true",
        help="Skip model export after training",
    )
    export_group.add_argument(
        "--representative-batches",
        type=int,
        default=100,
        help="Batches for TFLite quantization calibration (default: 100)",
    )
    
    # Mobile sync options
    mobile_group = parser.add_argument_group("Mobile Sync")
    mobile_group.add_argument(
        "--sync-mobile",
        type=str,
        default=None,
        help="Path to mobile app root for auto-syncing models",
        metavar="PATH",
    )
    mobile_group.add_argument(
        "--framework",
        choices=["flutter", "react-native", "swiftui"],
        default="flutter",
        help="Mobile framework (default: flutter)",
    )
    mobile_group.add_argument(
        "--verify-sync",
        action="store_true",
        default=True,
        help="Verify synced assets via checksum",
    )
    
    return parser.parse_args()



def load_config(config_path: str | None, args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configuration from file and CLI arguments."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    
    # Load from file if provided
    if config_path:
        try:
            with Path(config_path).open("r", encoding="utf-8") as file:
                user_config = json.load(file)
            config = deep_update(config, user_config)
            logging.info("Loaded config from: %s", config_path)
        except Exception as e:
            logging.error("Failed to load config file: %s", e)
            raise
    
    # Override with CLI arguments
    if args.batch_size != 16:
        config["data"]["batch_size"] = args.batch_size
    if args.image_size != [224, 224]:
        config["data"]["image_size"] = args.image_size
    if args.test_split != 0.2:
        config["data"]["test_split"] = args.test_split
    if args.dropout != 0.2:
        config["model"]["dropout_rate"] = args.dropout
    if args.early_stopping != 6:
        config["train"]["early_stopping_patience"] = args.early_stopping
    if args.seed != 42:
        config["data"]["seed"] = args.seed
    
    # Dataset path overrides
    config["data"]["dataset_root"] = args.dataset_root
    config["data"]["synthetic_dataset_root"] = args.synthetic_root
    config["data"]["rebuild_cache"] = args.rebuild_cache
    
    # Paths
    output_dir = Path(args.output_dir)
    config["paths"]["logs_dir"] = str(output_dir)
    config["paths"]["models_dir"] = str(output_dir / "checkpoints")
    
    # Export settings
    config["export"]["enabled"] = not args.export_disabled
    config["export"]["representative_batches"] = args.representative_batches
    config["dashboard"]["launch_tensorboard"] = args.dashboard
    config["dashboard"]["port"] = args.dashboard_port

    # Apply stage selection from CLI (1-based indices)
    if args.stages is not None:
        selected = []
        for n in args.stages:
            if not (1 <= n <= len(config["stages"])):
                raise ValueError(f"Requested stage {n} is out of range (1-{len(config['stages'])})")
            selected.append(config["stages"][n - 1])
        config["stages"] = selected

    # Override epochs per-stage if requested. Accept single value or list matching stages.
    if args.epochs is not None:
        if len(args.epochs) == 1:
            for s in config["stages"]:
                s["epochs"] = args.epochs[0]
        elif len(args.epochs) == len(config["stages"]):
            for s, e in zip(config["stages"], args.epochs):
                s["epochs"] = e
        else:
            raise ValueError("When specifying --epochs with multiple values, the count must match the selected stages")
    
    return config


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def configure_logging(logs_dir: Path) -> None:
    """Initialize console + file logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "training.log"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)


def representative_dataset_generator(
    dataset: tf.data.Dataset,
    batches: int,
) -> Iterator[list[tf.Tensor]]:
    """Yield samples for post-training integer quantization calibration."""
    for batch in dataset.take(batches):
        images = batch[0]
        for i in range(images.shape[0]):
            sample = tf.expand_dims(images[i], axis=0).numpy().astype(np.float32)
            yield [sample]


def build_dataset_registry(
    config: Dict[str, Any],
    required_datasets: set[str] | None = None,
) -> Dict[str, tuple[tf.data.Dataset, tf.data.Dataset]]:
    """Build dataset splits for all keys referenced by the active stages.

    Args:
        config: Full training config dict.
        required_datasets: Set of dataset keys that will actually be used.
            If None, attempts to load all configured datasets.

    Returns:
        Dict mapping dataset key → (train_ds, val_ds).
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading datasets…")
    datasets: dict[str, tuple[tf.data.Dataset, tf.data.Dataset]] = {}

    if required_datasets is None or "nih" in required_datasets:
        nih_dataset = MalariaDataset(
            dataset_root=config["data"]["dataset_root"],
            image_size=tuple(config["data"]["image_size"]),
            batch_size=config["data"]["batch_size"],
            test_split=config["data"]["test_split"],
            seed=config["data"]["seed"],
            zip_path=config["data"].get("zip_path"),
            cache_dir=config["data"].get("cache_dir"),
            use_tfrecord_cache=config["data"].get("use_tfrecord_cache", True),
            tfrecord_shards=config["data"].get("tfrecord_shards", 16),
            rebuild_cache=config["data"].get("rebuild_cache", False),
            extract_zip=False,
        )
        nih_train_ds, nih_val_ds = nih_dataset.create_datasets()
        datasets["nih"] = (nih_train_ds, nih_val_ds)
        logger.info("✓ NIH dataset loaded")

    if required_datasets is None or "synthetic" in required_datasets:
        synthetic_root = Path(config["data"].get("synthetic_dataset_root", ""))
        labels_csv_file = config["data"].get("synthetic_labels_csv", "labels.csv")
        labels_csv_path = synthetic_root / labels_csv_file
        if synthetic_root.exists() and labels_csv_path.exists():
            synthetic_dataset = SyntheticFieldReadyDataset(
                dataset_root=str(synthetic_root),
                labels_csv=labels_csv_file,
                image_size=tuple(config["data"]["image_size"]),
                batch_size=config["data"]["batch_size"],
                test_split=config["data"]["test_split"],
                seed=config["data"]["seed"],
                augment_training=True,
            )
            synthetic_train_ds, synthetic_val_ds = synthetic_dataset.create_datasets()
            datasets["synthetic"] = (synthetic_train_ds, synthetic_val_ds)
            logger.info("✓ Synthetic field-ready dataset loaded")
        else:
            logger.warning("Synthetic dataset not found or labels CSV missing: %s", labels_csv_path)

    if required_datasets is None or "smear_roi" in required_datasets:
        smear_root = Path(config["data"].get("smear_dataset_root", "datasets/blood_smear"))
        smear_labels = Path(config["data"].get("smear_labels_csv", smear_root / "labels.csv"))
        if smear_root.exists() and smear_labels.exists():
            smear_loader = SmearROILoader(
                smear_root=smear_root,
                labels_csv=smear_labels,
                image_size=tuple(config["data"]["image_size"]),
                batch_size=config["data"]["batch_size"],
                test_split=config["data"]["test_split"],
                seed=config["data"]["seed"],
                smear_type=config["data"].get("smear_type", "auto"),
                apply_stain_norm=config["data"].get("smear_apply_stain_norm", True),
            )
            smear_train_ds, smear_val_ds = smear_loader.create_datasets()
            datasets["smear_roi"] = (smear_train_ds, smear_val_ds)
            logger.info("✓ Blood smear ROI dataset loaded")
        else:
            logger.warning(
                "Blood smear dataset not found at '%s' or labels CSV missing: %s",
                smear_root, smear_labels,
            )

    return datasets


def run_training_stage(
    stage: Dict[str, Any],
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    models_dir: Path,
    stage_log_dir: Path,
    logs_dir: Path,
    dashboard_config: Dict[str, Any],
) -> tf.keras.Model:
    """Train one stage."""
    logger = logging.getLogger(__name__)
    stage_name = stage["name"]
    stage_checkpoint_path = models_dir / f"{stage_name}.weights.h5"

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(stage_log_dir),
            histogram_freq=dashboard_config.get("histogram_freq", 0),
            write_graph=False,
            write_images=False,
            update_freq=dashboard_config.get("update_freq", "epoch"),
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(stage_checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=stage["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.2,
            patience=max(1, stage["early_stopping_patience"] - 1),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / f"metrics_{stage_name}.csv")),
        LiveDashboardCallback(
            log_dir=stage_log_dir,
            validation_ds=val_ds,
            prediction_threshold=dashboard_config.get("prediction_threshold", 0.5),
            val_monitor_batches=dashboard_config.get("val_monitor_batches", 20),
        ),
    ]

    logger.info(
        "Training %s | dataset=%s | epochs=%d | lr=%s | train_backbone=%s",
        stage_name,
        stage["dataset"],
        stage["epochs"],
        stage["learning_rate"],
        stage["train_backbone"],
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
    
    logger.info("✓ %s completed after %d epoch(s)", stage_name, len(history.history["loss"]))
    return model


def calibrate_decision_threshold(
    model: tf.keras.Model,
    validation_ds: tf.data.Dataset,
    output_path: Path,
    target_recall: float = 0.95,
) -> dict[str, float]:
    """Calibrate decision threshold with a recall-first preference."""
    logger = logging.getLogger(__name__)
    logger.info("Calibrating decision threshold...")
    
    y_true_batches: list[np.ndarray] = []
    y_prob_batches: list[np.ndarray] = []

    for images, labels in validation_ds:
        probabilities = model.predict(images, verbose=0).reshape(-1)
        y_prob_batches.append(probabilities)
        y_true_batches.append(labels.numpy().reshape(-1))

    y_true = np.concatenate(y_true_batches).astype(np.int32)
    y_prob = np.concatenate(y_prob_batches).astype(np.float32)

    candidates: list[dict[str, float]] = []

    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= threshold).astype(np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (recall + specificity) / 2.0
        
        beta_sq = 4.0
        f2_score = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-10)

        candidates.append(
            {
                "threshold": float(threshold),
                "f2_score": float(f2_score),
                "precision": float(precision),
                "recall": float(recall),
                "balanced_accuracy": float(balanced_accuracy),
            }
        )

    recall_ready = [candidate for candidate in candidates if candidate["recall"] >= target_recall]
    if recall_ready:
        best = min(
            recall_ready,
            key=lambda candidate: (
                candidate["threshold"],
                -candidate["precision"],
                -candidate["f2_score"],
            ),
        )
        selection_mode = f"lowest_threshold_meeting_recall>={target_recall:.2f}"
    else:
        best = max(
            candidates,
            key=lambda candidate: (
                candidate["recall"],
                candidate["precision"],
                -candidate["threshold"],
            ),
        )
        selection_mode = "best_available_recall"

    best = {**best, "target_recall": float(target_recall), "selection_mode": selection_mode}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    logger.info("✓ Decision threshold calibrated: %.3f (F2=%.3f)", best["threshold"], best["f2_score"])
    return best


def main() -> None:
    """Main training pipeline."""
    args = parse_args()
    
    # Initialize logging
    output_dir = Path(args.output_dir)
    setup_logging(output_dir / "logs", verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("M.A.L.L.I. Training Pipeline started")
    logger.info("=" * 80)
    
    # Load configuration
    try:
        config = load_config(args.config, args)
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        sys.exit(1)
    
    # Set random seeds
    tf.random.set_seed(config["data"]["seed"])
    np.random.seed(config["data"]["seed"])
    
    models_dir = Path(config["paths"]["models_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_run_dir = logs_dir / "tensorboard" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_run_dir.mkdir(parents=True, exist_ok=True)

    if config["dashboard"]["launch_tensorboard"]:
        launch_tensorboard(
            log_dir=logs_dir / "tensorboard",
            port=config["dashboard"]["port"],
        )
        logger.info("TensorBoard launched on port %d", config["dashboard"]["port"])

    # Determine which datasets are actually required by the selected stages
    required_datasets = {stage["dataset"] for stage in config["stages"]}

    # Load datasets
    try:
        datasets = build_dataset_registry(config, required_datasets=required_datasets)
    except Exception as e:
        logger.error("Failed to load datasets: %s", e)
        sys.exit(1)

    best_model_path = models_dir / config["paths"]["best_model_name"]
    last_model_path = models_dir / config["paths"]["last_model_name"]

    model: tf.keras.Model | None = None
    previous_weights: list[np.ndarray] | None = None

    logger.info("Starting staged training (5 stages total)")
    logger.info("TensorBoard: tensorboard --logdir %s --port %d", 
               logs_dir / "tensorboard", config["dashboard"]["port"])

    # Execute training stages
    for i, stage in enumerate(config["stages"], 1):
        if stage["dataset"] not in datasets:
            logger.error(
                "Stage %s requires dataset '%s' which is not available. Check dataset paths and --stages selection.",
                stage.get("name", i),
                stage["dataset"],
            )
            sys.exit(1)
        stage_dataset = datasets[stage["dataset"]]
        stage_train_ds, stage_val_ds = stage_dataset
        
        model = build_mobilenetv3_small(
            input_shape=(
                config["data"]["image_size"][0],
                config["data"]["image_size"][1],
                3,
            ),
            dropout_rate=config["model"]["dropout_rate"],
            train_backbone=stage["train_backbone"],
        )
        model = compile_binary_model(model, learning_rate=stage["learning_rate"])
        if previous_weights is not None:
            model.set_weights(previous_weights)

        stage_log_dir = tensorboard_run_dir / stage["name"]
        stage_log_dir.mkdir(parents=True, exist_ok=True)

        model = run_training_stage(
            stage=stage,
            model=model,
            train_ds=stage_train_ds,
            val_ds=stage_val_ds,
            models_dir=models_dir,
            stage_log_dir=stage_log_dir,
            logs_dir=logs_dir,
            dashboard_config=config["dashboard"],
        )
        previous_weights = model.get_weights()

    if model is None:
        logger.error("No training stages were executed")
        sys.exit(1)

    # Evaluate on both datasets
    logger.info("Evaluating model on test sets...")
    nih_results = model.evaluate(datasets["nih"][1], verbose=0, return_dict=True)
    synthetic_results = model.evaluate(datasets["synthetic"][1], verbose=0, return_dict=True)
    logger.info("NIH test results: %s", nih_results)
    logger.info("Synthetic test results: %s", synthetic_results)

    # Calibrate threshold
    calibrate_decision_threshold(
        model=model,
        validation_ds=datasets["synthetic"][1],
        output_path=models_dir / "decision_threshold.json",
    )

    # Save final model
    model.save(last_model_path)
    logger.info("✓ Model saved to %s", last_model_path)

    # Export pipeline
    if config["export"]["enabled"]:
        logger.info("Starting model export pipeline...")
        try:
            export_pipeline = ExportPipeline(
                export_dir=models_dir / "exports",
                formats=args.export_formats,
            )
            
            export_results = export_pipeline.export_all(
                model=model,
                train_ds=datasets["synthetic"][0],
                model_name="malaria_detector",
                representative_batches=config["export"]["representative_batches"],
            )
            
            # Save metadata
            export_pipeline.save_metadata(models_dir / "exports" / "metadata.json")
            logger.info("✓ Export completed: %d format(s)", len(export_results))
            
            # Sync to mobile app if requested
            if args.sync_mobile:
                logger.info("Syncing models to mobile app...")
                sync_pipeline = MobileAssetSyncPipeline(
                    mobile_root=Path(args.sync_mobile),
                    export_root=models_dir / "exports",
                    app_framework=args.framework,
                )
                
                models_to_sync = {
                    fmt: metadata.file_path 
                    for fmt, metadata in export_results.items()
                }
                sync_results = sync_pipeline.sync_exports(
                    {fmt: Path(p) for fmt, p in models_to_sync.items()},
                    verify=args.verify_sync,
                )
                logger.info("✓ Mobile sync completed")
                logger.info("  Synced: %d, Verified: %d",
                           sum(sync_results["synced"].values()),
                           sum(sync_results["verified"].values()))
        
        except Exception as e:
            logger.error("Export pipeline failed: %s", e, exc_info=True)

    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
