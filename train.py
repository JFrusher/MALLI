"""End-to-end training pipeline for Project M.A.L.L.I."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from data.data_loader import MalariaDataset
from data.synthetic_data_loader import SyntheticFieldReadyDataset
from models.model_factory import build_mobilenetv3_small, compile_binary_model
from utils.dashboard import LiveDashboardCallback, launch_tensorboard


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "dataset_root": "nih_data",
        "zip_path": "dataverse_files.zip",
        "cache_dir": "datasets/blood_smear/processed/tfrecord",
        "use_tfrecord_cache": True,
        "tfrecord_shards": 16,
        "rebuild_cache": False,
        "image_size": [224, 224],
        "test_split": 0.2,
        "batch_size": 16,
        "seed": 42,
        "synthetic_dataset_root": "synthetic_field_ready",
        "synthetic_labels_csv": "labels.csv",
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
        "models_dir": "models",
        "logs_dir": "logs",
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
            "epochs": 1,
            "learning_rate": 1e-3,
            "train_backbone": False,
            "early_stopping_patience": 1,
        },
        {
            "name": "stage_2_nih_refine",
            "dataset": "nih",
            "epochs": 1,
            "learning_rate": 2e-4,
            "train_backbone": True,
            "early_stopping_patience": 1,
        },
        {
            "name": "stage_3_synth_warmup",
            "dataset": "synthetic",
            "epochs": 6,
            "learning_rate": 1e-4,
            "train_backbone": False,
            "early_stopping_patience": 2,
        },
        {
            "name": "stage_4_synth_refine",
            "dataset": "synthetic",
            "epochs": 6,
            "learning_rate": 5e-5,
            "train_backbone": True,
            "early_stopping_patience": 2,
        },
        {
            "name": "stage_5_synth_polish",
            "dataset": "synthetic",
            "epochs": 6,
            "learning_rate": 1e-5,
            "train_backbone": True,
            "early_stopping_patience": 3,
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train M.A.L.L.I. malaria classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file path to override defaults.",
    )
    parser.add_argument(
        "--launch-dashboard",
        action="store_true",
        help="Launch TensorBoard automatically while training is running.",
    )
    return parser.parse_args()


def load_config(config_path: str | None) -> Dict[str, Any]:
    """Load user config from JSON and merge it with defaults."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path is None:
        return config

    with Path(config_path).open("r", encoding="utf-8") as file:
        user_config = json.load(file)
    return deep_update(config, user_config)


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


def build_dataset_registry(config: Dict[str, Any]) -> Dict[str, tuple[tf.data.Dataset, tf.data.Dataset]]:
    """Build the NIH and synthetic dataset splits once for the staged run."""

    nih_dataset = MalariaDataset(
        dataset_root=config["data"]["dataset_root"],
        image_size=tuple(config["data"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        test_split=config["data"]["test_split"],
        seed=config["data"]["seed"],
        zip_path=config["data"]["zip_path"],
        cache_dir=config["data"]["cache_dir"],
        use_tfrecord_cache=config["data"].get("use_tfrecord_cache", True),
        tfrecord_shards=config["data"].get("tfrecord_shards", 16),
        rebuild_cache=config["data"].get("rebuild_cache", False),
        extract_zip=False,
    )
    nih_train_ds, nih_val_ds = nih_dataset.create_datasets()

    synthetic_dataset = SyntheticFieldReadyDataset(
        dataset_root=config["data"]["synthetic_dataset_root"],
        labels_csv=config["data"]["synthetic_labels_csv"],
        image_size=tuple(config["data"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        test_split=config["data"]["test_split"],
        seed=config["data"]["seed"],
        augment_training=True,
    )
    synthetic_train_ds, synthetic_val_ds = synthetic_dataset.create_datasets()

    return {
        "nih": (nih_train_ds, nih_val_ds),
        "synthetic": (synthetic_train_ds, synthetic_val_ds),
    }


def run_training_stage(
    stage: Dict[str, Any],
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    models_dir: Path,
    stage_log_dir: Path,
    logs_dir: Path,
    dashboard_config: Dict[str, Any],
    export_monitor: str = "val_auc",
) -> tf.keras.Model:
    """Train one stage and return the best-restored model."""

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
            monitor=export_monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=export_monitor,
            mode="max",
            patience=stage["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=export_monitor,
            mode="max",
            factor=0.2,
            patience=max(1, stage["early_stopping_patience"] - 1),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / f"metrics_{stage_name}.csv")),
    ]

    if stage.get("dashboard", True):
        callbacks.append(
            LiveDashboardCallback(
                log_dir=stage_log_dir,
                validation_ds=val_ds,
                prediction_threshold=dashboard_config.get("prediction_threshold", 0.5),
                val_monitor_batches=dashboard_config.get("val_monitor_batches", 20),
            )
        )

    logging.info(
        "Starting %s | dataset=%s | epochs=%d | lr=%s | train_backbone=%s",
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
    logging.info("Finished %s after %d epoch(s)", stage_name, len(history.history["loss"]))
    return model


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def calibrate_decision_threshold(
    model: tf.keras.Model,
    validation_ds: tf.data.Dataset,
    output_path: Path,
    target_recall: float = 0.95,
) -> dict[str, float]:
    """Pick a deployment threshold that prioritizes recall using the synthetic holdout."""

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

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        balanced_accuracy = (recall + specificity) / 2.0
        beta_sq = 4.0
        f2_score = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + tf.keras.backend.epsilon())

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

    output_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    logging.info("Calibrated decision threshold saved to %s: %s", output_path, best)
    return best


def export_int8_tflite(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    output_path: Path,
    representative_batches: int,
) -> None:
    """Export a fully-quantized INT8 TFLite model for mobile inference."""
    logging.info("Exporting INT8 TFLite model to %s", output_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_generator(
        train_ds,
        representative_batches,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    logging.info("Saved quantized model (%d bytes)", output_path.stat().st_size)


def main() -> None:
    """Run training, validation, checkpointing, and optional mobile export."""
    args = parse_args()
    config = load_config(args.config)

    models_dir = Path(config["paths"]["models_dir"])
    logs_dir = Path(config["paths"]["logs_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(logs_dir)

    tensorboard_run_dir = (
        logs_dir
        / "tensorboard"
        / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_run_dir.mkdir(parents=True, exist_ok=True)

    if config["dashboard"]["launch_tensorboard"] or args.launch_dashboard:
        launch_tensorboard(
            log_dir=logs_dir / "tensorboard",
            port=config["dashboard"]["port"],
        )

    tf.random.set_seed(config["data"]["seed"])

    datasets = build_dataset_registry(config)

    best_model_path = models_dir / config["paths"]["best_model_name"]
    last_model_path = models_dir / config["paths"]["last_model_name"]

    model: tf.keras.Model | None = None
    previous_weights: list[np.ndarray] | None = None

    logging.info("Starting staged training")
    logging.info("TensorBoard root directory: %s", tensorboard_run_dir)
    logging.info(
        "Open live dashboard with: tensorboard --logdir %s --port %d",
        logs_dir / "tensorboard",
        config["dashboard"]["port"],
    )

    for stage in config["stages"]:
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
        raise RuntimeError("No training stages were executed.")

    nih_results = model.evaluate(datasets["nih"][1], verbose=0, return_dict=True)
    synthetic_results = model.evaluate(datasets["synthetic"][1], verbose=0, return_dict=True)
    logging.info("NIH holdout results: %s", nih_results)
    logging.info("Synthetic holdout results: %s", synthetic_results)

    calibrate_decision_threshold(
        model=model,
        validation_ds=datasets["synthetic"][1],
        output_path=models_dir / "decision_threshold.json",
    )

    model.save(last_model_path)
    logging.info("Saved final checkpoint to %s", last_model_path)

    if config["export"]["enabled"]:
        export_int8_tflite(
            model=model,
            train_ds=datasets["synthetic"][0],
            output_path=models_dir / config["paths"]["tflite_name"],
            representative_batches=config["export"]["representative_batches"],
        )


if __name__ == "__main__":
    main()
