"""End-to-end training pipeline for Project M.A.L.L.I."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator

import tensorflow as tf

from data.data_loader import MalariaDataset
from models.model_factory import build_mobilenetv3_small, compile_binary_model


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "dataset_root": "data/nih_malaria",
        "zip_path": "archive.zip",
        "image_size": [224, 224],
        "val_split": 0.2,
        "batch_size": 64,
        "seed": 42,
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
        "best_model_name": "best_mobilenetv3_small.keras",
        "last_model_name": "last_mobilenetv3_small.keras",
        "tflite_name": "mobilenetv3_small_int8.tflite",
    },
    "export": {
        "enabled": True,
        "representative_batches": 100,
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def representative_dataset_generator(
    dataset: tf.data.Dataset,
    batches: int,
) -> Iterator[list[tf.Tensor]]:
    """Yield samples for post-training integer quantization calibration."""
    for images, _ in dataset.take(batches):
        for i in range(images.shape[0]):
            sample = tf.expand_dims(images[i], axis=0)
            yield [sample]


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

    tf.random.set_seed(config["data"]["seed"])

    dataset = MalariaDataset(
        dataset_root=config["data"]["dataset_root"],
        image_size=tuple(config["data"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        val_split=config["data"]["val_split"],
        seed=config["data"]["seed"],
        zip_path=config["data"]["zip_path"],
    )
    train_ds, val_ds = dataset.create_datasets()

    model = build_mobilenetv3_small(
        input_shape=(
            config["data"]["image_size"][0],
            config["data"]["image_size"][1],
            3,
        ),
        dropout_rate=config["model"]["dropout_rate"],
        train_backbone=config["model"]["train_backbone"],
    )
    model = compile_binary_model(model, learning_rate=config["train"]["learning_rate"])

    best_model_path = models_dir / config["paths"]["best_model_name"]
    last_model_path = models_dir / config["paths"]["last_model_name"]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_f1_score",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=config["train"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_recall",
            mode="max",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / "metrics.csv")),
    ]

    logging.info("Starting training")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["train"]["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
    logging.info("Training finished after %d epochs", len(history.history["loss"]))

    val_results = model.evaluate(val_ds, verbose=0, return_dict=True)
    logging.info("Validation results: %s", val_results)

    model.save(last_model_path)
    logging.info("Saved final checkpoint to %s", last_model_path)

    if config["export"]["enabled"]:
        export_int8_tflite(
            model=model,
            train_ds=train_ds,
            output_path=models_dir / config["paths"]["tflite_name"],
            representative_batches=config["export"]["representative_batches"],
        )


if __name__ == "__main__":
    main()
