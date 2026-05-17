r"""Copy a temporary SavedModel directory into the experiments folder and convert to INT8 TFLite.

This script is intended to be run with the project's Python environment (e.g. .venv-diag).

Example:
  .venv-diag\Scripts\python.exe scripts\save_and_convert_current_savedmodel.py \
      --src C:\Users\hp\AppData\Local\Temp\tmpftl1pzun \
      --output-root experiments/logs/checkpoints \
      --representative-batches 100
"""
from __future__ import annotations

import sys
import os
import argparse
import shutil
from pathlib import Path
import datetime
import logging

import tensorflow as tf

# Ensure repository root is on sys.path so top-level packages (data/, models/) import correctly
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from data.data_loader import MalariaDataset


def copy_saved_model(src: Path, dest_root: Path) -> Path:
    dest = dest_root / "saved_from_temp" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    # Copy tree contents into dest (preserve inner structure)
    try:
        shutil.copytree(src, dest, dirs_exist_ok=True)
    except Exception as e:
        # fallback: copy individual files
        for p in src.rglob("*"):
            rel = p.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if p.is_dir():
                continue
            shutil.copy2(p, target)
    return dest


def representative_dataset(train_root: str, image_size=(224, 224), batch_size=64, batches=100):
    ds = MalariaDataset(dataset_root=train_root, image_size=image_size, batch_size=batch_size, test_split=0.2, seed=42)
    train_ds, _ = ds.create_datasets()

    def gen():
        for batch in train_ds.take(batches):
            images, _ = batch
            for i in range(images.shape[0]):
                yield [tf.expand_dims(images[i], axis=0).numpy().astype('float32')]

    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to temporary SavedModel directory")
    parser.add_argument("--output-root", default="experiments/logs/checkpoints", help="Root for persistent models")
    parser.add_argument("--train-root", default="nih_data", help="NIH dataset root for representative data")
    parser.add_argument("--representative-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"Source SavedModel not found: {src}")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    saved_model_dir = copy_saved_model(src, out_root)
    logging.info("Persisted SavedModel to: %s", saved_model_dir)

    tflite_out = out_root / "exports" / "tflite" / f"malaria_from_saved_{saved_model_dir.name}.tflite"
    tflite_out.parent.mkdir(parents=True, exist_ok=True)

    # Prepare converter
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(args.train_root, batches=args.representative_batches)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    logging.info("Converting to INT8 TFLite: %s", tflite_out)
    tflite_model = converter.convert()
    tflite_out.write_bytes(tflite_model)
    logging.info("Wrote TFLite: %s", tflite_out)

    print(tflite_out)


if __name__ == "__main__":
    main()
