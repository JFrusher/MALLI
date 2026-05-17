"""Evaluate a TFLite model on NIH and synthetic test sets.

Produces classification reports and confusion matrices comparable to the Keras
`evaluate_weights` flow but using the TFLite interpreter for on-device parity.

Usage:
    python models/diagnose_tflite.py --tflite experiments/logs/checkpoints/mobilenetv3_small_int8.tflite \
        --data-root nih_data --synthetic-root datasets/synthetic_field_ready --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data.data_loader import MalariaDataset
from data.synthetic_data_loader import SyntheticFieldReadyDataset


def _load_tflite_interpreter(tflite_path: Path) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    return interp


def _run_inference_on_batch(interp: tf.lite.Interpreter, batch_images: np.ndarray) -> np.ndarray:
    # Ensure input dtype and shape
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Assume single input
    inp = input_details[0]
    out = output_details[0]

    # Convert float32 input images [0,1] to the interpreter's expected type
    if inp["dtype"] == np.int8 or inp["dtype"] == np.uint8:
        # Quantized model expects integer inputs
        scale, zero_point = inp.get("quantization", (1.0, 0))
        if isinstance(scale, np.ndarray):
            scale = float(scale.squeeze())
        if isinstance(zero_point, np.ndarray):
            zero_point = int(zero_point.squeeze())
        int_input = np.round(batch_images / scale + zero_point).astype(inp["dtype"])
        test_input = int_input
    else:
        test_input = batch_images.astype(inp["dtype"])

    preds = []
    batch_size = batch_images.shape[0]
    # Single-batch inference (we already pass the batch)
    interp.set_tensor(inp["index"], test_input)
    interp.invoke()
    raw_out = interp.get_tensor(out["index"])  # shape (N, 1) or (N,)

    # If output is quantized, dequantize
    if out.get("dtype") in (np.int8, np.uint8):
        o_scale, o_zero = out.get("quantization", (1.0, 0))
        if isinstance(o_scale, np.ndarray):
            o_scale = float(o_scale.squeeze())
        if isinstance(o_zero, np.ndarray):
            o_zero = int(o_zero.squeeze())
        raw_out = (raw_out.astype(np.float32) - o_zero) * o_scale

    # Ensure flatten
    raw_out = raw_out.reshape(-1)
    return raw_out


def _evaluate_tflite_on_dataset(interp: tf.lite.Interpreter, dataset: tf.data.Dataset, dataset_name: str) -> dict[str, Any]:
    y_true = []
    y_pred = []
    y_prob = []

    for batch in dataset:
        images, labels = batch
        # images are float32 in [0,1]; interpreter expects quantized input
        batch_images = images.numpy()
        probs = _run_inference_on_batch(interp, batch_images)
        preds = (probs >= 0.5).astype(int)

        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    report = classification_report(y_true, y_pred, target_names=["Uninfected", "Parasitized"], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {dataset_name.upper()} TFLITE EVALUATION ===")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)

    return {"report": report, "confusion_matrix": cm.tolist(), "probabilities": y_prob}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", required=True, help="Path to TFLite model file")
    parser.add_argument("--data-root", default="nih_data", help="Path to NIH data root")
    parser.add_argument("--synthetic-root", default="datasets/synthetic_field_ready", help="Path to synthetic dataset root")
    parser.add_argument("--synthetic-labels", default="labels.csv", help="Synthetic labels CSV name (relative to synthetic root)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--zip-path", default=None)
    args = parser.parse_args()

    tflite_path = Path(args.tflite)
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    interp = _load_tflite_interpreter(tflite_path)

    # Build the same test splits the Keras evaluation uses
    nih = MalariaDataset(
        dataset_root=args.data_root,
        image_size=(224, 224),
        batch_size=args.batch_size,
        test_split=0.2,
        seed=42,
        zip_path=args.zip_path,
        cache_dir=None,
        use_tfrecord_cache=False,
        extract_zip=False,
    )
    _, nih_test_ds = nih.create_datasets()

    synth = SyntheticFieldReadyDataset(
        dataset_root=args.synthetic_root,
        labels_csv=args.synthetic_labels,
        image_size=(224, 224),
        batch_size=args.batch_size,
        test_split=0.2,
        seed=42,
        augment_training=False,
    )
    _, synth_test_ds = synth.create_datasets()

    nih_results = _evaluate_tflite_on_dataset(interp, nih_test_ds, "NIH")
    synth_results = _evaluate_tflite_on_dataset(interp, synth_test_ds, "SYNTHETIC")

    out = {
        "tflite": str(tflite_path),
        "nih": nih_results,
        "synthetic": synth_results,
    }

    out_path = tflite_path.parent / (tflite_path.stem + ".diagnose.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved diagnostic JSON to: {out_path}")


if __name__ == "__main__":
    main()
