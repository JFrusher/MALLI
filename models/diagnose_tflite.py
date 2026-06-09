"""Evaluate a TFLite model on NIH and synthetic test sets.

Produces classification reports and confusion matrices comparable to the Keras
`evaluate_weights` flow but using the TFLite interpreter for on-device parity.

Usage:
    python models/diagnose_tflite.py --tflite experiments/logs/checkpoints/mobilenetv3_small_int8.tflite \
        --data-root nih_data --synthetic-root datasets/synthetic_field_ready --batch-size 64
"""
from __future__ import annotations

import os
import sys

os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

# Get the directory from which you ran the command (C:\Users\hp\OneDrive\Desktop\6 MALLI)
root_dir = os.getcwd()
if root_dir not in sys.path:
    sys.path.append(root_dir)

import numpy as np

_np_major = int(np.__version__.split('.', 1)[0])
if _np_major >= 2:
    raise RuntimeError(
        "This diagnostic script requires NumPy < 2 because the installed TensorFlow build "
        "was compiled against NumPy 1.x. Install a compatible stack, for example: "
        "pip install 'numpy<2' 'tensorflow==2.13.1'."
    )

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data.data_loader import MalariaDataset
from data.synthetic_data_loader import SyntheticFieldReadyDataset

from tensorflow.lite.python.interpreter import InterpreterWithCustomOps

def _load_tflite_interpreter(tflite_path: Path) -> tf.lite.Interpreter:
    """Create and allocate a TFLite interpreter, retrying without delegates on failure.

    Some TF builds on Windows fail when the XNNPACK delegate is selected. Try
    normal allocation first; on RuntimeError, retry with delegates disabled.
    """
    try:
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        return interp
    except RuntimeError as exc:
        logging.warning("TFLite interpreter allocate_tensors failed: %s. Retrying without delegates.", exc)
        try:
            interp = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=1, experimental_delegates=[])
            interp.allocate_tensors()
            return interp
        except Exception:
            logging.exception("Retrying TFLite interpreter without delegates also failed")
            raise


def _run_inference_on_batch(interp: tf.lite.Interpreter, batch_images: np.ndarray) -> np.ndarray:
    # Ensure input dtype and shape
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Assume single input
    inp = input_details[0]
    out = output_details[0]

    input_shape = inp.get("shape", np.array([]))
    fixed_batch_size = int(input_shape[0]) if len(input_shape) > 0 and input_shape[0] not in (-1, None) else None

    if fixed_batch_size == 1 and batch_images.shape[0] != 1:
        outputs = [_run_inference_on_batch(interp, batch_images[i : i + 1]) for i in range(batch_images.shape[0])]
        return np.concatenate(outputs, axis=0)

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
