from __future__ import annotations
"""Evaluate saved weights on the NIH and synthetic test sets and print reports."""
from pathlib import Path
from typing import Any

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .loaders.nih_loader import MalariaDataset
from .loaders.synthetic_loader import SyntheticFieldReadyDataset
from .factory import build_mobilenetv3_small, compile_binary_model


def load_and_compile(weights_path: str | Path, input_shape=(224, 224, 3), lr=1e-3) -> tf.keras.Model:
    model = build_mobilenetv3_small(input_shape=input_shape)
    model.load_weights(str(weights_path))
    model = compile_binary_model(model, learning_rate=lr)
    return model


def _evaluate_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    dataset_name: str,
) -> dict[str, Any]:
    """Run one analysis pass and print the results."""

    results = model.evaluate(dataset, return_dict=True, verbose=0)

    y_true = []
    y_pred = []
    y_prob = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).reshape(-1)
        preds = (probs >= 0.5).astype(int)
        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    report = classification_report(y_true, y_pred, target_names=["Uninfected", "Parasitized"], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {dataset_name.upper()} EVALUATION ===")
    print("Evaluation results (summary):", results)
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", cm)

    return {
        "summary": results,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "probabilities": y_prob,
    }


def evaluate_weights(
    weights_path: str | Path,
    dataset_root: str | Path,
    batch_size: int = 64,
    zip_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    synthetic_dataset_root: str | Path = "synthetic_field_ready",
    synthetic_labels_csv: str | Path = "labels.csv",
) -> dict[str, Any]:
    ds = MalariaDataset(
        dataset_root=dataset_root,
        image_size=(224, 224),
        batch_size=batch_size,
        test_split=0.2,
        seed=42,
        zip_path=zip_path,
        cache_dir=cache_dir,
        use_tfrecord_cache=True,
        extract_zip=False,
    )
    _, nih_test_ds = ds.create_datasets()

    synthetic_ds = SyntheticFieldReadyDataset(
        dataset_root=synthetic_dataset_root,
        labels_csv=synthetic_labels_csv,
        image_size=(224, 224),
        batch_size=batch_size,
        test_split=0.2,
        seed=42,
        augment_training=False,
    )
    _, synthetic_test_ds = synthetic_ds.create_datasets()

    model = load_and_compile(weights_path)
    nih_results = _evaluate_dataset(model, nih_test_ds, "NIH")
    synthetic_results = _evaluate_dataset(model, synthetic_test_ds, "Synthetic")

    return {"nih": nih_results, "synthetic": synthetic_results}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to weights file")
    parser.add_argument("--data-root", default="nih_data", help="Path to unzipped NIH data root")
    parser.add_argument("--zip-path", default=None, help="Optional path to the Dataverse ZIP archive")
    parser.add_argument("--cache-dir", default=None, help="Optional TFRecord cache directory")
    parser.add_argument(
        "--synthetic-root",
        default="synthetic_field_ready",
        help="Path to the synthetic dataset root.",
    )
    parser.add_argument(
        "--synthetic-labels-csv",
        default="labels.csv",
        help="Path to the synthetic labels CSV relative to the synthetic root.",
    )
    args = parser.parse_args()
    evaluate_weights(
        args.weights,
        args.data_root,
        zip_path=args.zip_path,
        cache_dir=args.cache_dir,
        synthetic_dataset_root=args.synthetic_root,
        synthetic_labels_csv=args.synthetic_labels_csv,
    )
