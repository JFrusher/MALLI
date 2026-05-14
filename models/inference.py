"""Inference helpers for MobileNetV3 model.

Functions to load the architecture, load weights, and run single/batch inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
import tensorflow as tf

from .model_factory import build_mobilenetv3_small


def load_decision_threshold(threshold_path: str | Path | None = None, default: float = 0.5) -> float:
    """Load a calibrated decision threshold from JSON if available."""

    if threshold_path is None:
        return default

    path = Path(threshold_path)
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return float(payload.get("threshold", default))


def load_model_with_weights(
    weights_path: str | Path,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    compile_model: bool = False,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build a MobileNetV3-Small model and load `weights_path`.

    If `compile_model` is True, a compilation step will be attempted with a
    default optimizer and loss so the model can be evaluated directly.
    """
    model = build_mobilenetv3_small(input_shape=input_shape)
    model.load_weights(str(weights_path))
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
        )
    return model


def _load_image(path: str | Path, size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def predict_image(
    weights_path: str | Path,
    image_path: str | Path,
    input_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.5,
    threshold_path: str | Path | None = None,
) -> Tuple[float, int]:
    """Return (probability, predicted_label) for a single image file.

    predicted_label is 1 for infected, 0 for uninfected.
    """
    model = load_model_with_weights(weights_path, input_shape=(input_size[0], input_size[1], 3), compile_model=False)
    arr = _load_image(image_path, input_size)
    probs = model.predict(np.expand_dims(arr, axis=0), verbose=0).reshape(-1)
    prob = float(probs[0])
    threshold = load_decision_threshold(threshold_path, default=threshold)
    label = int(prob >= threshold)
    return prob, label


def predict_batch(
    weights_path: str | Path,
    image_paths: List[str] | List[Path],
    input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> List[float]:
    """Predict probabilities for a list of image files.

    Returns a list of probabilities in the same order as `image_paths`.
    """
    model = load_model_with_weights(weights_path, input_shape=(input_size[0], input_size[1], 3), compile_model=False)
    arrs = [_load_image(p, input_size) for p in image_paths]
    probs = model.predict(np.stack(arrs, axis=0), batch_size=batch_size, verbose=0).reshape(-1)
    return [float(p) for p in probs]
