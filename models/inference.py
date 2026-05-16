"""Inference helpers for MobileNetV3 model.

Optimized for M.A.L.L.I. to handle both .keras (full models) and .h5 (weights).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
import tensorflow as tf

# Force Legacy Keras if needed for compatibility with older training sessions
if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
    try:
        import tf_keras as keras
    except ImportError:
        keras = tf.keras
else:
    keras = tf.keras

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
    model_path: str | Path,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    compile_model: bool = False,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Improved M.A.L.L.I. Loader:
    Forces weight loading by topology to bypass 'Conv' vs 'Conv2D' naming mismatches.
    """
    path = str(model_path)
    
    # 1. Always build the skeleton first so we have a target for the weights
    model = build_mobilenetv3_small(input_shape=input_shape)
    
    logging.debug("Attempting to load weights into architecture from %s", path)
    
    try:
        # 2. Load weights by TOPOLOGY (by_name=False)
        # This is the "magic fix" for the UserWarnings you're seeing.
        # It maps weights based on the order of layers rather than their string names.
        model.load_weights(path, by_name=False, skip_mismatch=False)
        logging.info("Weights loaded successfully via topology mapping.")
        
    except Exception as e:
        logging.warning("Topology load failed for %s: %s. Trying flexible name match.", path, e)
        try:
            # Fallback: if topology fails, try name matching but skip what doesn't fit
            model.load_weights(path, by_name=True, skip_mismatch=True)
            logging.info("Weights loaded via flexible name matching.")
        except Exception as e2:
            logging.error("Critical failure loading weights from %s: %s", path, e2)

    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
        )
    
    return model


def _load_image(path: str | Path, size: Tuple[int, int]) -> np.ndarray:
    """Preprocess image for MobileNetV3."""
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def predict_image(
    model_or_path: str | Path | tf.keras.Model,
    image_path: str | Path,
    input_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.5,
    threshold_path: str | Path | None = None,
) -> Tuple[float, int]:
    """Return (probability, predicted_label) for a single image file."""
    
    # Avoid reloading the model for every single image in a loop
    if isinstance(model_or_path, (str, Path)):
        model = load_model_with_weights(model_or_path, input_shape=(input_size[0], input_size[1], 3))
    else:
        model = model_or_path

    arr = _load_image(image_path, input_size)
    # expand_dims creates the batch dimension [1, 224, 224, 3]
    probs = model.predict(np.expand_dims(arr, axis=0), verbose=0).reshape(-1)
    prob = float(probs[0])
    
    threshold = load_decision_threshold(threshold_path, default=threshold)
    label = int(prob >= threshold)
    return prob, label


def predict_batch(
    model_or_path: str | Path | tf.keras.Model,
    image_paths: List[str] | List[Path],
    input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> List[float]:
    """Predict probabilities for a list of image files."""
    
    if isinstance(model_or_path, (str, Path)):
        model = load_model_with_weights(model_or_path, input_shape=(input_size[0], input_size[1], 3))
    else:
        model = model_or_path

    arrs = [_load_image(p, input_size) for p in image_paths]
    probs = model.predict(np.stack(arrs, axis=0), batch_size=batch_size, verbose=0).reshape(-1)
    return [float(p) for p in probs]
