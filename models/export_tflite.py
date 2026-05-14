"""Export utilities to convert Keras weights into quantized TFLite."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

from .model_factory import build_mobilenetv3_small


def representative_dataset_generator(dataset: tf.data.Dataset, batches: int) -> Iterator[list[tf.Tensor]]:
    """Yield representative samples for TFLite integer quantization."""
    for images, _ in dataset.take(batches):
        for i in range(images.shape[0]):
            yield [tf.expand_dims(images[i], axis=0).numpy().astype(np.float32)]


def export_tflite_from_weights(
    weights_path: str | Path,
    output_path: str | Path,
    train_dataset: tf.data.Dataset,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    representative_batches: int = 100,
) -> None:
    """Build model, load weights, and export a fully-quantized INT8 TFLite file.

    The provided `train_dataset` should yield (images, labels) with images matching
    the model's input preprocessing.
    """
    model = build_mobilenetv3_small(input_shape=input_shape)
    model.load_weights(str(weights_path))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_generator(train_dataset, representative_batches)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
