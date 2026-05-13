"""Model creation and compilation utilities for Project M.A.L.L.I."""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf


class F1Score(tf.keras.metrics.Metric):
    """Streaming F1-score metric for binary classification."""

    def __init__(self, name: str = "f1_score", threshold: float = 0.5, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        """Update confusion counts using thresholded predictions."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self) -> tf.Tensor:
        """Compute F1 from accumulated true positives/false positives/false negatives."""
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2.0 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self) -> None:
        """Reset metric state at the start of each epoch."""
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


def build_mobilenetv3_small(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
    train_backbone: bool = False,
) -> tf.keras.Model:
    """Create a transfer-learning MobileNetV3-Small model for binary output."""
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights="imagenet",
        pooling=None,
        dropout_rate=0.0,
    )
    base_model.trainable = train_backbone

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="infected_probability",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mobili_mobilenetv3_small")


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile model with recall-focused validation metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            F1Score(name="f1_score"),
        ],
    )
    return model
