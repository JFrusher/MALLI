"""Tests for model factory (MobileNetV3 builder and compilation)."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.factory import build_mobilenetv3_small, compile_binary_model, F1Score


def test_build_mobilenetv3_small_output_shape():
    model = build_mobilenetv3_small(input_shape=(224, 224, 3))
    assert model.output_shape == (None, 1), (
        f"Expected output shape (None, 1), got {model.output_shape}"
    )


def test_build_mobilenetv3_small_input_shape():
    model = build_mobilenetv3_small(input_shape=(224, 224, 3))
    assert model.input_shape == (None, 224, 224, 3)


def test_backbone_frozen_by_default():
    model = build_mobilenetv3_small(train_backbone=False)
    backbone = next(l for l in model.layers if "mobilenet" in l.name.lower())
    assert not backbone.trainable, "Backbone should be frozen when train_backbone=False"


def test_backbone_unfrozen_when_requested():
    model = build_mobilenetv3_small(train_backbone=True)
    backbone = next(l for l in model.layers if "mobilenet" in l.name.lower())
    assert backbone.trainable, "Backbone should be trainable when train_backbone=True"


def test_compile_binary_model_has_metrics():
    model = build_mobilenetv3_small()
    model = compile_binary_model(model, learning_rate=1e-3)
    metric_names = [m.name for m in model.metrics]
    assert "recall" in metric_names
    assert "precision" in metric_names
    assert "auc" in metric_names


def test_compile_binary_model_uses_adam():
    model = build_mobilenetv3_small()
    model = compile_binary_model(model, learning_rate=5e-4)
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)


def test_forward_pass_produces_probability():
    model = build_mobilenetv3_small()
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    out = model.predict(dummy, verbose=0)
    assert out.shape == (1, 1)
    assert 0.0 <= float(out[0, 0]) <= 1.0, "Output should be a probability in [0, 1]"


def test_f1score_metric_perfect_predictions():
    metric = F1Score(threshold=0.5)
    y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
    y_pred = tf.constant([0.9, 0.1, 0.8, 0.2])
    metric.update_state(y_true, y_pred)
    assert float(metric.result()) == pytest.approx(1.0, abs=1e-5)


def test_f1score_metric_reset():
    metric = F1Score(threshold=0.5)
    y_true = tf.constant([1.0])
    y_pred = tf.constant([0.9])
    metric.update_state(y_true, y_pred)
    metric.reset_state()
    assert float(metric.result()) == pytest.approx(0.0, abs=1e-5)
