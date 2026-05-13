"""Live training dashboard utilities for TensorBoard-based monitoring."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import tensorflow as tf


def launch_tensorboard(log_dir: Path, port: int = 6006) -> subprocess.Popen[str]:
    """Launch TensorBoard as a background process for live dashboard viewing."""
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--port",
        str(port),
        "--reload_interval",
        "2",
    ]
    process = subprocess.Popen(command)
    logging.info("TensorBoard launched at http://localhost:%d", port)
    return process


class LiveDashboardCallback(tf.keras.callbacks.Callback):
    """Write rich diagnostics to TensorBoard during training."""

    def __init__(
        self,
        log_dir: Path,
        validation_ds: tf.data.Dataset,
        prediction_threshold: float = 0.5,
        val_monitor_batches: Optional[int] = 20,
    ) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.validation_ds = validation_ds
        self.prediction_threshold = prediction_threshold
        self.val_monitor_batches = val_monitor_batches
        self.summary_writer = tf.summary.create_file_writer(str(log_dir / "diagnostics"))
        self._epoch_start_time = 0.0

    def on_epoch_begin(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        """Capture epoch start time for throughput and timing metrics."""
        del epoch, logs
        self._epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        """Log medical and optimization diagnostics after each epoch."""
        logs = logs or {}
        epoch_time = time.perf_counter() - self._epoch_start_time
        learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        y_true, y_prob = self._collect_validation_predictions()
        y_pred = (y_prob >= self.prediction_threshold).astype(np.int32)
        metrics = self._medical_metrics(y_true, y_pred)

        with self.summary_writer.as_default():
            tf.summary.scalar("optimizer/learning_rate", learning_rate, step=epoch)
            tf.summary.scalar("runtime/epoch_time_sec", epoch_time, step=epoch)

            if "loss" in logs and "val_loss" in logs:
                tf.summary.scalar(
                    "diagnostics/generalization_gap",
                    float(logs["val_loss"]) - float(logs["loss"]),
                    step=epoch,
                )

            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(f"medical/{metric_name}", metric_value, step=epoch)

            confusion_image = self._confusion_matrix_image(
                tn=metrics["tn"],
                fp=metrics["fp"],
                fn=metrics["fn"],
                tp=metrics["tp"],
            )
            tf.summary.image("medical/confusion_matrix", confusion_image, step=epoch)
            tf.summary.histogram("medical/val_predicted_probability", y_prob, step=epoch)

        self.summary_writer.flush()
        logging.info(
            "Epoch %d dashboard | FNR=%.4f Specificity=%.4f BalancedAcc=%.4f EpochTime=%.2fs",
            epoch + 1,
            metrics["false_negative_rate"],
            metrics["specificity"],
            metrics["balanced_accuracy"],
            epoch_time,
        )

    def _collect_validation_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Collect validation labels and model probabilities."""
        if self.val_monitor_batches is None:
            monitor_ds = self.validation_ds
        else:
            monitor_ds = self.validation_ds.take(self.val_monitor_batches)

        y_true: list[np.ndarray] = []
        y_prob: list[np.ndarray] = []

        for batch_images, batch_labels in monitor_ds:
            batch_probs = self.model.predict(batch_images, verbose=0).reshape(-1)
            y_prob.append(batch_probs)
            y_true.append(batch_labels.numpy().reshape(-1))

        return np.concatenate(y_true), np.concatenate(y_prob)

    @staticmethod
    def _medical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute diagnostic metrics important for medical screening."""
        y_true = y_true.astype(np.int32)
        y_pred = y_pred.astype(np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        sensitivity = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        precision = _safe_div(tp, tp + fp)
        npv = _safe_div(tn, tn + fn)
        false_negative_rate = _safe_div(fn, fn + tp)
        false_positive_rate = _safe_div(fp, fp + tn)
        balanced_accuracy = (sensitivity + specificity) / 2.0

        return {
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "negative_predictive_value": npv,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
            "balanced_accuracy": balanced_accuracy,
        }

    @staticmethod
    def _confusion_matrix_image(tn: float, fp: float, fn: float, tp: float) -> tf.Tensor:
        """Convert confusion matrix values into a TensorBoard-friendly heatmap image."""
        matrix = tf.convert_to_tensor([[tn, fp], [fn, tp]], dtype=tf.float32)
        matrix = matrix / (tf.reduce_max(matrix) + tf.keras.backend.epsilon())
        image = matrix[tf.newaxis, :, :, tf.newaxis]
        image = tf.image.resize(image, size=(256, 256), method="nearest")
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def _safe_div(numerator: float, denominator: float) -> float:
    """Safely divide two numbers and avoid zero-division crashes."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
