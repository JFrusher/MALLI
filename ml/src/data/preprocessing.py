"""Stain normalization for histological blood smear images.

Implements Macenko (2009) SVD-based stain normalization. Call
``normalize_stain(image_rgb)`` for numpy arrays, or wrap with
``tf_normalize_stain`` for use inside a ``tf.data`` pipeline.

Reference:
  Macenko et al., "A method for normalizing histology slides for quantitative
  analysis", ISBI 2009.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Reference stain matrix (H&E, pre-computed from a representative NIH slide).
# Columns are [Haematoxylin, Eosin] stain vectors in OD space.
# ---------------------------------------------------------------------------
_REF_STAIN_MATRIX = np.array(
    [[0.5626, 0.7201],
     [0.7201, 0.4688],
     [0.4062, 0.5112]],
    dtype=np.float32,
)

_REF_MAX_CONCENTRATIONS = np.array([1.9705, 1.0308], dtype=np.float32)

# Small constant to avoid log(0)
_EPS: float = 1e-6


def _rgb_to_od(image_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to optical density (OD) space."""
    image = image_rgb.astype(np.float32) / 255.0
    image = np.clip(image, _EPS, 1.0)
    return -np.log(image)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Convert OD image back to uint8 RGB."""
    image = np.exp(-od)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def _estimate_stain_matrix(
    od: np.ndarray,
    angular_percentile: float = 99.0,
) -> np.ndarray:
    """Estimate stain matrix from OD image via Macenko SVD method.

    Args:
        od: (H*W, 3) array of OD values for all pixels.
        angular_percentile: percentile used to select extreme stain vectors.

    Returns:
        (3, 2) stain matrix with columns = [stain1, stain2].
    """
    # Remove near-transparent pixels (background)
    od_hat = od[np.any(od > 0.15, axis=1)]
    if od_hat.shape[0] < 10:
        # Not enough tissue pixels — return reference matrix
        return _REF_STAIN_MATRIX.copy()

    # SVD to find the plane of stain variation
    _, _, Vt = np.linalg.svd(od_hat, full_matrices=False)
    plane = Vt[:2].T  # (3, 2) — principal plane

    # Project OD values onto plane and compute angles
    proj = od_hat @ plane  # (N, 2)
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    min_phi = np.percentile(phi, 100 - angular_percentile)
    max_phi = np.percentile(phi, angular_percentile)

    v1 = plane @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = plane @ np.array([np.cos(max_phi), np.sin(max_phi)])

    # Assign Haematoxylin (darker) to first column
    stain_matrix = np.column_stack([v1, v2]).astype(np.float32)
    if stain_matrix[0, 0] < stain_matrix[0, 1]:
        stain_matrix = stain_matrix[:, ::-1]

    # Normalize each column to unit length
    norms = np.linalg.norm(stain_matrix, axis=0, keepdims=True) + _EPS
    return (stain_matrix / norms).astype(np.float32)


def normalize_stain(
    image_rgb: np.ndarray,
    angular_percentile: float = 99.0,
    use_reference_matrix: bool = False,
) -> np.ndarray:
    """Normalize stain of an RGB blood smear image using Macenko's method.

    Args:
        image_rgb: uint8 ndarray of shape (H, W, 3).
        angular_percentile: percentile for stain vector estimation.
        use_reference_matrix: skip estimation and use the built-in reference
            stain matrix (faster, less adaptive).

    Returns:
        Stain-normalized uint8 ndarray of shape (H, W, 3).
    """
    h, w = image_rgb.shape[:2]
    od = _rgb_to_od(image_rgb)
    od_flat = od.reshape(-1, 3)  # (H*W, 3)

    if use_reference_matrix:
        stain_matrix = _REF_STAIN_MATRIX
    else:
        stain_matrix = _estimate_stain_matrix(od_flat, angular_percentile)

    # Solve for stain concentrations: OD = stain_matrix @ concentrations
    # Least-squares: concentrations = pinv(stain_matrix) @ od_flat.T
    stain_pinv = np.linalg.pinv(stain_matrix)  # (2, 3)
    concentrations = (stain_pinv @ od_flat.T).T  # (H*W, 2)

    # Scale concentrations to match reference maxima
    max_conc = np.percentile(concentrations, 99, axis=0) + _EPS
    scale = _REF_MAX_CONCENTRATIONS / max_conc
    concentrations_scaled = concentrations * scale

    # Reconstruct in OD space using reference stain matrix
    od_normalized = (concentrations_scaled @ _REF_STAIN_MATRIX.T).reshape(h, w, 3)

    return _od_to_rgb(od_normalized)


def tf_normalize_stain(
    image: tf.Tensor,
    angular_percentile: float = 99.0,
) -> tf.Tensor:
    """TensorFlow wrapper for ``normalize_stain``.

    Suitable for use inside ``tf.data.Dataset.map()``. Operates on a float32
    tensor in [0, 1] and returns a float32 tensor in [0, 1].

    Args:
        image: float32 tensor of shape (H, W, 3) with values in [0, 1].
        angular_percentile: passed through to ``normalize_stain``.

    Returns:
        Stain-normalized float32 tensor in [0, 1].
    """
    def _normalize(img_np: np.ndarray) -> np.ndarray:
        img_uint8 = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
        normalized = normalize_stain(img_uint8, angular_percentile=angular_percentile)
        return (normalized.astype(np.float32) / 255.0)

    normalized = tf.py_function(
        func=_normalize,
        inp=[image],
        Tout=tf.float32,
    )
    normalized.set_shape(image.shape)
    return normalized
