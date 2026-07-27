"""Activation functions — fully vectorized, numerically stable.

All functions come in pairs: the forward ``f(x)`` and its derivative
``f'(x)`` w.r.t. the pre-activation, which is what backprop needs.
"""

from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: ``max(0, z)``."""
    return np.maximum(0.0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU w.r.t. ``z``: 1 where z>0, else 0."""
    return (z > 0.0).astype(z.dtype)


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)


def leaky_relu_deriv(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax (max-subtraction trick).

    ``z`` shape: ``(batch, num_classes)``.
    """
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_deriv(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1.0 - t * t
