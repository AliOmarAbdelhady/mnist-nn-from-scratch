"""Loss functions — cross-entropy and a fused stable softmax+cross-entropy.

The fused version is preferred for the output layer because it avoids
``log(0)`` and is both faster and more accurate than computing softmax then
NLL separately.
"""

from __future__ import annotations

import numpy as np

from .activations import softmax


def cross_entropy(probs: np.ndarray, y_onehot: np.ndarray,
                  eps: float = 1e-12) -> float:
    """Categorical cross-entropy given probabilities (already softmaxed)."""
    m = probs.shape[0]
    return float(-np.sum(y_onehot * np.log(probs + eps)) / m)


def softmax_cross_entropy(logits: np.ndarray, y_onehot: np.ndarray):
    """Fused stable softmax + cross-entropy.

    Returns ``(loss, probs)`` where ``probs`` is the softmax output (reused by
    backprop). Uses the log-sum-exp trick for numerical stability.
    """
    m = logits.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_denom                  # log p_k for each class
    loss = float(-np.sum(y_onehot * log_probs) / m)  # mean NLL
    probs = np.exp(log_probs)
    return loss, probs


def softmax_cross_entropy_deriv(probs: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """Gradient of the fused softmax+CE w.r.t. the pre-softmax logits.

        dL/dlogits = (probs - y_onehot) / m
    """
    m = probs.shape[0]
    return (probs - y_onehot) / m
