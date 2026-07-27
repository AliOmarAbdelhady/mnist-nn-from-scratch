"""Regularization — L2 weight decay and inverted dropout.

Both are implemented in the textbook "inverted" style so that no rescaling is
needed at test time (expected activation is preserved automatically).
"""

from __future__ import annotations

import numpy as np


def l2_penalty(params, lam: float) -> float:
    """L2 regularization term added to the loss: ``(lam/2) * sum(W^2)``.

    ``params`` is an iterable of (name, weight_matrix, bias) tuples; only
    weight matrices are penalized (bias terms are conventionally left out).
    """
    if lam <= 0.0:
        return 0.0
    total = 0.0
    for _name, W, _b in params:
        total += float(np.sum(W * W))
    return 0.5 * lam * total


def l2_grad(W: np.ndarray, lam: float) -> np.ndarray:
    """Gradient of ``(lam/2) * sum(W^2)`` w.r.t. W = ``lam * W``."""
    return lam * W if lam > 0 else 0.0


def dropout_mask(shape, drop_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Inverted-dropout keep mask (multiply activations by this at train time).

    With probability ``drop_prob`` a unit is dropped (multiplied by 0); the
    kept units are scaled by ``1/(1-drop_prob)`` so the expected value is
    unchanged — hence nothing special is needed at inference time.
    """
    if drop_prob <= 0.0:
        return np.ones(shape, dtype=np.float64)
    keep = 1.0 - drop_prob
    mask = rng.random(shape) < keep
    return (mask / keep).astype(np.float64)
