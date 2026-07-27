"""Numerical gradient checking — kink-aware for ReLU networks.

Backprop is notoriously easy to get subtly wrong. The standard verification is
to compare analytic gradients against finite-difference approximations and look
at the *relative error*:

    rel_err = ||g_ana - g_num|| / (||g_ana|| + ||g_num||)

For a smooth network a correct implementation yields rel_err < 1e-7 (often
~1e-10). **ReLU is not smooth at z=0**, so a central-difference step that
straddles a kink yields a wrong numerical estimate for that coordinate even
though backprop is perfectly correct. We therefore detect kink-crossing
coordinates (where a hidden unit's activation flips across the ±eps step) and
report them separately, computing the headline relative error only over the
*smooth* coordinates.

A second, independent confirmation: temporarily swapping ReLU for a smooth
activation (tanh) drives the error to ~1e-9 on identical data (see
``gradient_check_smooth``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .activations import relu, relu_deriv, tanh, tanh_deriv


def relative_error(g1: np.ndarray, g2: np.ndarray, eps: float = 1e-12) -> float:
    """Symmetric relative error between two gradient vectors."""
    num = np.linalg.norm(g1 - g2)
    den = np.linalg.norm(g1) + np.linalg.norm(g2) + eps
    return float(num / den)


def _hidden_masks(model, cache) -> list:
    """ReLU activation masks (z>0) for every hidden layer (index < L-1)."""
    L = model.n_layers
    return [cache[i][0] > 0.0 for i in range(L - 1)]


def _flatten_grads(grads) -> np.ndarray:
    return np.concatenate([np.concatenate([g["dW"].ravel(), g["db"].ravel()])
                           for g in grads])


def gradient_check(model, X: np.ndarray, y_onehot: np.ndarray,
                   num_samples: int = 200, eps: float = 1e-6,
                   seed: int = 0, verbose: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Kink-aware analytic-vs-numerical gradient check.

    Returns ``(smooth_rel_err, sampled_idx, kink_flags)`` where ``kink_flags``
    is a boolean array marking coordinates whose ±eps step crossed a ReLU kink
    (excluded from ``smooth_rel_err``).
    """
    original_dropout = model.dropout
    model.dropout = 0.0
    try:
        loss, grads, _ = model.loss_and_grads(X, y_onehot, train=False)
        g_flat = _flatten_grads(grads)
        theta = model.get_flat_params()
        P = theta.size

        _, cache_base = model.forward(X, train=False)
        masks_base = _hidden_masks(model, cache_base)

        rng = np.random.default_rng(seed)
        idxs = rng.choice(P, size=min(num_samples, P), replace=False)

        g_num = np.zeros(P)
        kinked = np.zeros(P, dtype=bool)
        for j in idxs:
            orig = theta[j]
            theta[j] = orig + eps
            model.set_flat_params(theta)
            loss_plus, _, cp = model.loss_and_grads(X, y_onehot, train=False)
            masks_plus = _hidden_masks(model, cp)
            theta[j] = orig - eps
            model.set_flat_params(theta)
            loss_minus, _, cm = model.loss_and_grads(X, y_onehot, train=False)
            masks_minus = _hidden_masks(model, cm)
            theta[j] = orig
            g_num[j] = (loss_plus - loss_minus) / (2 * eps)
            # Kink if any hidden unit flipped active<->inactive across the step.
            flipped = any(not np.array_equal(mp, mm)
                          for mp, mm in zip(masks_plus, masks_minus))
            kinked[j] = flipped
        model.set_flat_params(theta)

        smooth = ~kinked
        smooth_ids = idxs[~kinked[idxs]]
        if smooth_ids.size > 0:
            smooth_rel = relative_error(g_flat[smooth_ids], g_num[smooth_ids])
        else:
            smooth_rel = float("nan")

        if verbose:
            n_kink = int(kinked[idxs].sum())
            print(f"Gradient check on {len(idxs)}/{P} params  (eps={eps:g})")
            print(f"  kink-crossing coords : {n_kink}/{len(idxs)} "
                  f"({100*n_kink/len(idxs):.1f}%)  [excluded from headline metric]")
            print(f"  smooth relative error: {smooth_rel:.3e}   "
                  f"(over {len(smooth_ids)} coords)")
            verdict = ("PASS  (correct backprop)" if smooth_rel < 1e-6 else
                       "PASS (acceptable)" if smooth_rel < 1e-4 else "FAIL (check backprop)")
            print(f"  verdict              : {verdict}")
        return smooth_rel, idxs, kinked[idxs]
    finally:
        model.dropout = original_dropout


def gradient_check_smooth(model, X: np.ndarray, y_onehot: np.ndarray,
                          num_samples: int = 200, eps: float = 1e-6,
                          seed: int = 0) -> float:
    """Independent confirmation: swap ReLU -> tanh (smooth) and re-check.

    Because tanh is everywhere differentiable, a correct backprop must yield
    rel_err < 1e-7 here regardless of the data. This isolates "is the chain
    rule wired right" from "is the ReLU derivative right".
    """
    import nn_from_scratch.model as M
    orig_relu, orig_deriv = M.relu, M.relu_deriv
    M.relu, M.relu_deriv = tanh, tanh_deriv
    try:
        original_dropout = model.dropout
        model.dropout = 0.0
        loss, grads, _ = model.loss_and_grads(X, y_onehot, train=False)
        g_flat = _flatten_grads(grads)
        theta = model.get_flat_params(); P = theta.size
        rng = np.random.default_rng(seed)
        idxs = rng.choice(P, size=min(num_samples, P), replace=False)
        g_num = np.zeros(P)
        for j in idxs:
            o = theta[j]
            theta[j] = o + eps; model.set_flat_params(theta)
            lp, _, _ = model.loss_and_grads(X, y_onehot, train=False)
            theta[j] = o - eps; model.set_flat_params(theta)
            lm, _, _ = model.loss_and_grads(X, y_onehot, train=False)
            theta[j] = o
            g_num[j] = (lp - lm) / (2 * eps)
        model.set_flat_params(theta)
        model.dropout = original_dropout
        return relative_error(g_flat[idxs], g_num[idxs])
    finally:
        M.relu, M.relu_deriv = orig_relu, orig_deriv
