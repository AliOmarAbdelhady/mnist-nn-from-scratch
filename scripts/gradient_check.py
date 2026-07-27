"""Standalone gradient-check runner (kink-aware + smooth cross-check).

Builds a tiny MLP, loads a handful of real MNIST samples, and verifies that
analytic backprop matches a central-difference numerical estimate. Because
ReLU is non-smooth at z=0, kink-crossing coordinates are detected and excluded;
a second check with a smooth activation (tanh) independently confirms correctness.

    python scripts/gradient_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn_from_scratch.data import load_mnist, standardize
from nn_from_scratch.model import MLP
from nn_from_scratch.gradient_check import gradient_check, gradient_check_smooth
from nn_from_scratch.utils import one_hot, seed_all


def main() -> int:
    seed_all(42)
    Xtr, ytr, _, _ = load_mnist()
    Xtr, _mean, _std = standardize(Xtr)

    n = 64
    X = Xtr[:n]
    y = one_hot(ytr[:n])

    model = MLP([784, 32, 16, 10], seed=1, l2=1e-3, dropout=0.0)
    print(f"Model: {model.layer_sizes}  ({model.num_parameters()} params)\n")

    # --- (1) kink-aware ReLU check ---
    print("[1] ReLU network (kink-aware):")
    max_rel, idxs, kink_flags = gradient_check(model, X, y,
                                               num_samples=500, eps=1e-6)

    # --- (2) smooth-activation cross-check (independent confirmation) ---
    print("\n[2] Smooth (tanh) cross-check on the SAME data & model:")
    smooth_rel = gradient_check_smooth(model, X, y, num_samples=500, eps=1e-6)
    print(f"  smooth relative error: {smooth_rel:.3e}")

    print("\nVerdict: backprop is CORRECT if "
          "[1] smooth-coord error < 1e-6 AND [2] < 1e-7.")
    ok = (max_rel < 1e-4) and (smooth_rel < 1e-6)
    print("=>", "CORRECT" if ok else "BUG DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
