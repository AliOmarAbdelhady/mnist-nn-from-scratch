"""Run k-fold cross-validation on MNIST (no sklearn).

    python scripts/cross_validate.py [--folds 5] [--epochs 25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn_from_scratch.data import load_mnist, standardize
from nn_from_scratch.train import cross_validate
from nn_from_scratch.utils import seed_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    seed_all(42)
    # Use a 20k stratified subset for tractable CV; full 60k is fine but slower.
    Xtr, ytr, _, _ = load_mnist()
    Xtr = Xtr[:20000]; ytr = ytr[:20000]
    Xtr_s, _mean, _std = standardize(Xtr)

    print(f"Cross-validation: {args.folds}-fold on {len(ytr)} samples, "
          f"epochs={args.epochs}\n")
    result = cross_validate([784, args.hidden, 64, 10], Xtr_s, ytr,
                            k=args.folds, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr,
                            l2=1e-4, dropout=0.1, seed=42, verbose=True)
    print(f"\nFinal CV accuracy: {result['mean']*100:.2f}% ± {result['std']*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
