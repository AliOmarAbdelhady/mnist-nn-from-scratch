"""Train the full MNIST model end-to-end, save weights + figures.

Pipeline:
  1. Load + standardize MNIST (using train statistics).
  2. Stratified train/val split.
  3. Gradient check on a tiny batch (must pass).
  4. Train MLP [784,128,64,10] with Adam + dropout + L2 + LR decay.
  5. Evaluate on the held-out test set.
  6. Save weights (models/nn.npz) and figures (figures/*.png).

    python scripts/train.py            # full run, ~5 min
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nn_from_scratch.data import load_mnist, standardize, train_val_split
from nn_from_scratch.model import MLP
from nn_from_scratch.train import train, save_weights
from nn_from_scratch.gradient_check import gradient_check
from nn_from_scratch.utils import (accuracy, confusion_matrix, one_hot,
                                   plot_confusion, plot_curves,
                                   plot_misclassified, plot_sample_predictions,
                                   seed_all)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-gradcheck", action="store_true")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable affine data augmentation")
    ap.add_argument("--patience", type=int, default=12,
                    help="early-stopping patience (epochs w/o val-loss improvement)")
    args = ap.parse_args()

    use_augment = not args.no_augment

    seed_all(args.seed)
    models_dir = ROOT / "models"; figures_dir = ROOT / "figures"
    models_dir.mkdir(exist_ok=True); figures_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------------- data
    print("Loading MNIST ...")
    Xtr_full, ytr_full, Xte, yte = load_mnist()
    print(f"  train={Xtr_full.shape}  test={Xte.shape}")
    Xtr, Xva, ytr, yva = train_val_split(Xtr_full, ytr_full, val_ratio=0.1,
                                         seed=args.seed)
    # Standardization stats from the training split only.
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True); std_safe = std.copy(); std_safe[std_safe < 1e-8] = 1.0
    Xte_s = (Xte - mean) / std_safe
    Xtr_s = (Xtr - mean) / std_safe   # used only for the gradient-check batch

    # ----------------------------------------------------- gradient check
    if not args.skip_gradcheck:
        print("\nGradient check (small model, kink-aware) ...")
        small = MLP([784, 32, 16, 10], seed=1, l2=1e-3, dropout=0.0)
        Xchk = Xtr_s[:64]; ychk = one_hot(ytr[:64])
        smooth_rel, _, _ = gradient_check(small, Xchk, ychk,
                                          num_samples=300, eps=1e-6)
        if smooth_rel >= 1e-5:
            print(f"!! Gradient check FAILED (smooth_rel={smooth_rel:.2e}). Aborting.")
            return 1

    # ----------------------------------------------------------- train
    aug_label = "WITH affine augmentation + early stopping + stronger reg" if use_augment \
        else "(no augmentation)"
    print(f"\nTraining MLP {[784,128,64,10]} for up to {args.epochs} epochs  {aug_label}")
    print(f"  dropout={args.dropout}  L2={args.l2}  lr_decay={args.lr_decay}  patience={args.patience}")
    model = MLP([784, 128, 64, 10], seed=args.seed,
                l2=args.l2, dropout=args.dropout)
    t0 = time.time()
    if use_augment:
        from nn_from_scratch.augment import make_augment
        aug_fn = make_augment(max_shift=2.0, max_rot=12.0, max_scale=0.10)
        history = train(model, Xtr, ytr, Xva, yva,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, optimizer="adam", lr_decay=args.lr_decay,
                        seed=args.seed, verbose=True,
                        augment_fn=aug_fn, mean=mean, std=std_safe,
                        early_stopping_patience=args.patience, restore_best=True)
    else:
        history = train(model, Xtr_s, ytr, (Xva-mean)/std_safe, yva,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, optimizer="adam", lr_decay=args.lr_decay,
                        seed=args.seed, verbose=True,
                        early_stopping_patience=args.patience, restore_best=True)
    dt = time.time() - t0
    n_run = len(history.train_loss)
    print(f"\nTraining time: {dt:.1f}s ({dt/max(n_run,1):.2f}s/epoch, stopped at epoch {n_run}"
          f"{' [early]' if history.stopped_early else ''})")

    # ----------------------------------------------------------- evaluate
    test_acc = accuracy(model, Xte_s, yte)
    history.test_acc = test_acc
    final_train = history.train_acc[history.best_epoch] if history.best_epoch is not None else history.train_acc[-1]
    final_val = history.val_acc[history.best_epoch] if history.best_epoch is not None else history.val_acc[-1]
    print(f"\n*** TEST ACCURACY: {test_acc*100:.2f}% ***")
    print(f"    (restored-epoch train={final_train*100:.2f}%  val={final_val*100:.2f}%  "
          f"gap={(final_train-final_val)*100:+.2f}pp)")
    # Compare to the previous baseline run if present.
    try:
        prev = json.load(open(models_dir / "metrics.json"))["test_acc"]
        print(f"    previous run test_acc: {prev*100:.2f}%   -> "
              f"{'IMPROVED' if test_acc > prev else 'no gain'} by {(test_acc-prev)*100:+.2f}pp")
    except Exception:
        pass

    y_pred = model.predict(Xte_s)
    cm = confusion_matrix(yte, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)

    # ----------------------------------------------------------- save
    save_weights(model, models_dir / "nn.npz", mean=mean, std=std_safe)
    with open(models_dir / "metrics.json", "w") as f:
        json.dump({"test_acc": test_acc, "per_class_acc": per_class.tolist(),
                   "history": history.as_dict(),
                   "config": vars(args), "train_seconds": dt,
                   "augmented": use_augment}, f, indent=2)

    plot_curves(history.as_dict(), figures_dir / "loss_accuracy_curves.png")
    plot_confusion(cm, figures_dir / "confusion_matrix.png")
    plot_sample_predictions(Xte, yte, y_pred, n=16,
                            savepath=figures_dir / "sample_predictions.png")
    plot_misclassified(Xte, yte, y_pred, n=16,
                       savepath=figures_dir / "misclassified.png")

    print("\nPer-class accuracy:")
    for d in range(10):
        print(f"  digit {d}: {per_class[d]*100:.2f}%")
    print("\nSaved:")
    print(f"  weights : {models_dir/'nn.npz'}")
    print(f"  metrics : {models_dir/'metrics.json'}")
    print(f"  figures : {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
