"""Utility helpers: reproducibility, metrics, one-hot encoding, plotting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

# Use a non-interactive backend so plots can be generated headless (servers /
# scripts). The notebook overrides this with the inline backend.
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def seed_all(seed: int = 42) -> None:
    """Seed every RNG we might touch for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Integer labels -> one-hot matrix of shape ``(n, num_classes)``."""
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out


def accuracy(model, X: np.ndarray, y: np.ndarray, batch: int = 4096) -> float:
    """Batched accuracy to keep memory bounded on large sets."""
    preds = []
    for s in range(0, X.shape[0], batch):
        preds.append(model.predict(X[s:s + batch]))
    preds = np.concatenate(preds)
    return float(np.mean(preds == y))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------- plotting
def plot_curves(history, savepath: str | Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss curve"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="train")
    if history.get("val_acc"):
        axes[1].plot(history["val_acc"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")


def plot_confusion(cm: np.ndarray, savepath: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")


def plot_sample_predictions(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                            shape=(28, 28), n: int = 16, ncols: int = 8,
                            savepath: str | Path | None = None) -> None:
    idx = np.arange(min(n, len(y_true)))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idx):
        ax.imshow(X[i].reshape(shape), cmap="gray")
        col = "green" if y_pred[i] == y_true[i] else "red"
        ax.set_title(f"p={y_pred[i]}/t={y_true[i]}", color=col, fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(idx):]:
        ax.set_axis_off()
    fig.suptitle("Sample predictions (green=correct, red=wrong)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")


def plot_misclassified(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                       shape=(28, 28), n: int = 16, ncols: int = 8,
                       savepath: str | Path | None = None) -> None:
    wrong = np.where(y_pred != y_true)[0]
    if len(wrong) == 0:
        return
    idx = wrong[:n]
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idx):
        ax.imshow(X[i].reshape(shape), cmap="gray")
        ax.set_title(f"t={y_true[i]} p={y_pred[i]}", color="red", fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(idx):]:
        ax.set_axis_off()
    fig.suptitle(f"Misclassified samples ({len(wrong)} total)")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")


def show_digits_grid(X: np.ndarray, y: np.ndarray, shape=(28, 28),
                     n: int = 16, ncols: int = 8,
                     savepath: str | Path | None = None) -> None:
    idx = np.arange(min(n, len(y)))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.5))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idx):
        ax.imshow(X[i].reshape(shape), cmap="gray")
        ax.set_title(str(int(y[i])), fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(idx):]:
        ax.set_axis_off()
    fig.suptitle("Sample MNIST digits")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")


def plot_class_distribution(y: np.ndarray, savepath: str | Path | None = None) -> None:
    counts = np.bincount(y.astype(int), minlength=10)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(10), counts, color="#4C72B0")
    ax.set_xticks(range(10))
    ax.set_xlabel("digit class"); ax.set_ylabel("count")
    ax.set_title("Class distribution")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
