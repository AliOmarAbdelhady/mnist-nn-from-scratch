"""Training loop, learning-rate decay, weight (de)serialization and k-fold CV.

Everything is implemented from scratch on top of the :class:`MLP` and the
optimizers. Mini-batch SGD with per-epoch shuffling; optional exponential LR
decay; records per-epoch train (and optional val) loss/accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from .model import MLP
from .optimizers import Adam, SGD
from .utils import accuracy, one_hot


@dataclass
class TrainerHistory:
    """Container for per-epoch metrics + final test accuracy."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    test_acc: Optional[float] = None
    best_epoch: Optional[int] = None
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return {"train_loss": self.train_loss, "val_loss": self.val_loss,
                "train_acc": self.train_acc, "val_acc": self.val_acc,
                "lr": self.lr, "test_acc": self.test_acc}


def _make_optimizer(name: str, lr: float, **kw):
    name = name.lower()
    if name == "adam":
        return Adam(lr=lr, **{k: v for k, v in kw.items() if k in ("beta1", "beta2", "eps")})
    if name == "sgd":
        return SGD(lr=lr, momentum=kw.get("momentum", 0.9))
    raise ValueError(f"unknown optimizer '{name}'")


def _std_safe(std):
    s = np.asarray(std).copy()
    s[s < 1e-8] = 1.0
    return s


def train(model: MLP, X_train: np.ndarray, y_train: np.ndarray,
          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
          epochs: int = 40, batch_size: int = 128, lr: float = 1e-3,
          optimizer: str = "adam", lr_decay: float = 0.0,
          seed: int = 42, verbose: bool = True, print_every: int = 1,
          augment_fn=None, mean=None, std=None,
          early_stopping_patience: Optional[int] = None,
          restore_best: bool = True) -> TrainerHistory:
    """Train ``model`` with mini-batch gradient descent.

    Parameters
    ----------
    lr_decay : float
        Exponential decay per epoch: ``lr_t = lr * exp(-lr_decay * epoch)``.
        ``0`` disables decay.
    augment_fn : callable, optional
        ``augment_fn(X_batch_raw, rng) -> X_batch_raw_augmented``. Applied to
        each training mini-batch. When given together with ``mean``/``std`` the
        data are expected RAW (in [0, 255]); augmentation happens in raw space
        and each batch is then standardized with the provided statistics.
    mean, std : array, optional
        Per-pixel training statistics. If both are given, ``X_train``/``X_val``
        are treated as RAW and standardized internally (val once, train
        per-batch after augmentation). If omitted, the inputs are assumed
        already standardized (legacy behaviour).
    early_stopping_patience : int, optional
        Stop if validation loss has not improved for this many epochs.
    restore_best : bool
        Restore the parameters with the lowest validation loss at the end.
    """
    rng = np.random.default_rng(seed)
    opt = _make_optimizer(optimizer, lr, momentum=0.9)
    hist = TrainerHistory()
    n = X_train.shape[0]

    # Resolve standardization mode.
    raw_mode = mean is not None and std is not None
    if raw_mode:
        _std = _std_safe(std)
        _mean = np.asarray(mean)
        X_val_eval = (X_val - _mean) / _std if X_val is not None else None
        X_train_eval = (X_train - _mean) / _std
    else:
        X_val_eval = X_val
        X_train_eval = X_train

    Y_val = one_hot(y_val) if X_val_eval is not None else None

    best_val_loss = np.inf
    best_params = None
    best_epoch = -1
    epochs_no_improve = 0
    stopped_early = False

    for epoch in range(epochs):
        cur_lr = lr * np.exp(-lr_decay * epoch)
        opt.lr = cur_lr
        perm = rng.permutation(n)
        Xs, ys = X_train[perm], y_train[perm]

        running_loss, steps = 0.0, 0
        for s in range(0, n, batch_size):
            xb = Xs[s:s + batch_size]
            yb = ys[s:s + batch_size]
            if augment_fn is not None:
                xb = augment_fn(xb, rng)
            if raw_mode:
                xb = (xb - _mean) / _std
            yb_oh = one_hot(yb, model.layer_sizes[-1])
            loss, grads, _ = model.loss_and_grads(xb, yb_oh, train=True)
            opt.step(model.params, grads)
            running_loss += loss; steps += 1

        train_loss = running_loss / steps
        tr_acc = _acc_subsample(model, X_train_eval, y_train, rng, max_n=20000)
        hist.train_loss.append(train_loss); hist.train_acc.append(tr_acc); hist.lr.append(cur_lr)

        if X_val_eval is not None:
            vloss, _, _ = model.loss_and_grads(X_val_eval, Y_val, train=False)
            v_acc = accuracy(model, X_val_eval, y_val)
            hist.val_loss.append(vloss); hist.val_acc.append(v_acc)
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                tag = ""
                if vloss < best_val_loss - 1e-6:
                    tag = "  *"
                print(f"epoch {epoch+1:3d}/{epochs}  lr={cur_lr:.4g}  "
                      f"loss={train_loss:.4f}  acc={tr_acc:.4f}  "
                      f"val_loss={vloss:.4f}  val_acc={v_acc:.4f}{tag}")
            # best-weight tracking + early stopping
            if vloss < best_val_loss - 1e-6:
                best_val_loss = vloss
                best_epoch = epoch
                if restore_best:
                    best_params = [{"W": p["W"].copy(), "b": p["b"].copy()}
                                   for p in model.params]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping_patience is not None and \
                        epochs_no_improve >= early_stopping_patience:
                    stopped_early = True
                    if verbose:
                        print(f"  >> early stopping: no val-loss improvement for "
                              f"{early_stopping_patience} epochs (best at epoch {best_epoch+1})")
                    break
        else:
            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                print(f"epoch {epoch+1:3d}/{epochs}  lr={cur_lr:.4g}  "
                      f"loss={train_loss:.4f}  acc={tr_acc:.4f}")

    if restore_best and best_params is not None:
        for p, bp in zip(model.params, best_params):
            p["W"] = bp["W"]; p["b"] = bp["b"]
        if verbose and X_val_eval is not None and best_epoch != epochs - 1:
            print(f"  >> restored best params from epoch {best_epoch+1} "
                  f"(val_loss={best_val_loss:.4f})")
    hist.best_epoch = best_epoch
    hist.stopped_early = stopped_early
    return hist


def _acc_subsample(model, X, y, rng, max_n=20000):
    if X.shape[0] > max_n:
        idx = rng.choice(X.shape[0], max_n, replace=False)
        return accuracy(model, X[idx], y[idx])
    return accuracy(model, X, y)


# ----------------------------------------------------- weight persistence
def save_weights(model: MLP, path: str | Path, mean=None, std=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {f"W{i}": p["W"] for i, p in enumerate(model.params)}
    data.update({f"b{i}": p["b"] for i, p in enumerate(model.params)})
    data["layer_sizes"] = np.array(model.layer_sizes)
    if mean is not None:
        data["mean"] = np.asarray(mean)
    if std is not None:
        data["std"] = np.asarray(std)
    np.savez(path, **data)


def load_weights(path: str | Path) -> MLP:
    data = np.load(path, allow_pickle=False)
    layer_sizes = data["layer_sizes"].tolist()
    model = MLP(layer_sizes, seed=0, l2=0.0, dropout=0.0)
    for i in range(len(layer_sizes) - 1):
        model.params[i]["W"] = data[f"W{i}"]
        model.params[i]["b"] = data[f"b{i}"]
    model._mean = data["mean"] if "mean" in data else None
    model._std = data["std"] if "std" in data else None
    return model


# ----------------------------------------------------- cross-validation
def cross_validate(layer_sizes: Sequence[int], X: np.ndarray, y: np.ndarray,
                   k: int = 5, epochs: int = 25, batch_size: int = 128,
                   lr: float = 1e-3, optimizer: str = "adam", lr_decay: float = 0.0,
                   l2: float = 1e-4, dropout: float = 0.1, seed: int = 42,
                   verbose: bool = True, augment: bool = False) -> dict:
    """Stratified k-fold cross-validation (no sklearn).

    Returns ``{"fold_accs": [...], "mean": float, "std": float}``.

    If ``augment`` is True, each fold trains with from-scratch affine
    augmentation (raw data + per-fold standardization).
    """
    from .augment import make_augment
    aug_fn = make_augment() if augment else None

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    folds = [[] for _ in range(k)]
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, sample in enumerate(idx):
            folds[j % k].append(sample)
    folds = [np.array(sorted(f)) for f in folds]

    accs = []
    for fold in range(k):
        val_idx = folds[fold]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != fold])
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[val_idx], y[val_idx]
        # Standardize with TRAIN-fold statistics (val never leaks into them).
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True)
        std_safe = std.copy(); std_safe[std_safe < 1e-8] = 1.0
        model = MLP(layer_sizes, seed=seed + fold, l2=l2, dropout=dropout)
        if augment:
            # raw-mode: augment raw batch, standardize with fold stats.
            train(model, Xtr, ytr, Xva, yva, epochs=epochs, batch_size=batch_size,
                  lr=lr, optimizer=optimizer, lr_decay=lr_decay, seed=seed + fold,
                  verbose=False, augment_fn=aug_fn, mean=mean, std=std_safe,
                  restore_best=True)
            Xva_s = (Xva - mean) / std_safe
        else:
            Xtr_s = (Xtr - mean) / std_safe
            Xva_s = (Xva - mean) / std_safe
            train(model, Xtr_s, ytr, epochs=epochs, batch_size=batch_size,
                  lr=lr, optimizer=optimizer, lr_decay=lr_decay, seed=seed + fold,
                  verbose=False)
        acc = accuracy(model, Xva_s, yva)
        accs.append(acc)
        if verbose:
            print(f"  fold {fold+1}/{k}  val_acc = {acc:.4f}")
    out = {"fold_accs": accs, "mean": float(np.mean(accs)),
           "std": float(np.std(accs))}
    if verbose:
        print(f"  CV accuracy: {out['mean']:.4f} ± {out['std']:.4f}")
    return out
