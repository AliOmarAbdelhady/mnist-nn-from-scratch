"""Data loading for MNIST — implemented from scratch.

We parse the *original* MNIST IDX binary files (the canonical format hosted by
Yann LeCun and mirrored at github.com/cvdfoundation/mnist). No sklearn /
keras / torchvision helpers are used: the IDX byte format is decoded by hand
with ``numpy.frombuffer``.

IDX format (little-endian header, big-endian body)
--------------------------------------------------
magic   : int32  -> first 2 bytes = 0x00 0x00, 3rd byte = dtype, 4th byte = ndim
dims    : ndim * int32  -> shape of each dimension
body    : raw numerical data in row-major (C) order
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Tuple

import numpy as np

# IDX dtype code -> numpy dtype (data is stored big-endian).
_IDX_DTYPES = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),   # int16 big-endian
    0x0D: np.dtype(">i4"),   # int32 big-endian
    0x0E: np.dtype(">f4"),   # float32 big-endian
    0x0F: np.dtype(">f8"),   # float64 big-endian
}

# Canonical MNIST filenames inside the project ``data/`` folder.
FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}

# File *sizes* (header included) act as a lightweight integrity check in the
# download script; the canonical decompressed byte sizes are hard-coded there.


def _read_idx(path: str | Path) -> np.ndarray:
    """Read a single IDX file (optionally gzipped) into a numpy array."""
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()

    magic = raw[:4]
    if len(magic) != 4:
        raise ValueError(f"{path}: file too small to be a valid IDX file")

    dtype_code = magic[2]
    ndim = magic[3]
    if dtype_code not in _IDX_DTYPES:
        raise ValueError(f"{path}: unknown IDX dtype code 0x{dtype_code:02x}")

    dtype = _IDX_DTYPES[dtype_code]
    # Dimensions are big-endian int32 following the magic number.
    dims = np.frombuffer(raw[4:4 + 4 * ndim], dtype=">i4")
    body_offset = 4 + 4 * ndim
    expected = int(np.prod(dims)) if ndim > 0 else 0
    actual = (len(raw) - body_offset) // dtype.itemsize
    if actual < expected:
        raise ValueError(
            f"{path}: body truncated — expected {expected} elements, got {actual}"
        )
    data = np.frombuffer(raw, dtype=dtype, offset=body_offset, count=expected)
    return data.reshape(dims)


def load_idx(path: str | Path) -> np.ndarray:
    """Public loader: IDX file -> ``np.ndarray`` (float64 for images, int64 for labels)."""
    arr = _read_idx(path)
    if arr.ndim == 3:  # image set: (N, 28, 28)
        return arr.astype(np.float64)
    return arr.astype(np.int64)


def load_mnist(data_dir: str | Path = None, flatten: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST as ``(X_train, y_train, X_test, y_test)``.

    Images are returned as float64 in ``[0, 255]``. Pass ``flatten=True``
    (default) to get shape ``(N, 784)``; ``False`` keeps ``(N, 28, 28)``.
    """
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
    paths = {k: data_dir / v for k, v in FILES.items()}

    for name, p in paths.items():
        if not p.exists() and not p.with_suffix(p.suffix + ".gz").exists():
            raise FileNotFoundError(
                f"MNIST file '{p.name}' not found in {data_dir}. "
                f"Run: python scripts/download_data.py"
            )

    def _img(path: Path) -> np.ndarray:
        if not path.exists():
            path = path.with_suffix(path.suffix + ".gz")
        arr = load_idx(path)
        return arr.reshape(arr.shape[0], -1) if flatten else arr

    def _lbl(path: Path) -> np.ndarray:
        if not path.exists():
            path = path.with_suffix(path.suffix + ".gz")
        return load_idx(path)

    Xtr = _img(paths["train_images"]); ytr = _lbl(paths["train_labels"])
    Xte = _img(paths["test_images"]);  yte = _lbl(paths["test_labels"])
    return Xtr, ytr, Xte, yte


def standardize(X_train: np.ndarray, X_val: np.ndarray | None = None,
                X_test: np.ndarray | None = None, eps: float = 1e-8
                ) -> tuple:
    """Standardize features using *training* statistics (mean / std per pixel).

    Returns ``(X_train_std, [X_val_std], [X_test_std], mean, std)`` so the exact
    same transform can be reused at inference time (Gradio app).
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < eps] = 1.0  # avoid division by zero on constant pixels
    out = [(X_train - mean) / std]
    if X_val is not None:
        out.append((X_val - mean) / std)
    if X_test is not None:
        out.append((X_test - mean) / std)
    out.extend([mean, std])
    return tuple(out)


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.1,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split (keeps class balance, deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    classes = np.unique(y)
    tr_idx, va_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * val_ratio))
        va_idx.append(idx[:k])
        tr_idx.append(idx[k:])
    tr_idx = np.concatenate(tr_idx); va_idx = np.concatenate(va_idx)
    rng.shuffle(tr_idx); rng.shuffle(va_idx)
    return X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
