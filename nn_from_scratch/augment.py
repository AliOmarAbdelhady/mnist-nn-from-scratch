"""From-scratch data augmentation for MNIST (pure NumPy, fully vectorized).

Implements random **affine** augmentation — small translations, rotations and
scale changes — via inverse warping with bilinear interpolation. Applied to
**raw** [0, 255] pixel data; out-of-bounds samples are filled with 0 (the MNIST
background). This is the single most effective lever for pushing a dense MLP
past ~98% on MNIST: the network can no longer memorize the exact training
images, so the train/val gap shrinks and generalization improves.

All ops are vectorized over the batch (one bilinear gather, no python loops
over pixels or samples).
"""

from __future__ import annotations

import numpy as np

_IMG = 28
_C = (_IMG - 1) / 2.0                      # center pixel coordinate (13.5)
_GRID = np.stack(np.meshgrid(              # (28,28,2): (x,y) output coords
    np.linspace(-_C, _C, _IMG), np.linspace(-_C, _C, _IMG), indexing="xy"),
    axis=-1).astype(np.float64)
GX, GY = _GRID[..., 0], _GRID[..., 1]      # (28,28) column=x, row=y


def _bilinear_sample(imgs: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``imgs`` (B,28,28) at float source coords ``xs``/``ys`` (B,28,28).

    Out-of-range coordinates are clamped to the border; the caller fills the
    resulting border band with the background value where desired.
    """
    B, H, W = imgs.shape
    x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1; y1 = y0 + 1
    dx = xs - x0; dy = ys - y0
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    bidx = np.arange(B)[:, None, None]
    Ia = imgs[bidx, y0c, x0c]
    Ib = imgs[bidx, y0c, x1c]
    Ic = imgs[bidx, y1c, x0c]
    Id = imgs[bidx, y1c, x1c]
    return (Ia * (1 - dx) * (1 - dy) + Ib * dx * (1 - dy)
            + Ic * (1 - dx) * dy + Id * dx * dy)


def augment_batch(X: np.ndarray, rng: np.random.Generator,
                  max_shift: float = 2.0, max_rot: float = 12.0,
                  max_scale: float = 0.10) -> np.ndarray:
    """Random affine augmentation of a batch of raw MNIST vectors.

    Parameters
    ----------
    X : (B, 784) float array in [0, 255]
    rng : numpy Generator (per-epoch seeded for reproducibility)
    max_shift : max translation in pixels (each axis, uniform ±)
    max_rot   : max rotation in degrees (uniform ±)
    max_scale : max scale change (uniform in [1-max_scale, 1+max_scale])

    Returns a new (B, 784) array; out-of-bounds pixels are set to 0.
    """
    X = np.asarray(X, dtype=np.float64)
    B = X.shape[0]
    imgs = X.reshape(B, _IMG, _IMG)

    angles = rng.uniform(-max_rot, max_rot, B) * np.pi / 180.0
    scales = rng.uniform(1.0 - max_scale, 1.0 + max_scale, B)
    tx = rng.uniform(-max_shift, max_shift, B)
    ty = rng.uniform(-max_shift, max_shift, B)
    cos = np.cos(angles); sin = np.sin(angles)

    # Inverse affine map: src = R(-theta) * (out - t) / s
    Dx = GX[None] - tx[:, None, None]
    Dy = GY[None] - ty[:, None, None]
    sx = (cos[:, None, None] * Dx + sin[:, None, None] * Dy) / scales[:, None, None]
    sy = (-sin[:, None, None] * Dx + cos[:, None, None] * Dy) / scales[:, None, None]

    xs = sx + _C            # back to pixel-index space
    ys = sy + _C
    sampled = _bilinear_sample(imgs, xs, ys)

    # Zero out pixels whose source was outside the canvas (border band), so the
    # padded region is the MNIST background (black) rather than a clamped edge.
    inside = (xs >= -0.5) & (xs <= _IMG - 0.5) & (ys >= -0.5) & (ys <= _IMG - 0.5)
    sampled = sampled * inside
    return sampled.reshape(B, _IMG * _IMG)


def make_augment(max_shift=2.0, max_rot=12.0, max_scale=0.10):
    """Return a callable ``aug(X, rng)`` with bound hyper-parameters."""
    def aug(X, rng):
        return augment_batch(X, rng, max_shift, max_rot, max_scale)
    return aug
