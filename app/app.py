"""Gradio web UI: draw a digit, get a prediction.

Loads the trained weights from ``models/nn.npz`` and applies the *exact same*
standardization used at train time (mean/std stored alongside the weights).

    python app/app.py
    # then open the printed local URL (default http://127.0.0.1:7860)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
from PIL import Image

from nn_from_scratch.train import load_weights


WEIGHTS = ROOT / "models" / "nn.npz"
assert WEIGHTS.exists(), (
    f"Trained weights not found at {WEIGHTS}. Run `python scripts/train.py` first."
)

print("Loading model ...")
model = load_weights(WEIGHTS)
mean = getattr(model, "_mean", None)
std = getattr(model, "_std", None)


def preprocess(drawing: dict) -> np.ndarray:
    """Convert a Gradio Sketchpad drawing into a 784-vector matching MNIST stats.

    Reproduces the *canonical MNIST normalization* (LeCun et al.):
      1. grayscale + invert -> bright ink on black background (like MNIST).
      2. crop to the ink bounding box.
      3. aspect-preserving resize so the longest side is 20 px (the box MNIST
         digits were fit into).
      4. embed in a 28x28 field and shift so the **center of mass** sits on the
         pixel center -- exactly how MNIST was centered.
      5. standardize with the training mean/std stored with the weights.

    This is size- and stroke-thickness-invariant, so any reasonably large
    drawing maps onto the training distribution.
    """
    if drawing is None or drawing.get("composite") is None:
        return np.zeros(784)
    img = Image.fromarray(drawing["composite"]).convert("L")
    arr = 255.0 - np.asarray(img, dtype=np.float64)   # invert -> ink is bright
    arr = _mnist_normalize(arr)                        # -> 28x28, centered by mass
    flat = arr.reshape(1, -1)
    if mean is not None and std is not None:
        safe_std = std.copy(); safe_std[std < 1e-8] = 1.0
        flat = (flat - mean) / safe_std
    return flat.reshape(784)


def _mnist_normalize(ink: np.ndarray, box: int = 20, field: int = 28) -> np.ndarray:
    """Crop, fit into a 20x20 box (aspect preserved), center by center of mass."""
    # Threshold to find ink (anything notably brighter than background).
    mask = ink > ink.max() * 0.2
    if not mask.any():
        return np.zeros((field, field), dtype=np.float64)

    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = ink[y0:y1, x0:x1]

    # Resize so the longest side == `box`, preserving aspect ratio.
    h, w = crop.shape
    scale = box / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.float32)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float64,
    )

    # Place in the 28x28 field, then shift so center of mass == field center.
    canvas = np.zeros((field, field), dtype=np.float64)
    off_y = (field - new_h) // 2
    off_x = (field - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    total = canvas.sum()
    if total > 0:
        yy, xx = np.indices(canvas.shape)
        cy, cx = (canvas * yy).sum() / total, (canvas * xx).sum() / total
        shift_y = int(round(field / 2 - cy))
        shift_x = int(round(field / 2 - cx))
        canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))
    return canvas


def predict(drawing):
    x = preprocess(drawing)
    probs = model.predict_proba(x.reshape(1, -1))[0]
    pred = int(np.argmax(probs))
    bar = {str(i): float(probs[i]) for i in range(10)}
    return pred, bar, x.reshape(28, 28)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Draw a digit (0-9)", type="numpy"),
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted digit"),
        gr.Label(label="Class probabilities"),
        gr.Image(label="28x28 input (what the network sees)", height=200),
    ],
    title="MNIST Neural Network — from scratch (NumPy only)",
    description=(
        "Draw a digit on the canvas. A 784-128-64-10 MLP, implemented entirely "
        "in NumPy, classifies it. The right panel shows the preprocessed 28x28 "
        "input the model actually receives (black background, centered, "
        "standardized like MNIST)."
    ),
    examples=None,
    live=False,
)


if __name__ == "__main__":
    demo.launch()
