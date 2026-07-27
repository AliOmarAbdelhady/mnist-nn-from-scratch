# Neural Network from Scratch — MNIST

A from-scratch feed-forward neural network that classifies handwritten digits,
implemented using **only NumPy** — no PyTorch, TensorFlow, or sklearn models.
Trains on MNIST to **~98% test accuracy** in ~2.5 min on a CPU, includes a
**gradient check** to verify backprop, **k-fold cross-validation**, and a
**Gradio web UI** where you can draw a digit and watch it be classified.

> **Why from scratch?** The point is to understand *exactly* how a neural
> network works: parameter initialization, the forward pass, the chain rule
> behind backprop, how loss and its gradient must stay consistent, why ReLU
> breaks naive gradient checks at its kink, and how optimizers like Adam move
> the weights. Using a framework would hide all of that.

---

## Quick start

```bash
cd /home/ali/mnist-nn-from-scratch

# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the canonical MNIST IDX files (~11 MB)
python scripts/download_data.py

# 3. Train (downloads data if missing, runs gradient check, ~2.5 min)
python scripts/train.py

# 4. Launch the draw-and-predict web UI
python app/app.py      # open the printed http://127.0.0.1:7860

# 5. Open the main deliverable notebook
jupyter notebook notebooks/mnist_nn_from_scratch.ipynb
```

> **No `.env` or API keys are needed** — this is a fully offline, pure-NumPy
> project. The only network access is the one-time MNIST download.

---

## Results

| Metric | Baseline (no aug) | **With augmentation + early stop** |
|---|---|---|
| Architecture | `784 → 128 → 64 → 10` (ReLU + softmax, He init) | same |
| Optimizer | Adam (lr 1e-3) + batch 128 | Adam (lr 1e-3) + batch 128 |
| Regularization | Dropout 0.1 + L2 1e-4 | **Dropout 0.2 + L2 3e-4** |
| Augmentation | none | **affine (shift ±2px, rotate ±12°, scale ±10%)** |
| Early stopping | no | **patience 10, best-weight restore** |
| Gradient check (kink-aware + smooth) | rel-err ≈ **1e-9** ✓ | rel-err ≈ **1e-9** ✓ |
| Train accuracy | 100.00 % (memorized) | 98.34 % |
| **Generalization gap** | **~2.0 pp (overfit)** | **~0.2 pp** |
| **Test accuracy** | 97.86 % | **98.16 %** |
| 5-fold cross-validation | 95.96 % ± 0.28 % | ~94 % ± 0.6 % (smaller subset/fewer epochs) |
| Training time | 140 s (40 ep, 3.5 s/ep) | 631 s (40 ep, 15.8 s/ep — augmentation cost) |

The baseline clearly **overfit** (100 % train vs 98 % val; val loss rising
after ~epoch 30). Adding from-scratch affine augmentation + stronger
regularization + early stopping **collapsed the train/val gap from ~2 pp to
~0.2 pp and lifted test accuracy to 98.16 %**. Per-class accuracy is now
96.8 %–99.5 %; the remaining errors are the classic ambiguous pairs
(4↔9, 8↔3, 7↔2) visible in the confusion matrix.

The default `python scripts/train.py` now trains the augmented model. The
baseline is preserved in `models/metrics_baseline.json` /
`models/nn_baseline.npz` for comparison.

---

## Project layout

```
mnist-nn-from-scratch/
├── nn_from_scratch/          # the core library (pure NumPy, importable, tested)
│   ├── data.py               #   hand-written IDX binary parser + standardization
│   ├── activations.py        #   ReLU, softmax (numerically stable), tanh, sigmoid
│   ├── losses.py             #   fused softmax + cross-entropy (log-sum-exp)
│   ├── model.py              #   MLP: He init, forward, backward, flat-param helpers
│   ├── optimizers.py         #   SGD+momentum, Adam (both from scratch)
│   ├── regularization.py     #   L2 weight decay + inverted dropout
│   ├── gradient_check.py     #   kink-aware numerical gradient check + smooth check
│   ├── train.py              #   mini-batch loop, LR decay, save/load, k-fold CV
│   └── utils.py              #   seeding, accuracy, confusion matrix, plotting
├── notebooks/
│   └── mnist_nn_from_scratch.ipynb   # ★ MAIN DELIVERABLE (code + plots + write-up)
├── scripts/
│   ├── download_data.py      #   fetch + verify the 4 IDX files (PyTorch S3 mirror)
│   ├── gradient_check.py     #   standalone gradient-check runner
│   ├── cross_validate.py     #   standalone k-fold runner
│   ├── train.py              #   train end-to-end → weights + figures + metrics
│   └── build_notebook.py     #   (re)generate + execute the main notebook
├── app/app.py                #   Gradio Sketchpad UI: draw → predict → prob bar
├── kaggle/mnist_nn_kaggle.ipynb      # Kaggle-ready backup (data from Kaggle dataset)
├── data/                     #   downloaded IDX files (gitignored)
├── models/                   #   nn.npz weights + metrics.json (gitignored)
├── figures/                  #   generated PNG plots
└── requirements.txt
```

---

## The deliverable notebook

`notebooks/mnist_nn_from_scratch.ipynb` is the single artifact that satisfies
the brief. It is fully executed (all outputs embedded) and walks through, in
order:

1. **Setup & seeding** (fixed `SEED = 42`).
2. **Data loading from scratch** — the IDX format decoded by hand, plus digit
   gallery and class distribution.
3. **Preprocessing** — standardization with training-set statistics.
4. **Activations & loss** — stability demos for softmax and the fused
   log-sum-exp cross-entropy.
5. **The model** — the actual `forward` / `backward` source (via `inspect`),
   so what you read is exactly what runs.
6. **Gradient check** — kink-aware ReLU check + an independent smooth-activation
   cross-check, both reporting rel-err ≈ 1e-9.
7. **Training** — fully-vectorized mini-batch Adam.
8. **Loss & accuracy curves.**
9. **Test accuracy + confusion matrix + per-class accuracy.**
10. **Sample predictions + the hardest misclassifications.**
11. **k-fold cross-validation** (mean ± std).
12. **Write-up & conclusions** (results table, design rationale, error
    analysis, limits of a feed-forward net, reproducibility).

---

## Design decisions (the "why")

- **Hand-written IDX parser.** It demonstrates real data loading (decoding a
  binary magic-number format with `np.frombuffer`) and keeps the project free
  of the sklearn/torch data helpers that would otherwise hide this step.
- **Fused log-sum-exp cross-entropy.** Computing the loss from the *logits*
  (rather than `log(softmax + eps)`) is both numerically stable **and** exactly
  consistent with the analytic gradient `(probs − y)/m`. An epsilon floor would
  have broken gradient checking whenever a ReLU output produced extreme logits.
- **Kink-aware gradient check.** ReLU is non-differentiable at 0; a naive
  finite-difference check reports a false "failure" whenever the step straddles
  a kink. We detect such coordinates (a hidden unit's active mask flips across
  the ±ε step) and report them separately, then add a smooth-activation
  cross-check as independent confirmation.
- **He initialization + Adam + dropout/L2.** The combination that reaches the
  ~98 % MLP ceiling quickly (minutes, not hours) on CPU without overshooting.
- **MNIST-style preprocessing in the UI.** The web app reproduces LeCun's
  original normalization (crop → fit 20×20 box → center by center of mass),
  so a hand-drawn digit of any size or position lands on the training
  distribution (validated at 100 % on transformed test images).

---

## Reproducibility

Everything is seeded (`SEED = 42`, NumPy + Python `random`). Re-running
`scripts/train.py` or the notebook reproduces the reported numbers to within
floating-point noise. The Gradio app loads the saved weights and applies the
identical preprocessing, so a drawn digit is classified exactly as test images
are.

## Going further

A dense MLP tops out near 98.4 % on MNIST because it throws away spatial
structure. Pushing past 99.5 % requires a **convolutional** network (local
receptive fields, weight sharing) — the natural next project, still buildable
from scratch in NumPy.
