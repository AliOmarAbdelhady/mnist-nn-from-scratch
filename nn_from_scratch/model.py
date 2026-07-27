"""Multilayer Perceptron — implemented purely with NumPy.

A fully-connected feed-forward network with arbitrary depth, ReLU hidden
activations and a softmax output layer. He-initialization is used (optimal for
ReLU). Forward and backward passes are **fully vectorized** (no per-example
loops), and dropout / L2 are supported.

Design notes
------------
* Parameters are a list of layer dicts ``{"W", "b"}``; gradients use the
  parallel dict ``{"dW", "db"}``. This keeps the optimizer code generic.
* The forward pass caches pre-activations (``z``) and the input activation to
  each layer so backprop can reuse them — the classic forward/backward split.
* Softmax + cross-entropy are *fused* at the output: the gradient w.r.t. the
  logits is simply ``(probs - y_onehot) / m``.
* Flat-parameter helpers (``get_flat_params`` / ``set_flat_params``) make the
  numerical gradient check trivial and implementation-agnostic.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .activations import relu, relu_deriv, softmax
from .losses import softmax_cross_entropy
from .regularization import dropout_mask, l2_grad, l2_penalty


class MLP:
    """Fully-connected neural network: input -> h1 -> ... -> 10.

    Parameters
    ----------
    layer_sizes : sequence of int
        Including input and output dims, e.g. ``[784, 128, 64, 10]``.
    seed : int
        RNG seed for reproducible initialization.
    l2 : float
        L2 regularization strength (weight decay applied to W only).
    dropout : float
        Probability of dropping a hidden unit (0 = off).
    """

    def __init__(self, layer_sizes: Sequence[int], seed: int = 42,
                 l2: float = 0.0, dropout: float = 0.0):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output sizes")
        self.layer_sizes = list(layer_sizes)
        self.seed = seed
        self.l2 = float(l2)
        self.dropout = float(dropout)
        self.rng = np.random.default_rng(seed)
        self.params: List[dict] = []
        self._init_params()

    # ------------------------------------------------------------------ init
    def _init_params(self) -> None:
        """He initialization for ReLU layers (He et al., 2015).

            W ~ N(0, sqrt(2 / fan_in))     b = 0
        """
        self.params = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            std = np.sqrt(2.0 / fan_in)
            W = self.rng.standard_normal((fan_in, fan_out)) * std
            b = np.zeros((1, fan_out))
            self.params.append({"W": W, "b": b})

    @property
    def n_layers(self) -> int:
        return len(self.params)

    # ------------------------------------------------------------ inference
    def forward(self, X: np.ndarray, train: bool = False
                ) -> Tuple[np.ndarray, list]:
        """Forward pass. Returns ``(probs, cache)``.

        ``cache`` is a list with one tuple per layer ``(z_i, a_prev_i, mask_i)``
        where ``a_prev_i`` is the input to layer ``i`` (== activation of layer
        ``i-1``) and ``mask_i`` is the dropout mask applied to layer ``i``'s
        activation (hidden layers only, ``None`` otherwise).
        """
        cache: list = []
        a = X
        L = self.n_layers
        for i, p in enumerate(self.params):
            z = a @ p["W"] + p["b"]                      # pre-activation
            is_output = (i == L - 1)
            a_prev = a
            if is_output:
                probs = softmax(z, axis=1)
                cache.append((z, a_prev, None))
                a_out = probs
            else:
                h = relu(z)
                if train and self.dropout > 0:
                    mask = dropout_mask(h.shape, self.dropout, self.rng)
                    h = h * mask
                else:
                    mask = None
                cache.append((z, a_prev, mask))
                a = h
                a_out = h
        return a_out, cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X, train=False)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # --------------------------------------------------------------- losses
    def compute_loss(self, logits: np.ndarray, y_onehot: np.ndarray) -> float:
        """Stable cross-entropy computed from the *output logits* via log-sum-exp.

        Using log-sum-exp on the logits (rather than ``log(softmax + eps)``)
        guarantees the loss and its analytic gradient ``(probs - y)/m`` are
        EXACTLY consistent — no epsilon floor to break gradient checking, no
        ``log(0)``. L2 is added on top.
        """
        m = logits.shape[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_denom = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_denom                # = log softmax
        data_loss = float(-np.sum(y_onehot * log_probs) / m)
        reg_loss = sum(self.l2 * 0.5 * float(np.sum(p["W"] * p["W"]))
                       for p in self.params)
        return data_loss + reg_loss

    def loss_and_grads(self, X: np.ndarray, y_onehot: np.ndarray,
                       train: bool = False) -> Tuple[float, List[dict], np.ndarray]:
        """Forward + backward. Returns ``(loss, grads, probs)``.

        The loss is computed from the output logits (log-sum-exp) so it is
        exactly consistent with the analytic output-layer gradient.
        """
        probs, cache = self.forward(X, train=train)
        logits = cache[-1][0]
        loss = self.compute_loss(logits, y_onehot)
        grads = self.backward(cache, probs, y_onehot)
        return loss, grads, probs

    # ----------------------------------------------------------- backprop
    def backward(self, cache: list, probs: np.ndarray, y_onehot: np.ndarray
                 ) -> List[dict]:
        """Backpropagation. Returns gradients ``[{"dW","db"}, ...]``.

        For the output layer, the fused softmax+CE gradient is
        ``dL/dz = (probs - y_onehot) / m``. We fold the ``1/m`` into delta so
        that downstream ``dW = a_prev^T @ delta`` is already averaged.
        """
        m = y_onehot.shape[0]
        grads = [{"dW": None, "db": None} for _ in self.params]
        delta = (probs - y_onehot) / m                       # output-layer delta

        L = self.n_layers
        for i in reversed(range(L)):
            z_i, a_prev, mask_i = cache[i]
            grads[i]["dW"] = (a_prev.T @ delta
                              + l2_grad(self.params[i]["W"], self.l2))
            grads[i]["db"] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                # Propagate to previous layer's activation, then through its
                # dropout mask + ReLU non-linearity.
                da_prev = delta @ self.params[i]["W"].T
                z_prev, _, mask_prev = cache[i - 1]
                if mask_prev is not None:
                    da_prev = da_prev * mask_prev
                delta = da_prev * relu_deriv(z_prev)
        return grads

    # ------------------------------------------------- flat-param helpers
    def num_parameters(self) -> int:
        return sum(p["W"].size + p["b"].size for p in self.params)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([np.concatenate([p["W"].ravel(), p["b"].ravel()])
                               for p in self.params])

    def set_flat_params(self, theta: np.ndarray) -> None:
        idx = 0
        for p in self.params:
            nw = p["W"].size
            nb = p["b"].size
            p["W"] = theta[idx:idx + nw].reshape(p["W"].shape).copy()
            idx += nw
            p["b"] = theta[idx:idx + nb].reshape(p["b"].shape).copy()
            idx += nb
