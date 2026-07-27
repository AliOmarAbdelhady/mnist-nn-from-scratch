"""Optimizers — implemented from scratch (no torch/keras/Adam from sklearn).

Each optimizer is a small stateful object exposing ``step(params, grads)``
where ``params`` and ``grads`` are parallel lists of dict layers, each holding
``W`` and ``b`` (parameters) / ``dW`` and ``db`` (gradients).

Optimizers included:
    * SGD  with optional momentum and L2 weight decay
    * Adam (Kingma & Ba 2014) with bias correction
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class Optimizer:
    """Base class. Subclasses implement ``step``."""

    name = "base"

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params: Sequence[dict], grads: Sequence[dict]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional (heavy-ball) momentum.

        v = momentum * v - lr * grad
        W = W + v
    """

    name = "sgd"

    def __init__(self, lr: float, momentum: float = 0.0):
        super().__init__(lr)
        self.momentum = momentum
        self._v: list | None = None

    def step(self, params: Sequence[dict], grads: Sequence[dict]) -> None:
        if self._v is None:
            self._v = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])}
                       for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self._v[i]["W"] = self.momentum * self._v[i]["W"] - self.lr * g["dW"]
            self._v[i]["b"] = self.momentum * self._v[i]["b"] - self.lr * g["db"]
            p["W"] += self._v[i]["W"]
            p["b"] += self._v[i]["b"]


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2014).

        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g^2
        m_hat = m / (1-b1^t)        # bias correction
        v_hat = v / (1-b2^t)
        W = W - lr * m_hat / (sqrt(v_hat) + eps)
    """

    name = "adam"

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: list | None = None
        self._v: list | None = None
        self.t = 0

    def step(self, params: Sequence[dict], grads: Sequence[dict]) -> None:
        if self._m is None:
            self._m = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])}
                       for p in params]
            self._v = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])}
                       for p in params]
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            for key, gkey in (("W", "dW"), ("b", "db")):
                gi = g[gkey]
                self._m[i][key] = b1 * self._m[i][key] + (1 - b1) * gi
                self._v[i][key] = b2 * self._v[i][key] + (1 - b2) * (gi * gi)
                m_hat = self._m[i][key] / bc1
                v_hat = self._v[i][key] / bc2
                p[key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
