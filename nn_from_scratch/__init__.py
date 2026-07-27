"""Neural Network from Scratch (MNIST).

A from-scratch feed-forward neural network implemented with **only NumPy**
(no PyTorch / TensorFlow / sklearn models). Trains on MNIST and reaches
~98% test accuracy. Includes gradient checking, k-fold cross-validation,
and a Gradio draw-and-predict web UI.
"""

from .model import MLP
from .data import load_mnist, load_idx, standardize, train_val_split
from .train import train, TrainerHistory
from .gradient_check import gradient_check, gradient_check_smooth, relative_error

__all__ = [
    "MLP",
    "load_mnist",
    "load_idx",
    "standardize",
    "train_val_split",
    "train",
    "TrainerHistory",
    "gradient_check",
    "gradient_check_smooth",
    "relative_error",
]

__version__ = "1.0.0"
