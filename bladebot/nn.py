"""A tiny neural-network library written with nothing but NumPy.

It contains everything BladeBot needs to train and run its parry network:

* :class:`MLP` - a fully-connected feed-forward network (multi-layer perceptron)
  with a hand-written forward and backward pass (backpropagation).
* :class:`Adam` - the Adam optimiser.
* :func:`bce_with_logits` - numerically stable binary cross-entropy.

Keeping it NumPy-only means the bot installs in seconds (no PyTorch download)
and inference for one frame takes a few microseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ACTIVATIONS = ("tanh", "relu", "leaky_relu")
MODEL_FORMAT = "bladebot-mlp-v1"


def _act_forward(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "leaky_relu":
        return np.where(z > 0.0, z, 0.01 * z)
    raise ValueError(f"unknown activation {name!r}")


def _act_backward(name: str, z: np.ndarray, a: np.ndarray, grad: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return grad * (1.0 - a * a)
    if name == "relu":
        return grad * (z > 0.0)
    if name == "leaky_relu":
        return grad * np.where(z > 0.0, 1.0, 0.01).astype(grad.dtype)
    raise ValueError(f"unknown activation {name!r}")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (``0.5 * (1 + tanh(z / 2))``)."""
    z = np.asarray(z)
    if not np.issubdtype(z.dtype, np.floating):
        z = z.astype(np.float64)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def bce_with_logits(
    logits: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy on raw logits.

    Returns ``(mean_loss, d_loss/d_logits)``. ``weights`` (optional) has one
    weight per sample (row) and is normalised so the loss is a weighted mean.
    """
    z = logits
    y = targets
    per_elem = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    grad = sigmoid(z) - y
    n_rows, n_out = z.shape
    if weights is None:
        loss = float(per_elem.mean())
        grad = grad / (n_rows * n_out)
    else:
        w = weights.reshape(-1, 1).astype(z.dtype)
        wsum = float(w.sum()) * n_out
        loss = float((per_elem * w).sum() / wsum)
        grad = grad * w / wsum
    return loss, grad.astype(z.dtype, copy=False)


class MLP:
    """Fully-connected network: ``Linear -> act -> ... -> Linear`` (logits out)."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "tanh",
        seed: int = 0,
        dtype: Any = np.float32,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("need at least an input and an output layer")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.layer_sizes = [int(n) for n in layer_sizes]
        self.activation = activation
        self.dtype = np.dtype(dtype)
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            gain = 2.0 if activation in ("relu", "leaky_relu") else 1.0  # He / Xavier
            w = rng.standard_normal((n_in, n_out)) * np.sqrt(gain / n_in)
            self.weights.append(w.astype(self.dtype))
            self.biases.append(np.zeros(n_out, dtype=self.dtype))
        self._cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

    # ------------------------------------------------------------------ core
    @property
    def n_layers(self) -> int:
        return len(self.weights)

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        """Run the network. With ``train=True`` activations are cached for backward()."""
        a = np.asarray(x, dtype=self.dtype)
        cache = [] if train else None
        last = self.n_layers - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            out = z if i == last else _act_forward(self.activation, z)
            if cache is not None:
                cache.append((a, z, out))
            a = out
        self._cache = cache
        return a

    def backward(self, grad_out: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagate ``d_loss/d_output``; returns gradients for weights and biases."""
        if self._cache is None:
            raise RuntimeError("call forward(x, train=True) before backward()")
        grads_w: list[np.ndarray] = [np.empty(0)] * self.n_layers
        grads_b: list[np.ndarray] = [np.empty(0)] * self.n_layers
        grad = np.asarray(grad_out, dtype=self.dtype)
        last = self.n_layers - 1
        for i in range(last, -1, -1):
            a_in, z, out = self._cache[i]
            if i != last:
                grad = _act_backward(self.activation, z, out, grad)
            grads_w[i] = a_in.T @ grad
            grads_b[i] = grad.sum(axis=0)
            if i > 0:
                grad = grad @ self.weights[i].T
        return grads_w, grads_b

    def parameters(self) -> list[np.ndarray]:
        return [*self.weights, *self.biases]

    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    def copy(self) -> "MLP":
        clone = MLP(self.layer_sizes, self.activation, dtype=self.dtype)
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        return clone

    # ------------------------------------------------------------ save/load
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        meta = dict(metadata or {})
        meta.update(
            format=MODEL_FORMAT,
            layer_sizes=self.layer_sizes,
            activation=self.activation,
        )
        arrays: dict[str, np.ndarray] = {"__meta__": np.array(json.dumps(meta))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends ".npz" to names without it; write to an explicit file handle.
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> tuple["MLP", dict[str, Any]]:
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            if meta.get("format") != MODEL_FORMAT:
                raise ValueError(f"{path} is not a BladeBot model file")
            net = cls(meta["layer_sizes"], meta["activation"])
            net.weights = [data[f"W{i}"].astype(np.float32) for i in range(net.n_layers)]
            net.biases = [data[f"b{i}"].astype(np.float32) for i in range(net.n_layers)]
        return net, meta


class Adam:
    """Adam optimiser (Kingma & Ba, 2015) with optional decoupled weight decay."""

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self.weight_decay and p.ndim > 1:
                update = update + self.weight_decay * p
            p -= (lr * update).astype(p.dtype, copy=False)
