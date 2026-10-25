"""A convolutional neural network written from scratch in NumPy: forward pass, backpropagation, Adam.

Layers: Conv2D (im2col), ReLU, MaxPool2D, Flatten, Dense, Dropout, softmax cross-entropy.
Images use NCHW layout (batch, channels, height, width).

Written for clarity first; im2col keeps convolutions vectorised so a small
network trains on CPU in seconds. Gradients are verified numerically in tests.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- helpers

def im2col(x: np.ndarray, k: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """(N, C, H, W) → (N * out_h * out_w, C * k * k) patches."""
    n, c, h, w = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    cols = np.empty((n, c, k, k, oh, ow), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = x[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, -1)


def col2im(cols: np.ndarray, shape, k: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Inverse of im2col: scatter-add patch gradients back into an image gradient."""
    n, c, h, w = shape
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    cols = cols.reshape(n, oh, ow, c, k, k).transpose(0, 3, 4, 5, 1, 2)
    x = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(k):
        for j in range(k):
            x[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride] += cols[:, :, i, j]
    return x[:, :, pad:pad + h, pad:pad + w]


# ----------------------------------------------------------------------------- layers

class Layer:
    params: dict
    grads: dict
    training = True

    def __init__(self):
        self.params, self.grads = {}, {}


class Conv2D(Layer):
    def __init__(self, c_in: int, c_out: int, k: int = 3, pad: int = 1, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        self.k, self.pad = k, pad
        self.params["W"] = rng.normal(0, np.sqrt(2 / (c_in * k * k)), (c_out, c_in, k, k))  # He initialisation
        self.params["b"] = np.zeros(c_out)

    def forward(self, x):
        self.x_shape = x.shape
        n, _, h, w = x.shape
        self.cols = im2col(x, self.k, 1, self.pad)
        W = self.params["W"].reshape(self.params["W"].shape[0], -1)
        out = self.cols @ W.T + self.params["b"]
        return out.reshape(n, h, w, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        c_out = dout.shape[1]
        d = dout.transpose(0, 2, 3, 1).reshape(-1, c_out)
        W = self.params["W"].reshape(c_out, -1)
        self.grads["W"] = (d.T @ self.cols).reshape(self.params["W"].shape)
        self.grads["b"] = d.sum(axis=0)
        return col2im(d @ W, self.x_shape, self.k, 1, self.pad)


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D(Layer):
    def __init__(self, size: int = 2):
        super().__init__()
        self.s = size

    def forward(self, x):
        n, c, h, w = x.shape
        s = self.s
        xr = x.reshape(n, c, h // s, s, w // s, s)
        out = xr.max(axis=(3, 5))
        self.mask = xr == out[:, :, :, None, :, None]
        self.mask = self.mask / self.mask.sum(axis=(3, 5), keepdims=True)  # split ties evenly
        self.shape = x.shape
        return out

    def backward(self, dout):
        return (self.mask * dout[:, :, :, None, :, None]).reshape(self.shape)


class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        self.params["W"] = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.params["b"] = np.zeros(n_out)

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


class Dropout(Layer):
    def __init__(self, p: float = 0.3, rng=None):
        super().__init__()
        self.p, self.rng = p, rng or np.random.default_rng(0)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        self.mask = (self.rng.random(x.shape) > self.p) / (1 - self.p)  # inverted dropout
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask if self.training and self.p > 0 else dout


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean loss and gradient w.r.t. logits (softmax + NLL combined for stability)."""
    p = softmax(logits)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    grad = p.copy()
    grad[np.arange(n), y] -= 1
    return float(loss), grad / n


# ----------------------------------------------------------------------------- model

class CNN:
    """conv(16) → relu → pool → conv(32) → relu → pool → dense(64) → relu → dropout → dense(classes)"""

    def __init__(self, in_shape=(1, 8, 8), n_classes: int = 10, seed: int = 0, dropout: float = 0.3):
        rng = np.random.default_rng(seed)
        c, h, w = in_shape
        self.layers = [
            Conv2D(c, 16, rng=rng), ReLU(), MaxPool2D(2),
            Conv2D(16, 32, rng=rng), ReLU(), MaxPool2D(2),
            Flatten(), Dense(32 * (h // 4) * (w // 4), 64, rng=rng), ReLU(), Dropout(dropout, rng=rng),
            Dense(64, n_classes, rng=rng),
        ]
        self.n_classes = n_classes

    def forward(self, x, training: bool = False):
        for layer in self.layers:
            layer.training = training
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict_proba(self, x, batch: int = 512) -> np.ndarray:
        return np.vstack([softmax(self.forward(x[i:i + batch])) for i in range(0, len(x), batch)])

    def predict(self, x) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def param_layers(self):
        return [layer for layer in self.layers if layer.params]

    def state_dict(self) -> dict:
        return {f"{i}.{k}": v for i, layer in enumerate(self.layers) for k, v in layer.params.items()}

    def load_state_dict(self, state: dict):
        for key, v in state.items():
            i, k = key.split(".")
            self.layers[int(i)].params[k] = v


class Adam:
    def __init__(self, model: CNN, lr: float = 2e-3, betas=(0.9, 0.999), weight_decay: float = 1e-4):
        self.model, self.lr, self.b1, self.b2, self.wd = model, lr, *betas, weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self):
        self.t += 1
        for li, layer in enumerate(self.model.param_layers()):
            for k, p in layer.params.items():
                g = layer.grads[k] + (self.wd * p if k == "W" else 0)
                key = (li, k)
                self.m[key] = self.b1 * self.m.get(key, 0) + (1 - self.b1) * g
                self.v[key] = self.b2 * self.v.get(key, 0) + (1 - self.b2) * g * g
                mh = self.m[key] / (1 - self.b1 ** self.t)
                vh = self.v[key] / (1 - self.b2 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def augment(x: np.ndarray, rng: np.random.Generator, max_shift: int = 1) -> np.ndarray:
    """Random ±1-pixel translations: cheap augmentation that teaches position invariance."""
    out = np.zeros_like(x)
    for i in range(len(x)):
        dy, dx = rng.integers(-max_shift, max_shift + 1, 2)
        out[i] = np.roll(np.roll(x[i], dy, axis=1), dx, axis=2)
        if dy > 0: out[i, :, :dy] = 0  # noqa: E701
        if dy < 0: out[i, :, dy:] = 0  # noqa: E701
        if dx > 0: out[i, :, :, :dx] = 0  # noqa: E701
        if dx < 0: out[i, :, :, dx:] = 0  # noqa: E701
    return out


def train(model: CNN, x_tr, y_tr, x_val, y_val, epochs: int = 30, batch: int = 64, lr: float = 2e-3,
          augment_data: bool = True, seed: int = 0, patience: int = 6, verbose: bool = False) -> dict:
    """Mini-batch training with Adam, augmentation and early stopping on validation loss."""
    rng = np.random.default_rng(seed)
    opt = Adam(model, lr=lr)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        idx = rng.permutation(len(x_tr))
        losses, correct = [], 0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            xb = augment(x_tr[b], rng) if augment_data else x_tr[b]
            logits = model.forward(xb, training=True)
            loss, g = cross_entropy(logits, y_tr[b])
            model.backward(g)
            opt.step()
            losses.append(loss * len(b))
            correct += (logits.argmax(1) == y_tr[b]).sum()
        val_logits = model.forward(x_val)
        vl, _ = cross_entropy(val_logits, y_val)
        hist["train_loss"].append(sum(losses) / len(idx)); hist["train_acc"].append(correct / len(idx))
        hist["val_loss"].append(vl); hist["val_acc"].append(float((val_logits.argmax(1) == y_val).mean()))
        if verbose:
            print(f"epoch {ep + 1:>2}  loss {hist['train_loss'][-1]:.3f}  val_loss {vl:.3f}  val_acc {hist['val_acc'][-1]:.3f}")
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.copy() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val_loss"]) + 1)
    return hist
