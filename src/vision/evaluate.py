"""Train and compare image classifiers on handwritten digits, with robustness tests and explanations.

Usage:  python -m vision.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from vision.numpy_cnn import CNN, train  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "reports" / "figures"


def load_data(seed: int = 0):
    d = load_digits()
    X = (d.images / 16.0)[:, None].astype(np.float64)  # (N, 1, 8, 8), scaled to [0, 1]
    y = d.target
    x_tmp, x_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=seed)
    return x_tr, y_tr, x_val, y_val, x_te, y_te


def shift(x, dy, dx):
    out = np.roll(np.roll(x, dy, axis=2), dx, axis=3)
    if dy > 0: out[:, :, :dy] = 0  # noqa: E701
    if dy < 0: out[:, :, dy:] = 0  # noqa: E701
    if dx > 0: out[:, :, :, :dx] = 0  # noqa: E701
    if dx < 0: out[:, :, :, dx:] = 0  # noqa: E701
    return out


def corruptions(x, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dirs = rng.integers(0, 4, len(x))
    moved = np.concatenate([shift(x[i:i + 1], *[(1, 0), (-1, 0), (0, 1), (0, -1)][d]) for i, d in enumerate(dirs)])
    return {"clean": x, "shifted 1px": moved, "gaussian noise σ=0.2": np.clip(x + rng.normal(0, 0.2, x.shape), 0, 1)}


def ece(y, p, bins: int = 10) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum(((conf > lo) & (conf <= hi)).mean() * abs((pred == y)[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
                     for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any()))


def saliency(model: CNN, x: np.ndarray, cls: int) -> np.ndarray:
    """Gradient of the class score w.r.t. the input pixels: which pixels drive the decision."""
    logits = model.forward(x[None])
    g = np.zeros_like(logits)
    g[0, cls] = 1
    return np.abs(model.backward(g)[0, 0])


def main(epochs: int = 60) -> dict:
    x_tr, y_tr, x_val, y_val, x_te, y_te = load_data()
    flat = lambda a: a.reshape(len(a), -1)  # noqa: E731

    models = {
        "Logistic regression (pixels)": LogisticRegression(max_iter=3000, C=1.0).fit(flat(x_tr), y_tr),
        "MLP (1 hidden layer, sklearn)": MLPClassifier((128,), max_iter=600, random_state=0).fit(flat(x_tr), y_tr),
    }
    cnn_plain = CNN(seed=0)
    hist_plain = train(cnn_plain, x_tr, y_tr, x_val, y_val, epochs=epochs, augment_data=False)
    cnn = CNN(seed=0)
    hist = train(cnn, x_tr, y_tr, x_val, y_val, epochs=epochs, augment_data=True)

    def proba(name, x):
        if name.startswith("NumPy CNN"):
            return (cnn if name.endswith("+ augmentation") else cnn_plain).predict_proba(x)
        return models[name].predict_proba(flat(x))

    names = list(models) + ["NumPy CNN (no augmentation)", "NumPy CNN + augmentation"]
    tests = corruptions(x_te)
    results = {}
    for name in names:
        p = proba(name, x_te)
        results[name] = {"accuracy": round(accuracy_score(y_te, p.argmax(1)), 4), "macro_f1": round(f1_score(y_te, p.argmax(1), average="macro"), 4),
                         "log_loss": round(log_loss(y_te, p, labels=range(10)), 4), "ece": round(ece(y_te, p), 4),
                         **{f"acc_{k}": round(accuracy_score(y_te, proba(name, v).argmax(1)), 4) for k, v in tests.items() if k != "clean"}}

    out = {"data": {"dataset": "UCI/sklearn handwritten digits (8×8 greyscale)", "train": len(x_tr), "validation": len(x_val), "test": len(x_te)},
           "cnn": {"parameters": int(sum(v.size for v in cnn.state_dict().values())), "best_epoch": hist["best_epoch"],
                   "epochs_run": len(hist["val_loss"])},
           "results": results}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(out, indent=2))
    (ROOT / "models").mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "models" / "numpy_cnn_digits.npz", **cnn.state_dict())
    _plots(cnn, hist, hist_plain, x_te, y_te)
    return out


def _plots(cnn, hist, hist_plain, x_te, y_te):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for h, lab in [(hist, "with augmentation"), (hist_plain, "no augmentation")]:
        axes[0].plot(h["train_loss"], label=f"train ({lab})"); axes[0].plot(h["val_loss"], ls="--", label=f"val ({lab})")
        axes[1].plot(h["val_acc"], label=lab)
    axes[0].set(title="Loss", xlabel="epoch"); axes[1].set(title="Validation accuracy", xlabel="epoch")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGS / "learning_curves.png", dpi=130); plt.close(fig)

    pred = cnn.predict(x_te)
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="predicted", ylabel="true", title="NumPy CNN: test confusion matrix", xticks=range(10), yticks=range(10))
    fig.tight_layout(); fig.savefig(FIGS / "confusion_matrix.png", dpi=130); plt.close(fig)

    wrong = np.where(pred != y_te)[0][:8]
    probs = cnn.predict_proba(x_te)
    fig, axes = plt.subplots(2, max(len(wrong), 1), figsize=(1.6 * max(len(wrong), 1), 3.6))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for a, i in zip(axes.T, wrong):
        a[0].imshow(x_te[i, 0], cmap="gray_r"); a[0].set_title(f"true {y_te[i]}\npred {pred[i]} ({probs[i].max():.0%})", fontsize=8)
        a[1].imshow(saliency(cnn, x_te[i], pred[i]), cmap="hot"); a[1].set_title("saliency", fontsize=8)
        for b in a: b.axis("off")
    fig.suptitle("Misclassified test digits and what the CNN looked at", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "errors_saliency.png", dpi=130); plt.close(fig)

    W = cnn.layers[0].params["W"]
    fig, axes = plt.subplots(2, 8, figsize=(9, 2.6))
    for a, w in zip(axes.ravel(), W):
        a.imshow(w[0], cmap="RdBu_r"); a.axis("off")
    fig.suptitle("Learned first-layer 3×3 filters (edge and stroke detectors)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / "conv_filters.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
