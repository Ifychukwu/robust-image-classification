"""Transfer learning on real photos: fine-tune a pretrained ResNet-18 (PyTorch) on EuroSAT.

EuroSAT: 27,000 Sentinel-2 satellite images (64×64 RGB), 10 land-use classes:
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake. torchvision downloads it automatically.

Use any other dataset arranged as ``root/<class_name>/*.jpg`` with ``--data folder --root path``.

Usage (GPU recommended; free on Google Colab, see notebooks/transfer_learning_colab.ipynb):
    python -m vision.transfer_learning --epochs 5
    python -m vision.transfer_learning --data folder --root /path/to/images --epochs 8
Writes models/resnet18_<data>.pt and reports/transfer_metrics_<data>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms(size: int = 224):
    from torchvision import transforms as T
    train = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),  # satellite images have no "up"
                       T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    test = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train, test


def load_datasets(data: str, root: str | None, seed: int = 42):
    """Stratified-ish 70/15/15 split by random permutation; train split gets augmentation."""
    import torch
    from torchvision import datasets

    train_tf, test_tf = build_transforms()
    if data == "eurosat":
        base = datasets.EuroSAT(root=root or str(ROOT / "data"), download=True)
    else:
        base = datasets.ImageFolder(root)
    classes = base.classes
    n = len(base)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    a, b = int(0.7 * n), int(0.85 * n)

    class Subset(torch.utils.data.Dataset):
        def __init__(self, indices, tf):
            self.indices, self.tf = indices, tf

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            img, y = base[self.indices[i]]
            return self.tf(img.convert("RGB")), y

    return Subset(idx[:a], train_tf), Subset(idx[a:b], test_tf), Subset(idx[b:], test_tf), classes


def build_model(n_classes: int, freeze_backbone: bool = False):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)  # new classification head
    return model


def evaluate_loader(model, loader, device):
    import torch

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def main(data: str = "eurosat", root: str | None = None, epochs: int = 5, batch_size: int = 64, lr: float = 1e-3,
         freeze_backbone: bool = False, num_workers: int = 2) -> dict:
    import torch
    from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score

    from vision.evaluate import ece

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train_ds, val_ds, test_ds, classes = load_datasets(data, root)
    dl = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,  # noqa: E731
                                                        num_workers=num_workers, pin_memory=device == "cuda")
    train_dl, val_dl, test_dl = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    model = build_model(len(classes), freeze_backbone).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    history, best, best_state = [], 0.0, None
    for ep in range(epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        yv, pv = evaluate_loader(model, val_dl, device)
        val_acc = accuracy_score(yv, pv.argmax(1))
        history.append({"epoch": ep + 1, "train_loss": total / n, "val_acc": val_acc, "seconds": round(time.time() - t0, 1)})
        print(history[-1])
        if val_acc > best:
            best, best_state = val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    yt, pt = evaluate_loader(model, test_dl, device)
    results = {
        "data": data, "classes": classes, "train/val/test": [len(train_ds), len(val_ds), len(test_ds)],
        "model": "ResNet-18 (ImageNet-pretrained), fine-tuned" + (" head only" if freeze_backbone else ""),
        "history": history,
        "test": {"accuracy": round(accuracy_score(yt, pt.argmax(1)), 4), "macro_f1": round(f1_score(yt, pt.argmax(1), average="macro"), 4),
                 "top3_accuracy": round(top_k_accuracy_score(yt, pt, k=3, labels=range(len(classes))), 4), "ece": round(ece(yt, pt), 4),
                 "per_class": classification_report(yt, pt.argmax(1), target_names=classes, output_dict=True)},
    }
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, ROOT / "models" / f"resnet18_{data}.pt")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"transfer_metrics_{data}.json").write_text(json.dumps(results, indent=2))
    return results


def grad_cam(model, image_tensor, target_class: int | None = None) -> np.ndarray:
    """Grad-CAM heat-map from ResNet's last conv block: where the network looked (explainability)."""
    import torch

    acts, grads = {}, {}
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault("g", go[0]))
    model.eval()
    out = model(image_tensor[None])
    cls = int(out.argmax()) if target_class is None else target_class
    model.zero_grad()
    out[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eurosat", choices=["eurosat", "folder"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze-backbone", action="store_true")
    a = ap.parse_args()
    r = main(a.data, a.root, a.epochs, a.batch_size, a.lr, a.freeze_backbone)
    print(json.dumps(r["test"], indent=2)[:600])
