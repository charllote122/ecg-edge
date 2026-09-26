"""
Evaluate the trained model on the held-out test set.

Computes:
    - macro F1 (primary metric)
    - per-class F1, precision, recall
    - per-class ROC AUC
    - confusion matrices (per class, multi-label)

Saves:
    - results/metrics.json
    - results/confusion_matrices.png
    - results/roc_curves.png

Usage:
    python -m src.training.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PTBXLDataset
from src.data.labels import SUPERCLASSES
from src.models.resnet1d import ResNet1D, count_parameters
from src.utils.config import get_config


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Run model on loader, return (probabilities, labels) as numpy arrays."""
    model.eval()
    all_probs = []
    all_labels = []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all evaluation metrics."""
    preds = (probs > threshold).astype(int)
    metrics = {}

    # Overall
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(labels, preds, average="micro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))

    # Per-class
    per_class = {}
    for i, name in enumerate(SUPERCLASSES):
        y_true = labels[:, i]
        y_pred = preds[:, i]
        y_prob = probs[:, i]

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")   # class has only one label in test set

        per_class[name] = {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "auc": auc,
            "support": int(y_true.sum()),
        }
    metrics["per_class"] = per_class
    return metrics


def plot_confusion_matrices(labels: np.ndarray, probs: np.ndarray,
                            threshold: float, out_path: Path) -> None:
    """Plot one confusion matrix per class."""
    preds = (probs > threshold).astype(int)
    fig, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(4 * len(SUPERCLASSES), 4))
    if len(SUPERCLASSES) == 1:
        axes = [axes]

    for i, (name, ax) in enumerate(zip(SUPERCLASSES, axes)):
        cm = confusion_matrix(labels[:, i], preds[:, i])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"])
        ax.set_yticklabels(["neg", "pos"])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_roc_curves(labels: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    """Plot ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(SUPERCLASSES):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Test Set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def main() -> None:
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    # --- Load model ---
    model_path = Path(cfg.paths.project_root) / "models" / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Did you run training and download the model?"
        )
    print(f"[eval] loading model from {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = ResNet1D(
        in_channels=ckpt["config"]["in_channels"],
        num_classes=ckpt["config"]["num_classes"],
        dropout=ckpt["config"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[eval] parameters: {count_parameters(model):,}")
    print(f"[eval] trained to epoch {ckpt['epoch']}, val F1 = {ckpt['val_f1']:.4f}")

    # --- Load test data ---
    test_ds = PTBXLDataset("test", augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.dataloader.batch_size,
        shuffle=False, num_workers=0,
    )
    print(f"[eval] test samples: {len(test_ds)}")

    # --- Predict ---
    probs, labels = predict(model, test_loader, device)

    # --- Metrics ---
    threshold = cfg.evaluation.threshold
    metrics = compute_metrics(probs, labels, threshold=threshold)

    # --- Save reports ---
    results_dir = Path(cfg.paths.project_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    cm_path = results_dir / "confusion_matrices.png"
    plot_confusion_matrices(labels, probs, threshold, cm_path)

    roc_path = results_dir / "roc_curves.png"
    plot_roc_curves(labels, probs, roc_path)

    # --- Print summary ---
    print()
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print()
    print(f"{'Class':<8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8} {'Support':>8}")
    print("-" * 60)
    for name, m in metrics["per_class"].items():
        print(
            f"{name:<8} {m['f1']:>8.4f} {m['precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['auc']:>8.4f} {m['support']:>8d}"
        )
    print()
    print(f"Reports saved to {results_dir}/")


if __name__ == "__main__":
    main()
