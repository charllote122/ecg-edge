"""
Training loop for multi-label ECG classification.

Handles:
    - class-weighted BCE loss
    - Adam optimizer + OneCycle LR schedule
    - per-epoch validation
    - best-model checkpointing
    - early stopping
    - training history (JSON)

Usage:
    python -m src.training.train
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PTBXLDataset
from src.models.resnet1d import ResNet1D, count_parameters
from src.utils.config import get_config


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Class weights for imbalance
# -----------------------------------------------------------------------------

def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute positive class weights for BCEWithLogitsLoss.

    pos_weight[i] = (number of negatives for class i) / (number of positives)

    Higher weight = more penalty for missing that class.
    """
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    # Avoid division by zero on classes with no positives
    pos = np.maximum(pos, 1)
    weights = neg / pos
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Train / validate for one epoch
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []

    for x, y in tqdm(loader, desc="  train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    avg_loss = total_loss / max(n_batches, 1)

    # Compute macro F1
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))       # sigmoid
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "macro_f1": macro_f1}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []

    for x, y in tqdm(loader, desc="  val  ", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "macro_f1": macro_f1}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = get_config()
    set_seed(cfg.training.seed)

    # --- Paths ---
    models_dir = Path(cfg.paths.project_root) / "models"
    logs_dir = Path(cfg.paths.project_root) / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.pt"
    history_path = logs_dir / "train_history.json"

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # --- Data ---
    train_ds = PTBXLDataset("train", augment=True)
    val_ds = PTBXLDataset("val", augment=False)

    # num_workers=0 on Windows to avoid multiprocessing issues
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"[train] train: {len(train_ds)} | val: {len(val_ds)}")

    # --- Class weights from training labels ---
    pos_weights = compute_pos_weights(train_ds.labels).to(device)
    print(f"[train] pos_weights: {pos_weights.cpu().numpy().round(2)}")

    # --- Model ---
    model = ResNet1D(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    ).to(device)
    print(f"[train] parameters: {count_parameters(model):,}")

    # --- Loss / Optimizer / Scheduler ---
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        steps_per_epoch=len(train_loader),
    )

    # --- Training loop ---
    best_val_f1 = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        print(f"\n[train] Epoch {epoch}/{cfg.training.epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, grad_clip=cfg.training.grad_clip,
        )
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(
            f"  train loss={train_metrics['loss']:.4f} "
            f"f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"f1={val_metrics['macro_f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
        })

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "config": {
                    "in_channels": cfg.model.in_channels,
                    "num_classes": cfg.model.num_classes,
                    "dropout": cfg.model.dropout,
                },
            }, best_model_path)
            print(f"  [save] new best val F1 = {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= cfg.training.early_stopping_patience:
            print(f"[train] early stopping at epoch {epoch}")
            break

    # --- Save history ---
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[train] done. best val F1: {best_val_f1:.4f}")
    print(f"[train] best model: {best_model_path}")
    print(f"[train] history: {history_path}")


if __name__ == "__main__":
    main()
