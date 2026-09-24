"""
PyTorch Dataset for preprocessed PTB-XL.

Loads cached .npy files (memory-mapped), converts to channels-first tensors,
and applies training-time augmentation.

Usage:
    from src.data.dataset import PTBXLDataset
    from torch.utils.data import DataLoader

    train_ds = PTBXLDataset("train", augment=True)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    for x, y in train_loader:
        print(x.shape)   # (64, 12, 5000)
        print(y.shape)   # (64, 5)
        break
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import get_config


class PTBXLDataset(Dataset):
    """
    Multi-label ECG dataset backed by cached .npy files.

    Args:
        split: one of "train", "val", "test"
        augment: apply training augmentation. True for train, False otherwise.

    Returns per __getitem__:
        signal: torch.Tensor, shape (12, 5000), dtype float32
        label:  torch.Tensor, shape (5,),     dtype float32
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(self, split: str, augment: bool = False):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got {split}")

        cfg = get_config()
        self.cfg = cfg
        self.split = split
        self.augment = augment

        data_dir = cfg.paths.data_processed
        signals_path = data_dir / f"{split}_signals.npy"
        labels_path = data_dir / f"{split}_labels.npy"

        if not signals_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Missing cached data for split '{split}'. "
                f"Run: python -m src.data.preprocess"
            )

        # Memory-map signals (don't load 3.6 GB into RAM)
        # Copy labels into RAM (tiny)
        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.load(labels_path)

        assert self.signals.shape[0] == self.labels.shape[0]

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]      # (5000, 12)
        label = self.labels[idx]        # (5,)

        if self.augment:
            signal = self._augment(signal)

        # Channels-first: (5000, 12) -> (12, 5000)
        signal = np.ascontiguousarray(signal.T)

        return (
            torch.from_numpy(signal).float(),
            torch.from_numpy(label).float(),
        )

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        """Apply noise, scaling, time shift. Each with configured probability."""
        aug = self.cfg.augmentation

        if aug.noise.enabled and np.random.rand() < aug.noise.probability:
            signal = signal + np.random.normal(
                0.0, aug.noise.std, signal.shape
            ).astype(signal.dtype)

        if aug.scaling.enabled and np.random.rand() < aug.scaling.probability:
            scale = np.random.uniform(aug.scaling.min_factor, aug.scaling.max_factor)
            signal = signal * scale

        if aug.time_shift.enabled and np.random.rand() < aug.time_shift.probability:
            max_shift = aug.time_shift.max_shift
            shift = int(np.random.randint(-max_shift, max_shift + 1))
            signal = np.roll(signal, shift, axis=0)

        return signal


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("Loading train dataset (with augmentation)...")
    ds = PTBXLDataset("train", augment=True)
    print(f"  length: {len(ds)}")
    print(f"  single signal shape: {ds[0][0].shape}")
    print(f"  single label shape:  {ds[0][1].shape}")
    print(f"  single signal dtype: {ds[0][0].dtype}")
    print()

    print("DataLoader batch (batch_size=64)...")
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    x, y = next(iter(loader))
    print(f"  batch signal shape: {x.shape}")
    print(f"  batch label shape:  {y.shape}")
    print(f"  batch signal range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  batch label range:  [{y.min():.0f}, {y.max():.0f}]")
    print()

    print("Loading val dataset (no augmentation)...")
    val_ds = PTBXLDataset("val", augment=False)
    print(f"  length: {len(val_ds)}")
    print(f"  sample 0 mean/std: {val_ds[0][0].mean():.4f} / {val_ds[0][0].std():.4f}")
    print()
    print("All checks passed.")
