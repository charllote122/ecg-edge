"""
Preprocess PTB-XL records: load WFDB, filter, normalize, cache to .npy.

Pipeline:
    raw WFDB file (.dat + .hea)
        -> wfdb.rdrecord
    (5000, 12) float64 array
        -> handle NaNs
        -> bandpass filter (0.5-40 Hz, zero-phase)
        -> per-channel z-normalization
        -> pad/truncate to exactly (5000, 12)
    (5000, 12) float32 array
        -> save
    data/processed/{split}_signals.npy + {split}_labels.npy

Usage:
    # Run full pipeline (train / val / test)
    python -m src.data.preprocess

    # Or use as a library
    from src.data.preprocess import load_and_preprocess
    signal = load_and_preprocess("data/raw/ptb-xl/records500/00000/00001_hr")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from src.data.labels import SUPERCLASSES, load_metadata
from src.utils.config import get_config


# -----------------------------------------------------------------------------
# Signal processing primitives
# -----------------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: int,
    low: float,
    high: float,
    order: int,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Uses second-order-sections (SOS) form for numerical stability at higher
    orders, and sosfiltfilt for zero phase distortion (forward + backward pass).
    Zero-phase matters because ECG morphology -- the QRS complex, T-wave peak --
    must stay aligned with the time axis for downstream analysis.

    Args:
        signal: (T, C) array. T = timesteps, C = channels.
        sampling_rate: Hz.
        low: lower cutoff in Hz.
        high: upper cutoff in Hz.
        order: filter order.

    Returns:
        Filtered array, same shape.
    """
    nyquist = sampling_rate / 2.0
    sos = butter(
        order,
        [low / nyquist, high / nyquist],
        btype="bandpass",
        output="sos",
    )
    # axis=0 because time is the first axis
    return sosfiltfilt(sos, signal, axis=0)


def z_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Per-channel z-score normalization: (x - mean) / std.

    Per-channel (not global) because each of the 12 leads has its own natural
    amplitude scale. Global normalization would destroy the inter-lead
    relationships that carry diagnostic information.
    """
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    # Avoid division by zero on flat channels
    std = np.where(std < 1e-8, 1.0, std)
    return (signal - mean) / std


# -----------------------------------------------------------------------------
# Single-record pipeline
# -----------------------------------------------------------------------------

def load_and_preprocess(record_path: str | Path) -> np.ndarray:
    """
    Load a single WFDB record and return a preprocessed (T, C) float32 array.

    Args:
        record_path: Path to the WFDB record WITHOUT the .hea or .dat extension.
            Example: "data/raw/ptb-xl/records500/00000/00001_hr"

    Returns:
        (signal_length, 12) float32 array.
    """
    cfg = get_config()
    signal_length = cfg.dataset.signal_length
    num_channels = cfg.dataset.num_channels

    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal  # (T, 12), float64, may contain NaN

    # --- Handle NaNs -----------------------------------------------------
    # PTB-XL has occasional NaN samples. Replace with 0 (mean-centered by
    # the subsequent z-normalization step).
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal, nan=0.0)

    # --- Filter ----------------------------------------------------------
    signal = bandpass_filter(
        signal,
        sampling_rate=cfg.dataset.sampling_rate,
        low=cfg.preprocessing.bandpass_low,
        high=cfg.preprocessing.bandpass_high,
        order=cfg.preprocessing.filter_order,
    )

    # --- Normalize -------------------------------------------------------
    if cfg.preprocessing.normalize == "zscore":
        signal = z_normalize(signal)

    # --- Force exact length ---------------------------------------------
    if signal.shape[0] < signal_length:
        pad = signal_length - signal.shape[0]
        signal = np.pad(signal, ((0, pad), (0, 0)), mode="edge")
    elif signal.shape[0] > signal_length:
        signal = signal[:signal_length]

    # --- Force exact channel count --------------------------------------
    if signal.shape[1] != num_channels:
        raise ValueError(
            f"Expected {num_channels} channels, got {signal.shape[1]} "
            f"for record {record_path}"
        )

    return signal.astype(np.float32)


# -----------------------------------------------------------------------------
# Full split pipeline
# -----------------------------------------------------------------------------

def preprocess_split(
    df,
    split_name: str,
    output_dir: Path,
) -> None:
    """
    Preprocess all records in a split and save signals + labels as .npy.

    Args:
        df: DataFrame from load_metadata(), filtered to one split.
        split_name: "train", "val", or "test".
        output_dir: Where to write the .npy files.
    """
    cfg = get_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(df)
    signal_length = cfg.dataset.signal_length
    num_channels = cfg.dataset.num_channels

    # Preallocate for speed and predictable memory
    signals = np.zeros((n, signal_length, num_channels), dtype=np.float32)
    labels = np.zeros((n, len(SUPERCLASSES)), dtype=np.float32)
    ecg_ids = np.zeros(n, dtype=np.int64)

    failed = 0
    written = 0
    for ecg_id, row in tqdm(df.iterrows(), total=n, desc=split_name):
        rec_path = cfg.paths.data_raw / row["filename_hr"]
        try:
            sig = load_and_preprocess(rec_path)
        except Exception as exc:
            failed += 1
            print(f"[warn] failed {ecg_id}: {exc}")
            continue

        signals[written] = sig
        labels[written] = row[SUPERCLASSES].values.astype(np.float32)
        ecg_ids[written] = ecg_id
        written += 1

    # Trim to actually-written rows (in case of failures)
    signals = signals[:written]
    labels = labels[:written]
    ecg_ids = ecg_ids[:written]

    np.save(output_dir / f"{split_name}_signals.npy", signals)
    np.save(output_dir / f"{split_name}_labels.npy", labels)
    np.save(output_dir / f"{split_name}_ids.npy", ecg_ids)

    print(
        f"[{split_name}] saved {written} records "
        f"({failed} failed) -> {output_dir}"
    )


def preprocess_all() -> None:
    """
    Load metadata, split by official strat_fold, preprocess all three splits.
    """
    cfg = get_config()
    output_dir = cfg.paths.data_processed
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[preprocess] loading metadata...")
    df = load_metadata()

    # Official PTB-XL split from Wagner et al. (2020):
    #   strat_fold 1-8 -> train
    #   strat_fold 9   -> validation
    #   strat_fold 10  -> test
    train_df = df[df["strat_fold"] <= cfg.dataset.fold_train_max]
    val_df = df[df["strat_fold"] == cfg.dataset.fold_val]
    test_df = df[df["strat_fold"] == cfg.dataset.fold_test]

    print(
        f"[preprocess] split sizes -- "
        f"train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}"
    )

    preprocess_split(train_df, "train", output_dir)
    preprocess_split(val_df, "val", output_dir)
    preprocess_split(test_df, "test", output_dir)

    print("[preprocess] done.")


if __name__ == "__main__":
    preprocess_all()
