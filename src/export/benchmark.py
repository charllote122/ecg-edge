"""
Benchmark: PyTorch FP32 vs ONNX FP32 vs ONNX INT8 on the test set.

Compares model size, inference latency, and test macro F1.

Saves results to results/quantization_benchmark.json.

Usage:
    python -m src.export.benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PTBXLDataset
from src.data.labels import SUPERCLASSES
from src.models.resnet1d import ResNet1D
from src.utils.config import get_config


# -----------------------------------------------------------------------------
# Predictions
# -----------------------------------------------------------------------------

@torch.no_grad()
def predict_pytorch(model_path: Path, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    """Run PyTorch model on loader. Returns (probs, labels, elapsed_seconds)."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = ResNet1D(
        in_channels=c["in_channels"],
        num_classes=c["num_classes"],
        dropout=c["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs, all_labels = [], []
    t0 = time.perf_counter()
    for x, y in tqdm(loader, desc="  pytorch-fp32", leave=False):
        logits = model(x)
        all_probs.append(torch.sigmoid(logits).numpy())
        all_labels.append(y.numpy())
    elapsed = time.perf_counter() - t0

    return np.concatenate(all_probs), np.concatenate(all_labels), elapsed


def predict_onnx(onnx_path: Path, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    """Run ONNX model on loader. Returns (probs, labels, elapsed_seconds)."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    all_probs, all_labels = [], []
    t0 = time.perf_counter()
    for x, y in tqdm(loader, desc=f"  onnx-{onnx_path.stem}", leave=False):
        x_np = x.numpy().astype(np.float32)
        logits = sess.run(None, {input_name: x_np})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_probs.append(probs)
        all_labels.append(y.numpy())
    elapsed = time.perf_counter() - t0

    return np.concatenate(all_probs), np.concatenate(all_labels), elapsed


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs > threshold).astype(int)
    macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    per_class = {}
    for i, name in enumerate(SUPERCLASSES):
        per_class[name] = float(f1_score(labels[:, i], preds[:, i], zero_division=0))
    return {"macro": macro, "per_class": per_class}


def main() -> None:
    cfg = get_config()
    root = Path(cfg.paths.project_root)

    pt_path = root / "models" / "best_model.pt"
    fp32_path = root / "models" / "ecg_model_fp32.onnx"
    int8_path = root / "models" / "ecg_model_int8.onnx"

    for p in [pt_path, fp32_path, int8_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # Load test data
    test_ds = PTBXLDataset("test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    n_samples = len(test_ds)
    print(f"[benchmark] test samples: {n_samples}")

    results = {}

    # --- PyTorch FP32 ---
    print("\n[benchmark] PyTorch FP32")
    probs, labels, elapsed = predict_pytorch(pt_path, test_loader)
    f1s = compute_f1(probs, labels, cfg.evaluation.threshold)
    results["pytorch_fp32"] = {
        "size_mb": pt_path.stat().st_size / 1e6,
        "total_seconds": elapsed,
        "ms_per_sample": 1000 * elapsed / n_samples,
        "samples_per_sec": n_samples / elapsed,
        "macro_f1": f1s["macro"],
        "per_class_f1": f1s["per_class"],
    }
    print(f"  macro F1 = {f1s['macro']:.4f} | {1000 * elapsed / n_samples:.2f} ms/sample")

    # --- ONNX FP32 ---
    print("\n[benchmark] ONNX FP32")
    probs, labels, elapsed = predict_onnx(fp32_path, test_loader)
    f1s = compute_f1(probs, labels, cfg.evaluation.threshold)
    results["onnx_fp32"] = {
        "size_mb": fp32_path.stat().st_size / 1e6,
        "total_seconds": elapsed,
        "ms_per_sample": 1000 * elapsed / n_samples,
        "samples_per_sec": n_samples / elapsed,
        "macro_f1": f1s["macro"],
        "per_class_f1": f1s["per_class"],
    }
    print(f"  macro F1 = {f1s['macro']:.4f} | {1000 * elapsed / n_samples:.2f} ms/sample")

    # --- ONNX INT8 ---
    print("\n[benchmark] ONNX INT8")
    probs, labels, elapsed = predict_onnx(int8_path, test_loader)
    f1s = compute_f1(probs, labels, cfg.evaluation.threshold)
    results["onnx_int8"] = {
        "size_mb": int8_path.stat().st_size / 1e6,
        "total_seconds": elapsed,
        "ms_per_sample": 1000 * elapsed / n_samples,
        "samples_per_sec": n_samples / elapsed,
        "macro_f1": f1s["macro"],
        "per_class_f1": f1s["per_class"],
    }
    print(f"  macro F1 = {f1s['macro']:.4f} | {1000 * elapsed / n_samples:.2f} ms/sample")

    # --- Save report ---
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "quantization_benchmark.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    # --- Summary table ---
    print()
    print("=" * 78)
    print(f"{'Model':<20} {'Size (MB)':>12} {'Latency (ms)':>15} {'Samples/s':>12} {'Macro F1':>12}")
    print("-" * 78)
    for name, r in results.items():
        print(
            f"{name:<20} {r['size_mb']:>12.2f} {r['ms_per_sample']:>15.2f} "
            f"{r['samples_per_sec']:>12.1f} {r['macro_f1']:>12.4f}"
        )
    print("=" * 78)
    print(f"\n[benchmark] report saved to {out_path}")


if __name__ == "__main__":
    main()
