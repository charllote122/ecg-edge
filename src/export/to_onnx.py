"""
Export the trained PyTorch model to ONNX.

Produces a framework-agnostic FP32 ONNX model at models/ecg_model_fp32.onnx.

Usage:
    python -m src.export.to_onnx
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.resnet1d import ResNet1D, count_parameters
from src.utils.config import get_config


def export_to_onnx(
    pt_path: Path,
    onnx_path: Path,
    opset: int = 17,
) -> None:
    """Export a PyTorch checkpoint to ONNX with dynamic batch axis."""
    print(f"[export] loading PyTorch model from {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]

    model = ResNet1D(
        in_channels=cfg_dict["in_channels"],
        num_classes=cfg_dict["num_classes"],
        dropout=cfg_dict["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # important: disables dropout at inference time

    print(f"[export] parameters: {count_parameters(model):,}")

    # Dummy input for tracing: (batch=1, channels=12, timesteps=5000)
    dummy = torch.randn(1, cfg_dict["in_channels"], 5000)

    print(f"[export] exporting to {onnx_path} (opset {opset})")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,   # fold constants for optimization
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},   # variable batch
            "output": {0: "batch_size"},
        },
    )
    print(f"[export] done. size: {onnx_path.stat().st_size / 1e6:.2f} MB")


def verify_onnx(onnx_path: Path, pt_path: Path) -> None:
    """
    Verify ONNX model matches PyTorch output within numerical tolerance.
    Loads both, runs random input through both, compares.
    """
    import numpy as np
    import onnxruntime as ort

    print("[verify] loading PyTorch model for comparison")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    model = ResNet1D(
        in_channels=cfg_dict["in_channels"],
        num_classes=cfg_dict["num_classes"],
        dropout=cfg_dict["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[verify] running random input through both models")
    x = np.random.randn(4, cfg_dict["in_channels"], 5000).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"[verify] max abs diff: {max_diff:.6f}")

    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX output differs too much: {max_diff}")
    print("[verify] OK - outputs match")


def main() -> None:
    cfg = get_config()
    root = Path(cfg.paths.project_root)

    pt_path = root / "models" / "best_model.pt"
    onnx_path = root / "models" / "ecg_model_fp32.onnx"

    if not pt_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {pt_path}")

    export_to_onnx(pt_path, onnx_path)
    print()
    verify_onnx(onnx_path, pt_path)
    print()
    print(f"[done] ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
