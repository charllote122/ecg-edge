"""
Convert the FP32 ONNX model to INT8 via dynamic quantization.

Produces models/ecg_model_int8.onnx at ~4x smaller size than FP32.

Usage:
    python -m src.export.quantize
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from src.utils.config import get_config


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    """
    Apply dynamic INT8 quantization to an ONNX model.

    Weights are quantized to INT8. Activations remain FP32 but are quantized
    on-the-fly at inference time.
    """
    print(f"[quantize] input:  {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")
    print(f"[quantize] output: {int8_path}")

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,   # signed INT8 weights
    )

    print(f"[quantize] done. size: {int8_path.stat().st_size / 1e6:.2f} MB")
    reduction = 1 - (int8_path.stat().st_size / fp32_path.stat().st_size)
    print(f"[quantize] size reduction: {reduction:.1%}")


def main() -> None:
    cfg = get_config()
    root = Path(cfg.paths.project_root)

    fp32_path = root / "models" / "ecg_model_fp32.onnx"
    int8_path = root / "models" / "ecg_model_int8.onnx"

    if not fp32_path.exists():
        raise FileNotFoundError(
            f"FP32 ONNX model not found: {fp32_path}\n"
            f"Run: python -m src.export.to_onnx"
        )

    quantize_int8(fp32_path, int8_path)
    print()
    print(f"[done] INT8 model: {int8_path}")


if __name__ == "__main__":
    main()
