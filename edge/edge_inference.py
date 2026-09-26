"""
Edge inference: load ONNX INT8 model, run inference on a single signal.

Simulates what would run on a Raspberry Pi / Jetson Nano.

Usage:
    from edge.edge_inference import EdgeInference
    engine = EdgeInference()
    result = engine.predict(signal)  # signal shape (12, 5000)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.data.labels import SUPERCLASSES
from src.utils.config import get_config


class EdgeInference:
    """
    ONNX Runtime inference wrapper for the ECG edge model.

    Runs on CPU with the INT8-quantized model. Would work identically on
    ARM hardware (Raspberry Pi, Jetson) with ONNX Runtime ARM builds.
    """

    def __init__(self, model_path: Path | str | None = None):
        cfg = get_config()
        if model_path is None:
            model_path = Path(cfg.paths.project_root) / "models" / "ecg_model_int8.onnx"

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                f"Run: python -m src.export.quantize"
            )

        self.model_path = model_path
        # Single-threaded inference — matches real edge device constraints
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = cfg.evaluation.threshold

    def predict(self, signal: np.ndarray) -> dict:
        """
        Run inference on a single signal.

        Args:
            signal: (12, 5000) float32 array, already preprocessed.

        Returns:
            {
                "predictions": {class_name: probability, ...},
                "detected": [class_name, ...],       # above threshold
                "top_class": str,                    # highest probability
                "top_prob": float,
                "latency_ms": float,
            }
        """
        # Ensure channels-first and float32
        if signal.shape == (5000, 12):
            signal = signal.T
        signal = signal[np.newaxis, ...].astype(np.float32)  # (1, 12, 5000)

        # Time the inference
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: signal})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Sigmoid → probabilities
        probs = 1.0 / (1.0 + np.exp(-logits[0]))  # (5,)

        predictions = {name: float(p) for name, p in zip(SUPERCLASSES, probs)}
        detected = [name for name, p in predictions.items() if p > self.threshold]
        top_class = max(predictions, key=predictions.get)

        return {
            "predictions": predictions,
            "detected": detected,
            "top_class": top_class,
            "top_prob": float(predictions[top_class]),
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    # Smoke test: run inference on the first test sample
    from src.utils.config import get_config

    cfg = get_config()
    data_dir = cfg.paths.data_processed

    print("Loading edge inference engine...")
    engine = EdgeInference()
    print(f"  model: {engine.model_path}")
    print()

    print("Loading one test signal...")
    signals = np.load(data_dir / "test_signals.npy", mmap_mode="r")
    labels = np.load(data_dir / "test_labels.npy")
    signal = signals[0]     # (5000, 12)
    true_label = labels[0]  # (5,)

    print(f"  shape: {signal.shape}")
    print(f"  true labels: {dict(zip(SUPERCLASSES, true_label.astype(int)))}")
    print()

    result = engine.predict(signal)
    print("Prediction:")
    print(f"  top class: {result['top_class']} ({result['top_prob']:.3f})")
    print(f"  detected:  {result['detected']}")
    print(f"  latency:   {result['latency_ms']:.2f} ms")
    print()
    print("All class probabilities:")
    for name, p in result["predictions"].items():
        marker = " ←" if name in result["detected"] else ""
        print(f"  {name:<6} {p:.4f}{marker}")
    print()
    print("Smoke test passed.")
