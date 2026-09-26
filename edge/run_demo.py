"""
Edge demo: process test signals, publish alerts over MQTT.

Simulates what an edge device would do: load signal -> run inference ->
publish alert -> repeat. Run `edge/subscriber.py` in another terminal
to see alerts arrive in real time.

Usage:
    python -m edge.run_demo                    # 10 samples
    python -m edge.run_demo --n 3              # 3 samples
    python -m edge.run_demo --device-id pi-01  # custom device ID
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from edge.edge_inference import EdgeInference
from edge.mqtt_publisher import MQTTPublisher
from src.data.labels import SUPERCLASSES
from src.utils.config import get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="number of test samples to process")
    parser.add_argument("--device-id", type=str, default="edge-001",
                        help="MQTT device identifier")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds between samples")
    args = parser.parse_args()

    cfg = get_config()
    data_dir = cfg.paths.data_processed

    print("[demo] loading test signals...")
    signals = np.load(data_dir / "test_signals.npy", mmap_mode="r")
    labels = np.load(data_dir / "test_labels.npy")
    n = min(args.n, len(signals))
    print(f"[demo] processing {n} of {len(signals)} test samples")
    print()

    print("[demo] initializing edge inference engine...")
    engine = EdgeInference()
    print()

    publisher = MQTTPublisher(device_id=args.device_id)
    publisher.connect()
    print()

    print(f"[demo] running {n} samples...")
    print("=" * 72)
    for i in range(n):
        signal = signals[i]
        true_label = labels[i]
        true_classes = [SUPERCLASSES[j] for j in range(len(SUPERCLASSES))
                        if true_label[j] > 0.5]

        result = engine.predict(signal)
        publisher.publish(result)

        pred_str = f"{result['top_class']}({result['top_prob']:.2f})"
        true_str = ",".join(true_classes) if true_classes else "NONE"
        match = "OK " if result["top_class"] in true_classes else "MISS"

        print(
            f"  [{i+1:02d}/{n:02d}] {match}  "
            f"true={true_str:<14} pred={pred_str:<14} "
            f"{result['latency_ms']:.0f}ms"
        )

        if i < n - 1:
            time.sleep(args.delay)

    print("=" * 72)
    print(f"[demo] sent {n} alerts to topic: {publisher.topic}")
    publisher.disconnect()
    print("[demo] complete")


if __name__ == "__main__":
    main()
