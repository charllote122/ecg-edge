"""
MQTT subscriber for ECG edge alerts.

Listens on ecg-edge/+/predictions and prints alerts as they arrive.

In a real deployment, this would be a cloud service that stores alerts
in a database, triggers notifications, or feeds a dashboard.

Usage:
    python -m edge.subscriber
"""

from __future__ import annotations

import json

import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ecg-edge/+/predictions"


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"[subscriber] connected to {BROKER}:{PORT}")
        client.subscribe(TOPIC, qos=1)
        print(f"[subscriber] subscribed to {TOPIC}")
        print("[subscriber] listening for alerts... (Ctrl+C to stop)")
        print()
    else:
        print(f"[subscriber] connection failed: rc={reason_code}")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        device = data.get("device_id", "?")
        top = data.get("top_class", "?")
        prob = data.get("top_prob", 0.0)
        detected = data.get("detected", [])
        ts = data.get("timestamp", "")

        detected_str = ",".join(detected) if detected else "-"
        print(
            f"[ALERT] {ts}  device={device:<10}  "
            f"top={top}({prob:.3f})  detected=[{detected_str}]"
        )
    except Exception as e:
        print(f"[subscriber] failed to parse message: {e}")


def main() -> None:
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ecg-edge-subscriber",
    )
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[subscriber] connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print()
        print("[subscriber] stopping...")
        client.disconnect()


if __name__ == "__main__":
    main()
