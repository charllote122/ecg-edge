"""
MQTT publisher for ECG edge alerts.

Publishes prediction results to a public MQTT broker. Only classification
results are transmitted -- never the raw signal (privacy, bandwidth, power).

Usage:
    from edge.mqtt_publisher import MQTTPublisher
    pub = MQTTPublisher(device_id="edge-001")
    pub.connect()
    pub.publish(result)
    pub.disconnect()
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


class MQTTPublisher:
    """
    MQTT publisher for edge alerts.

    Connects to a public broker and publishes JSON alerts to
    ecg-edge/{device_id}/predictions.
    """

    def __init__(
        self,
        device_id: str = "edge-001",
        broker: str = "test.mosquitto.org",
        port: int = 1883,
        topic_prefix: str = "ecg-edge",
    ):
        self.device_id = device_id
        self.broker = broker
        self.port = port
        self.topic = f"{topic_prefix}/{device_id}/predictions"

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=device_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self._connected = False

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self._connected = True
            print(f"[mqtt] connected to {self.broker}:{self.port}")
            print(f"[mqtt] topic: {self.topic}")
        else:
            print(f"[mqtt] connection failed: rc={reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self._connected = False
        print(f"[mqtt] disconnected (rc={reason_code})")

    def _on_publish(self, client, userdata, mid, reason_code=None, properties=None):
        # Silent — uncomment if you want to see each publish
        # print(f"[mqtt] message {mid} delivered")
        pass

    def connect(self, timeout: float = 5.0) -> None:
        """Connect to the broker and wait for the connection to establish."""
        print(f"[mqtt] connecting to {self.broker}:{self.port}...")
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()   # background thread for network I/O

        # Wait for connection
        t0 = time.time()
        while not self._connected and time.time() - t0 < timeout:
            time.sleep(0.1)

        if not self._connected:
            raise ConnectionError(
                f"Could not connect to {self.broker}:{self.port} within {timeout}s"
            )

    def publish(self, result: dict) -> None:
        """
        Publish an inference result.

        Args:
            result: dict with keys top_class, top_prob, detected, latency_ms.
        """
        message = {
            "device_id": self.device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "top_class": result["top_class"],
            "top_prob": round(result["top_prob"], 4),
            "detected": result["detected"],
            "latency_ms": round(result["latency_ms"], 2),
        }
        payload = json.dumps(message)
        info = self.client.publish(self.topic, payload, qos=1)
        info.wait_for_publish(timeout=5.0)

    def disconnect(self) -> None:
        """Clean shutdown."""
        self.client.loop_stop()
        self.client.disconnect()


if __name__ == "__main__":
    # Smoke test: connect, publish a fake result, disconnect
    pub = MQTTPublisher(device_id="test-device")
    pub.connect()

    fake_result = {
        "top_class": "NORM",
        "top_prob": 0.982,
        "detected": ["NORM"],
        "latency_ms": 123.4,
    }
    pub.publish(fake_result)
    print(f"[mqtt] published: {fake_result}")

    time.sleep(1.0)  # give broker a moment
    pub.disconnect()
    print("[mqtt] smoke test complete")
