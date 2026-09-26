"""
MQTT → API bridge.

Subscribes to ecg-edge/+/predictions and POSTs each alert to the API's
/alerts endpoint for persistent storage.

Run alongside `uvicorn api.main:app`:
    Terminal 1: uvicorn api.main:app --port 8000
    Terminal 2: python -m edge.mqtt_to_api
    Terminal 3: python -m edge.run_demo --n 5    # generates alerts

Usage:
    python -m edge.mqtt_to_api
    python -m edge.mqtt_to_api --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json

import paho.mqtt.client as mqtt
import requests

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "ecg-edge/+/predictions"


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print(f"[bridge] connected to {BROKER}:{PORT}")
        client.subscribe(TOPIC, qos=1)
        print(f"[bridge] subscribed to {TOPIC}")
        print("[bridge] forwarding alerts to API...")
        print()
    else:
        print(f"[bridge] connection failed: rc={reason_code}")


def on_message(client, userdata, msg):
    api_url = userdata["api_url"]
    try:
        data = json.loads(msg.payload.decode())
        # Forward to API
        r = requests.post(f"{api_url}/alerts", json=data, timeout=5.0)
        if r.status_code == 201:
            row = r.json()
            print(
                f"[bridge] stored alert id={row['id']} "
                f"device={row['device_id']} top={row['top_class']}({row['top_prob']:.3f})"
            )
        else:
            print(f"[bridge] API returned {r.status_code}: {r.text}")
    except requests.RequestException as e:
        print(f"[bridge] failed to POST to API: {e}")
    except Exception as e:
        print(f"[bridge] error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Base URL of the ECG Edge API")
    args = parser.parse_args()

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="ecg-edge-bridge",
        userdata={"api_url": args.api_url},
    )
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[bridge] connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print()
        print("[bridge] stopping...")
        client.disconnect()


if __name__ == "__main__":
    main()
