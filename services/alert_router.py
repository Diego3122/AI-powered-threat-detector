#!/usr/bin/env python
import json
import os
import sys
import time

try:
    from confluent_kafka import Consumer, KafkaException
except Exception as exc:
    raise SystemExit(f"confluent_kafka is required for alert routing: {exc}")

import requests

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
TOPIC = os.getenv("ALERTS_TOPIC", os.getenv("OUT_TOPIC", "alerts"))
GROUP_ID = os.getenv("GROUP_ID", "alert-router-dev")
OFFSET_RESET = os.getenv("OFFSET_RESET", "earliest")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "dev-internal-api-key")


def _wait_for_api(api_base_url: str) -> None:
    while True:
        try:
            response = requests.get(f"{api_base_url}/api/health", timeout=3)
            response.raise_for_status()
            print(f"Connected to API at {api_base_url}")
            return
        except Exception as exc:
            print(f"INFO: Waiting for API availability: {exc}", file=sys.stderr)
            time.sleep(2)


def _forward_alert(api_base_url: str, alert: dict) -> None:
    response = requests.post(
        f"{api_base_url}/api/internal/alerts/ingest",
        json={
            "timestamp": alert["timestamp"],
            "window_id": alert["window_id"],
            "model_type": alert["model_type"],
            "model_score": alert["model_score"],
            "threshold": alert["threshold"],
            "model_label": alert.get("model_label"),
            "explanation_summary": alert.get("explanation_summary"),
            "feature_schema": alert.get("feature_schema"),
            "window": alert.get("window"),
        },
        headers={"X-Internal-API-Key": INTERNAL_API_KEY},
        timeout=5,
    )
    response.raise_for_status()


def main() -> int:
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": GROUP_ID,
            "auto.offset.reset": OFFSET_RESET,
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([TOPIC])

    _wait_for_api(API_BASE_URL)

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            try:
                alert = json.loads(msg.value().decode("utf-8"))
            except Exception as exc:
                print(f"WARNING: Dropping malformed alert message: {exc}", file=sys.stderr)
                consumer.commit(message=msg, asynchronous=False)
                continue

            try:
                _forward_alert(API_BASE_URL, alert)
                consumer.commit(message=msg, asynchronous=False)
                print(f"Persisted alert {alert.get('window_id', 'unknown-window')}")
            except Exception as exc:
                print(f"WARNING: Failed to forward alert: {exc}", file=sys.stderr)
                time.sleep(2)
    except KeyboardInterrupt:
        print("Interrupted, exiting.", file=sys.stderr)
        return 0
    finally:
        consumer.close()


if __name__ == "__main__":
    raise SystemExit(main())
