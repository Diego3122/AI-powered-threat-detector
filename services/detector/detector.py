#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from typing import Any, Optional

try:
    from confluent_kafka import Consumer, Producer, KafkaException

    _HAVE_KAFKA = True
except Exception:
    Consumer = Producer = KafkaException = None
    _HAVE_KAFKA = False

import requests

from services.ml.ml_utils import (
    resolve_window_feature_schema,
    window_to_text,
)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
IN_TOPIC = os.getenv("IN_TOPIC", "unsw_nb15.windowed")
OUT_TOPIC = os.getenv("OUT_TOPIC", "alerts")
GROUP_ID = os.getenv("GROUP_ID", "detector-dev")
OFFSET_RESET = os.getenv("OFFSET_RESET", "latest")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000")


def _window_to_text(window: dict[str, Any]) -> str:
    return window_to_text(window)


def _extract_window_payload(record: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    if not isinstance(record, dict):
        raise ValueError("Window record must be an object")

    raw_window = record.get("window") if isinstance(record.get("window"), dict) else record
    if not isinstance(raw_window, dict):
        raise ValueError("Window payload must be an object")

    window_start = raw_window.get("window_start_ms", int(time.time() * 1000))
    window_end = raw_window.get("window_end_ms", window_start)
    window_id = str(record.get("id") or raw_window.get("id") or f"{window_start}-{window_end}")
    text = record.get("text") or window_to_text(raw_window)
    return raw_window, text, window_id


def _build_network_flow_explanation(window: dict[str, Any], model_type: str, score: float) -> str:
    proto = window.get("proto", "unknown")
    svc = window.get("service", "-")
    state = window.get("state", "-")
    sbytes = window.get("sbytes", 0)
    dbytes = window.get("dbytes", 0)
    attack_cat = (window.get("metadata") or {}).get("attack_cat", "")
    cat_str = f" [{attack_cat}]" if attack_cat else ""
    return (
        f"{model_type} flagged anomalous network flow{cat_str} "
        f"(proto={proto}, service={svc}, state={state}, "
        f"sbytes={sbytes}, dbytes={dbytes}, score={score:.2f})"
    )


def _build_explanation(window: dict[str, Any], model_type: str, score: float, result: Optional[dict[str, Any]] = None) -> str:
    return _build_network_flow_explanation(window, model_type, score)


def _score_text(model_server_url: str, text: str, window: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    response = requests.post(
        f"{model_server_url}/score",
        json={"text": text, "window": window},
        timeout=5,
    )
    response.raise_for_status()
    result = response.json()
    parsed = {
        "label": int(result["label"]),
        "score": float(result["score"]),
        "model": result.get("model", "distilbert"),
    }
    if "threshold" in result:
        parsed["threshold"] = float(result["threshold"])
    return parsed


def _build_alert(window: dict[str, Any], window_id: str, result: dict[str, Any], threshold: float) -> dict[str, Any]:
    alert_timestamp = int(time.time() * 1000)
    effective_label = int(float(result["score"]) >= float(threshold))
    alert = {
        "alert_type": "model_alert",
        "timestamp": alert_timestamp,
        "window_id": window_id,
        "window": window,
        "feature_schema": resolve_window_feature_schema(window),
        "model_label": effective_label,
        "model_type": result["model"],
        "model_score": result["score"],
        "threshold": threshold,
        "explanation_summary": _build_explanation(window, result["model"], result["score"], result),
    }
    return alert


def _check_model_server(model_server_url: str) -> None:
    try:
        response = requests.get(f"{model_server_url}/health", timeout=2)
        response.raise_for_status()
        print(f"Connected to model server at {model_server_url}")
    except Exception as exc:
        print(f"WARNING: Model server not reachable: {exc}", file=sys.stderr)
        print("Continuing anyway; will retry on the first window.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Threat detector: score windows and emit alerts")
    ap.add_argument("--input-file", default=None, help="Path to file with window JSON (one per line). Use '-' for stdin. If not set, reads from Kafka.")
    ap.add_argument("--output-file", default=None, help="Write alerts to this file (default: stdout for file mode, Kafka for Kafka mode)")
    ap.add_argument("--bootstrap", default=BOOTSTRAP)
    ap.add_argument("--in-topic", default=IN_TOPIC)
    ap.add_argument("--out-topic", default=OUT_TOPIC)
    ap.add_argument("--model-server", default=MODEL_SERVER_URL, help="Model server URL (e.g., http://localhost:8000)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Alert threshold override (default: use model-provided threshold, else 0.5)",
    )
    args = ap.parse_args()

    file_mode = args.input_file is not None
    consumer = None
    producer = None
    out_fd = None

    if file_mode:
        input_stream = sys.stdin if args.input_file == "-" else open(args.input_file, "r", encoding="utf-8")
    else:
        if not _HAVE_KAFKA:
            print("ERROR: confluent_kafka not available and no --input-file given.", file=sys.stderr)
            sys.exit(1)

        consumer = Consumer(
            {
                "bootstrap.servers": args.bootstrap,
                "group.id": GROUP_ID,
                "auto.offset.reset": OFFSET_RESET,
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([args.in_topic])

    if file_mode:
        if args.output_file and args.output_file != "-":
            out_fd = open(args.output_file, "w", encoding="utf-8")
    else:
        producer = Producer({"bootstrap.servers": args.bootstrap})

    def emit_alert(alert: dict[str, Any]):
        if producer is not None:
            producer.produce(args.out_topic, json.dumps(alert).encode("utf-8"))
            producer.poll(0)
        else:
            serialized = json.dumps(alert)
            if out_fd is None:
                print(serialized)
            else:
                out_fd.write(serialized + "\n")

    _check_model_server(args.model_server)

    try:
        if file_mode:
            for line in input_stream:
                record = line.strip()
                if not record:
                    continue
                try:
                    raw_window = json.loads(record)
                    window, text, window_id = _extract_window_payload(raw_window)
                except Exception as exc:
                    print(f"WARNING: Failed to parse window record: {exc}", file=sys.stderr)
                    continue

                try:
                    result = _score_text(args.model_server, text, window)
                except Exception as exc:
                    print(f"WARNING: Model server error: {exc}", file=sys.stderr)
                    continue

                effective_threshold = float(result.get("threshold", 0.5) if args.threshold is None else args.threshold)
                if result["score"] >= effective_threshold:
                    alert = _build_alert(window, window_id, result, effective_threshold)
                    emit_alert(alert)
                    print(f"[ALERT] score={result['score']:.3f} window_id={window_id}")
        else:
            while True:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    raise KafkaException(msg.error())

                try:
                    raw_window = json.loads(msg.value().decode("utf-8"))
                    window, text, window_id = _extract_window_payload(raw_window)
                except Exception as exc:
                    print(f"WARNING: Failed to decode window message: {exc}", file=sys.stderr)
                    consumer.commit(message=msg, asynchronous=False)
                    continue

                try:
                    result = _score_text(args.model_server, text, window)
                except Exception as exc:
                    print(f"WARNING: Model server error: {exc}", file=sys.stderr)
                    time.sleep(2)
                    continue

                effective_threshold = float(result.get("threshold", 0.5) if args.threshold is None else args.threshold)
                if result["score"] >= effective_threshold:
                    alert = _build_alert(window, window_id, result, effective_threshold)
                    emit_alert(alert)
                    print(f"[ALERT] score={result['score']:.3f} window_id={window_id}")

                consumer.commit(message=msg, asynchronous=False)

    except KeyboardInterrupt:
        print("Interrupted, exiting.", file=sys.stderr)
    finally:
        if file_mode:
            if args.input_file != "-" and input_stream is not None:
                input_stream.close()
            if out_fd is not None:
                out_fd.close()
        if consumer is not None:
            consumer.close()
        if producer is not None:
            producer.flush(5)


if __name__ == "__main__":
    main()
