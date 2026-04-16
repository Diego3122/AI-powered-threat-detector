#!/usr/bin/env python
"""
UNSW-NB15 replay producer.

Reads a JSONL file produced by scripts/build_unsw_nb15_dataset.py and emits
each record as a Kafka message directly to the windowed topic (bypassing the
window-builder, since UNSW-NB15 records are already complete network-flow windows).

Usage:
    python services/ingest_sim/replay_producer.py \\
        --input-file data/unsw_nb15.jsonl \\
        --topic unsw_nb15.windowed \\
        --rate 50 \\
        --count 1000

Environment:
    KAFKA_BOOTSTRAP   Kafka broker address (default: 127.0.0.1:9092)
    OUT_TOPIC         Default Kafka topic (default: unsw_nb15.windowed)
"""
import argparse
import json
import os
import sys
import time

try:
    from confluent_kafka import Producer

    _HAVE_KAFKA = True
except ImportError:
    Producer = None
    _HAVE_KAFKA = False

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "127.0.0.1:9092")
DEFAULT_TOPIC = os.getenv("OUT_TOPIC", "unsw_nb15.windowed")


def _delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for {msg.key()}: {err}", file=sys.stderr)


def replay(input_file: str, topic: str, rate: float, count: int) -> None:
    if not _HAVE_KAFKA:
        print("ERROR: confluent_kafka not available. Install requirements.", file=sys.stderr)
        sys.exit(1)

    producer = Producer({"bootstrap.servers": BOOTSTRAP})
    interval = 1.0 / rate if rate > 0 else 0.0

    src = sys.stdin if input_file == "-" else open(input_file, "r", encoding="utf-8")
    sent = 0
    try:
        for line in src:
            if count > 0 and sent >= count:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: Skipping malformed JSON line: {exc}", file=sys.stderr)
                continue

            record_id = record.get("id", str(sent))
            producer.produce(
                topic,
                value=json.dumps(record).encode("utf-8"),
                key=str(record_id).encode("utf-8"),
                on_delivery=_delivery_report,
            )
            producer.poll(0)
            sent += 1

            if sent % 500 == 0:
                producer.flush(5)
                print(f"  … {sent} records sent to {topic}")

            if interval > 0:
                time.sleep(interval)
    finally:
        if input_file != "-":
            src.close()

    producer.flush(10)
    print(f"Done. {sent} records sent to topic '{topic}'.")


def main():
    ap = argparse.ArgumentParser(
        description="Replay UNSW-NB15 JSONL windows to Kafka (bypasses window-builder)"
    )
    ap.add_argument(
        "--input-file",
        default="-",
        help="Path to UNSW JSONL file (use '-' for stdin). Default: stdin",
    )
    ap.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"Kafka topic to produce to (default: {DEFAULT_TOPIC})",
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=0,
        help="Records per second (0 = as fast as possible). Default: 0",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=0,
        help="Max records to send (0 = all). Default: 0",
    )
    ap.add_argument(
        "--bootstrap",
        default=BOOTSTRAP,
        help=f"Kafka bootstrap servers (default: {BOOTSTRAP})",
    )
    args = ap.parse_args()

    # allow CLI override of bootstrap
    global BOOTSTRAP
    BOOTSTRAP = args.bootstrap

    replay(args.input_file, args.topic, args.rate, args.count)


if __name__ == "__main__":
    main()
