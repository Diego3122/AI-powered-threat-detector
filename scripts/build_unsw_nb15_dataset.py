#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from services.ml.ml_utils import (
    NETWORK_FLOW_FEATURE_SCHEMA,
    resolve_record_label_quality_tier,
    window_to_text,
)

UNSW_NUMERIC_FIELDS = (
    "dur",
    "sport",
    "dsport",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)

UNSW_SOURCE_ALIASES = ("srcip", "src_ip", "sip")
UNSW_DEST_ALIASES = ("dstip", "dst_ip", "dip")
UNSW_ID_ALIASES = ("id", "record_id", "row_id")
UNSW_START_TIME_ALIASES = ("stime", "start_time")
UNSW_END_TIME_ALIASES = ("ltime", "end_time")
UNSW_DURATION_ALIASES = ("dur",)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def _normalize_row(raw_row: dict[str, Any]) -> dict[str, str]:
    return {str(key).strip().lstrip("\ufeff").lower(): str(value).strip() for key, value in raw_row.items()}


def _resolve_value(row: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in row and row[alias]:
            return row[alias]
    return None


def _epoch_ms_from_any(raw_value: str | None) -> int | None:
    value = _coerce_float(raw_value)
    if value is None:
        return None
    if value > 1_000_000_000_000:
        return int(value)
    return int(value * 1000)


def _derive_window_bounds_ms(row: dict[str, str], fallback_index: int) -> tuple[int, int]:
    start_ms = _epoch_ms_from_any(_resolve_value(row, UNSW_START_TIME_ALIASES))
    end_ms = _epoch_ms_from_any(_resolve_value(row, UNSW_END_TIME_ALIASES))
    duration_seconds = _coerce_float(_resolve_value(row, UNSW_DURATION_ALIASES))

    if start_ms is None:
        start_ms = 1_700_000_000_000 + fallback_index
    if end_ms is None:
        if duration_seconds is not None and duration_seconds >= 0:
            end_ms = start_ms + max(1, int(duration_seconds * 1000))
        else:
            end_ms = start_ms + 1
    if end_ms <= start_ms:
        end_ms = start_ms + 1
    return start_ms, end_ms


def _coerce_binary_label(raw_value: str | None) -> int | None:
    value = _coerce_int(raw_value)
    if value is None:
        return None
    return int(value != 0)


def _build_unsw_window(row: dict[str, str], index: int, dataset_source: str) -> tuple[dict[str, Any], int] | None:
    label = _coerce_binary_label(row.get("label"))
    if label is None:
        return None

    start_ms, end_ms = _derive_window_bounds_ms(row, fallback_index=index)
    src_ip = _resolve_value(row, UNSW_SOURCE_ALIASES)
    dst_ip = _resolve_value(row, UNSW_DEST_ALIASES)
    proto = (row.get("proto") or "unknown").lower()
    service = (row.get("service") or "unknown").lower()
    state = (row.get("state") or "unknown").lower()
    attack_cat = row.get("attack_cat") or None

    window: dict[str, Any] = {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        "window_start_ms": start_ms,
        "window_end_ms": end_ms,
        "event_count": 1,
        "srcip": src_ip or "unknown",
        "dstip": dst_ip or "unknown",
        "proto": proto,
        "service": service,
        "state": state,
        "unique_source_ip_count": 1 if src_ip else 0,
        "unique_destination_ip_count": 1 if dst_ip else 0,
        "unique_proto_count": 1 if proto and proto != "unknown" else 0,
        "unique_service_count": 1 if service and service != "unknown" else 0,
        "unique_state_count": 1 if state and state != "unknown" else 0,
        "counts_by_source_ip": {src_ip: 1} if src_ip else {},
        "counts_by_destination_ip": {dst_ip: 1} if dst_ip else {},
        "counts_by_proto": {proto: 1} if proto else {},
        "counts_by_service": {service: 1} if service else {},
        "counts_by_state": {state: 1} if state else {},
        "metadata": {
            "dataset_origin": "unsw_nb15",
            "dataset_source": dataset_source,
            "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
            "label_source": "dataset_ground_truth",
            "attack_cat": attack_cat,
        },
    }

    for field in UNSW_NUMERIC_FIELDS:
        numeric_value = _coerce_float(row.get(field))
        if numeric_value is not None:
            window[field] = float(numeric_value)
    return window, label


def build_dataset_rows(csv_path: Path, *, dataset_source: str | None = None, max_rows: int = 0) -> list[dict[str, Any]]:
    resolved_source = (dataset_source or csv_path.stem).strip() or "unsw_nb15_unknown"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, raw_row in enumerate(reader, start=1):
            if max_rows > 0 and len(rows) >= max_rows:
                break

            normalized = _normalize_row(raw_row)
            built = _build_unsw_window(normalized, index=index, dataset_source=resolved_source)
            if built is None:
                continue

            window, label = built
            record_id = _resolve_value(normalized, UNSW_ID_ALIASES) or f"{resolved_source}-{index}"
            row = {
                "id": record_id,
                "window_start_ms": window["window_start_ms"],
                "window_end_ms": window["window_end_ms"],
                "text": window_to_text(window),
                "label": int(label),
                "label_source": "dataset_ground_truth",
                "label_quality_tier": resolve_record_label_quality_tier({"label_source": "dataset_ground_truth"}),
                "window": window,
            }
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert UNSW-NB15 CSV rows to the threat-detector JSONL dataset format",
    )
    parser.add_argument("--input-csv", required=True, help="Path to UNSW-NB15 CSV file")
    parser.add_argument("--out", required=True, help="Output JSONL dataset path")
    parser.add_argument("--dataset-source", default=None, help="Override dataset_source tag")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to export (0 = all)")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows = build_dataset_rows(
        input_csv,
        dataset_source=args.dataset_source,
        max_rows=args.max_rows,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
