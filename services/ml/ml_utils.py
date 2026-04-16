from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

LABEL_TRUE_STRINGS = {
    "1",
    "true",
    "yes",
    "y",
    "attack",
    "attacked",
    "malicious",
    "threat",
    "suspicious",
    "anomaly",
    "incident",
    "compromised",
}

LABEL_FALSE_STRINGS = {
    "0",
    "false",
    "no",
    "n",
    "benign",
    "normal",
    "clean",
    "safe",
}

LABEL_KEYS = (
    "label",
    "is_malicious",
    "malicious",
    "attack",
    "scenario_label",
    "ground_truth",
    "ground_truth_label",
    "truth_label",
    "attack_family",
    "scenario_type",
)

LABEL_QUALITY_TIER_BY_SOURCE = {
    "metadata": "high",
    "dataset_ground_truth": "high",
    "external_heuristic": "medium",
    "legacy_rule": "low",
    "unknown": "low",
}

LABEL_SAMPLE_WEIGHT_BY_TIER = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.25,
}

EXPLICIT_ATTACK_LABEL_KEYS = (
    "label",
    "is_malicious",
    "malicious",
    "attack",
    "scenario_label",
    "ground_truth",
    "ground_truth_label",
    "truth_label",
)

FRACTURED_THREAT_POLICIES = {"off", "drop", "relabel"}
NETWORK_FLOW_FEATURE_SCHEMA = "network_flow_v1"
NETWORK_FLOW_SCHEMA_HINT_KEYS = {
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "proto",
    "service",
    "state",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "counts_by_proto",
    "counts_by_service",
    "counts_by_state",
}
NETWORK_FLOW_NUMERIC_FIELDS = (
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
NETWORK_FLOW_CATEGORICAL_FIELDS = ("proto", "service", "state")
FRACTURED_THREAT_METADATA_KEYS = (
    "attack_family",
    "family",
    "scenario",
    "scenario_name",
    "scenario_type",
    "primary_scenario",
)
FRACTURED_THREAT_LIST_METADATA_KEYS = (
    "component_scenarios",
    "scenario_signature",
    "campaign_signature",
)
FRACTURED_THREAT_MAP_METADATA_KEYS = (
    "simulation_counts_by_attack_family",
    "simulation_counts_by_scenario",
)


@dataclass(frozen=True)
class SplitRecordsResult:
    train_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    requested_split_mode: str
    actual_split_mode: str

    def __iter__(self):
        yield self.train_records
        yield self.test_records

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return (self.train_records, self.test_records)[index]


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _coerce_optional_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in LABEL_TRUE_STRINGS:
            return 1
        if normalized in LABEL_FALSE_STRINGS:
            return 0
    return None


def _coerce_count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}

    counts: dict[str, int] = {}
    for key, raw_value in value.items():
        try:
            counts[str(key)] = max(0, int(raw_value))
        except Exception:
            continue
    return counts


def _population_std(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator



def _bucketize(value: float, *, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _map_stats(prefix: str, value: Any) -> dict[str, float]:
    counts = _coerce_count_map(value)
    raw_values = list(counts.values())
    total = sum(raw_values)
    unique = len(raw_values)
    peak = max(raw_values) if raw_values else 0
    mean = total / unique if unique else 0.0
    std = _population_std(raw_values)
    concentration = peak / total if total else 0.0
    return {
        f"{prefix}_total": float(total),
        f"{prefix}_unique": float(unique),
        f"{prefix}_peak": float(peak),
        f"{prefix}_mean": float(mean),
        f"{prefix}_std": float(std),
        f"{prefix}_concentration": float(concentration),
    }


def _sparse_count_features(prefix: str, value: Any) -> dict[str, float]:
    counts = _coerce_count_map(value)
    total = sum(counts.values())
    features: dict[str, float] = {}
    for raw_key, count in sorted(counts.items()):
        category = str(raw_key).strip() or "unknown"
        features[f"{prefix}={category}"] = float(count)
        features[f"{prefix}_rate={category}"] = float(_safe_ratio(count, total))
    return features


def resolve_window_feature_schema(window: dict[str, Any]) -> str:
    if not isinstance(window, dict):
        return NETWORK_FLOW_FEATURE_SCHEMA

    raw_schema = window.get("feature_schema")
    if isinstance(raw_schema, str) and raw_schema.strip():
        return raw_schema.strip().lower()

    metadata = window.get("metadata")
    if isinstance(metadata, dict):
        raw_metadata_schema = metadata.get("feature_schema")
        if isinstance(raw_metadata_schema, str) and raw_metadata_schema.strip():
            return raw_metadata_schema.strip().lower()

    return NETWORK_FLOW_FEATURE_SCHEMA


def _safe_category(value: Any) -> str:
    text = str(value).strip().lower()
    return text or "unknown"


def extract_window(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    window = record.get("window")
    if isinstance(window, dict):
        return window
    return record


def infer_dataset_source_name(data_file: str | Path) -> str:
    path = Path(str(data_file))
    stem = path.stem.strip()
    return stem or "unknown"


def stamp_record_dataset_source(record: dict[str, Any], dataset_source: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    normalized_source = str(dataset_source).strip() or "unknown"
    stamped = dict(record)
    stamped["dataset_source"] = normalized_source

    return stamped


def resolve_record_dataset_source(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return "unknown"

    source = str(record.get("dataset_source", "")).strip()
    if source:
        return source

    window = record.get("window")
    if isinstance(window, dict):
        metadata = window.get("metadata")
        if isinstance(metadata, dict):
            nested_source = str(metadata.get("dataset_source", "")).strip()
            if nested_source:
                return nested_source
    return "unknown"


def _resolve_explicit_record_label(record: dict[str, Any]) -> int | None:
    if not isinstance(record, dict) or "label" not in record:
        return None
    return _coerce_optional_label(record.get("label"))


def _iter_attack_family_candidates(window: dict[str, Any]) -> set[str]:
    if not isinstance(window, dict):
        return set()

    candidates: set[str] = set()
    candidate_dicts = [window]
    for key in ("metadata", "simulation", "scenario", "ground_truth", "labels"):
        nested = window.get(key)
        if isinstance(nested, dict):
            candidate_dicts.append(nested)

    for candidate in candidate_dicts:
        for key in FRACTURED_THREAT_METADATA_KEYS:
            value = candidate.get(key)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized:
                    candidates.add(normalized)
            elif isinstance(value, list):
                for item in value:
                    normalized = str(item).strip().lower()
                    if normalized:
                        candidates.add(normalized)

    for map_key in FRACTURED_THREAT_MAP_METADATA_KEYS:
        counts = _coerce_count_map(window.get(map_key))
        for key, count in counts.items():
            if count <= 0:
                continue
            normalized = str(key).strip().lower()
            if normalized:
                candidates.add(normalized)

    return candidates


def is_fractured_threat_record(
    record: dict[str, Any],
    *,
    min_event_count: int = 2,
) -> bool:
    window = extract_window(record)
    fallback_event_count = sum(_coerce_count_map(window.get("counts_by_user", {})).values())
    event_count = _coerce_int(window.get("event_count"), fallback_event_count)
    if event_count >= min_event_count:
        return False

    explicit_label = _resolve_explicit_record_label(record)
    if explicit_label == 1:
        return True

    metadata_label = resolve_label_from_metadata(window)
    return metadata_label == 1


def _clear_attack_metadata_from_window(window: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(window)
    event_count = _coerce_int(
        sanitized.get("event_count"),
        sum(_coerce_count_map(sanitized.get("counts_by_user", {})).values()),
    )
    benign_event_count = max(_coerce_int(sanitized.get("simulation_benign_event_count"), 0), event_count, 1)

    candidate_dicts = [sanitized]
    for key in ("metadata", "simulation", "scenario", "ground_truth", "labels"):
        nested = sanitized.get(key)
        if isinstance(nested, dict):
            candidate_dicts.append(nested)

    for candidate in candidate_dicts:
        for key in EXPLICIT_ATTACK_LABEL_KEYS:
            if key in candidate:
                candidate[key] = 0
        for key in FRACTURED_THREAT_METADATA_KEYS:
            candidate.pop(key, None)
        for key in FRACTURED_THREAT_LIST_METADATA_KEYS:
            candidate.pop(key, None)

    sanitized["simulation_malicious_event_count"] = 0
    sanitized["simulation_benign_event_count"] = benign_event_count
    sanitized["simulation_counts_by_type"] = {"benign": benign_event_count}
    for key in FRACTURED_THREAT_MAP_METADATA_KEYS:
        sanitized[key] = {}
    return sanitized


def _relabel_fractured_threat_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(record)
    updated["label"] = 0
    for key in FRACTURED_THREAT_METADATA_KEYS + FRACTURED_THREAT_LIST_METADATA_KEYS:
        updated.pop(key, None)

    if isinstance(updated.get("window"), dict):
        updated["window"] = _clear_attack_metadata_from_window(updated["window"])
        return updated

    sanitized_window = _clear_attack_metadata_from_window(updated)
    return sanitized_window


def clean_fractured_threat_records(
    records: list[dict[str, Any]],
    *,
    policy: str = "drop",
    min_event_count: int = 2,
) -> list[dict[str, Any]]:
    normalized_policy = str(policy).strip().lower()
    if normalized_policy not in FRACTURED_THREAT_POLICIES:
        raise ValueError(
            f"Unsupported fractured threat policy: {policy}. "
            f"Expected one of {sorted(FRACTURED_THREAT_POLICIES)}."
        )
    if min_event_count < 1:
        raise ValueError("min_event_count must be at least 1")
    if normalized_policy == "off":
        return list(records)

    cleaned_records: list[dict[str, Any]] = []
    affected_count = 0
    for record in records:
        if not is_fractured_threat_record(record, min_event_count=min_event_count):
            cleaned_records.append(record)
            continue

        affected_count += 1
        if normalized_policy == "drop":
            continue
        cleaned_records.append(_relabel_fractured_threat_record(record))

    if normalized_policy == "drop":
        logging.info("Dropped %d fractured single-event threat windows.", affected_count)
    else:
        logging.info("Relabeled %d fractured single-event threat windows as benign.", affected_count)
    return cleaned_records


def clean_fractured_threat_dataframe(
    df_features: Any,
    *,
    policy: str = "drop",
    min_event_count: int = 2,
    event_count_column: str = "event_count",
    label_column: str = "label",
    family_column: str = "family",
    scenario_columns: tuple[str, ...] = ("scenario", "scenario_name", "attack_family", "primary_scenario"),
    attack_families: set[str] | None = None,
) -> Any:
    normalized_policy = str(policy).strip().lower()
    if normalized_policy not in FRACTURED_THREAT_POLICIES:
        raise ValueError(
            f"Unsupported fractured threat policy: {policy}. "
            f"Expected one of {sorted(FRACTURED_THREAT_POLICIES)}."
        )
    if min_event_count < 1:
        raise ValueError("min_event_count must be at least 1")
    if not hasattr(df_features, "copy") or not hasattr(df_features, "loc") or not hasattr(df_features, "columns"):
        raise TypeError("df_features must be a pandas DataFrame-like object")
    if normalized_policy == "off":
        return df_features.copy()
    if event_count_column not in df_features.columns:
        raise ValueError(f"DataFrame is missing required column: {event_count_column}")

    has_label_column = label_column in df_features.columns
    has_family_column = family_column in df_features.columns
    if not has_label_column and not has_family_column:
        raise ValueError(
            f"DataFrame must include at least one ground-truth column: {label_column!r} or {family_column!r}"
        )

    normalized_attack_families = {value.strip().lower() for value in (attack_families or KNOWN_ATTACK_FAMILIES)}
    cleaned_df = df_features.copy()
    low_volume_mask = cleaned_df[event_count_column].apply(lambda value: _coerce_int(value, 0) < min_event_count)
    fractured_mask = low_volume_mask & False

    if has_label_column:
        attack_label_mask = cleaned_df[label_column].apply(lambda value: _coerce_optional_label(value) == 1)
        fractured_mask = fractured_mask | (low_volume_mask & attack_label_mask)
    if has_family_column:
        family_mask = (
            cleaned_df[family_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(normalized_attack_families)
        )
        fractured_mask = fractured_mask | (low_volume_mask & family_mask)

    affected_count = int(fractured_mask.sum())
    if normalized_policy == "drop":
        logging.info("Dropped %d fractured single-event threat windows.", affected_count)
        return cleaned_df.loc[~fractured_mask].copy()

    if has_label_column:
        cleaned_df.loc[fractured_mask, label_column] = 0
    columns_to_clear = [family_column, *scenario_columns]
    for column_name in dict.fromkeys(columns_to_clear):
        if column_name in cleaned_df.columns:
            cleaned_df.loc[fractured_mask, column_name] = None

    logging.info("Relabeled %d fractured single-event threat windows as benign.", affected_count)
    return cleaned_df


def resolve_label_from_metadata(window: dict[str, Any]) -> int | None:
    candidates = [window]
    for key in ("metadata", "simulation", "scenario", "ground_truth", "labels"):
        nested = window.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for candidate in candidates:
        for key in LABEL_KEYS:
            if key not in candidate:
                continue
            value = candidate.get(key)
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value != 0)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in LABEL_TRUE_STRINGS:
                    return 1
                if normalized in LABEL_FALSE_STRINGS:
                    return 0

    malicious_event_count = _coerce_int(window.get("simulation_malicious_event_count"), default=-1)
    benign_event_count = _coerce_int(window.get("simulation_benign_event_count"), default=-1)
    if malicious_event_count >= 0 or benign_event_count >= 0:
        if malicious_event_count > 0:
            return 1
        if benign_event_count >= 0:
            return 0

    simulation_type_counts = _coerce_count_map(window.get("simulation_counts_by_type"))
    if simulation_type_counts:
        attack_total = sum(
            count
            for key, count in simulation_type_counts.items()
            if key.strip().lower() in LABEL_TRUE_STRINGS
        )
        benign_total = sum(
            count
            for key, count in simulation_type_counts.items()
            if key.strip().lower() in LABEL_FALSE_STRINGS
        )
        if attack_total > 0:
            return 1
        if benign_total > 0:
            return 0

    simulation_attack_family_counts = _coerce_count_map(window.get("simulation_counts_by_attack_family"))
    if any(count > 0 for count in simulation_attack_family_counts.values()):
        return 1
    return None


def label_from_window(window: dict[str, Any], threshold: int = 3) -> tuple[int, str]:
    metadata_label = resolve_label_from_metadata(window)
    if metadata_label is not None:
        return metadata_label, "dataset_ground_truth"
    return 0, "unknown"


def resolve_record_label_source(record: dict[str, Any], threshold: int = 3) -> str:
    if not isinstance(record, dict):
        return "unknown"

    label_source = str(record.get("label_source", "")).strip()
    if label_source:
        return label_source

    if "label" in record:
        return "unknown"

    window = extract_window(record)
    return label_from_window(window, threshold)[1]


def resolve_record_label_quality_tier(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return "low"

    explicit_tier = str(record.get("label_quality_tier", "")).strip().lower()
    if explicit_tier in LABEL_SAMPLE_WEIGHT_BY_TIER:
        return explicit_tier

    label_source = resolve_record_label_source(record)
    return LABEL_QUALITY_TIER_BY_SOURCE.get(label_source, "low")


def resolve_record_sample_weight(record: dict[str, Any]) -> float:
    tier = resolve_record_label_quality_tier(record)
    return float(LABEL_SAMPLE_WEIGHT_BY_TIER.get(tier, LABEL_SAMPLE_WEIGHT_BY_TIER["low"]))


def resolve_window_threat_family(window: dict[str, Any]) -> str:
    if not isinstance(window, dict):
        return "network_intrusion"
    metadata = window.get("metadata")
    if isinstance(metadata, dict):
        attack_cat = metadata.get("attack_cat")
        if isinstance(attack_cat, str) and attack_cat.strip():
            return f"network_{attack_cat.strip().lower()}"
    return "network_intrusion"




def _network_flow_window_to_feature_dict(window: dict[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    event_count = max(1, _coerce_int(window.get("event_count"), 1))
    features["event_count"] = float(event_count)

    for field_name in NETWORK_FLOW_NUMERIC_FIELDS:
        if field_name not in window:
            continue
        numeric_value = _coerce_float(window.get(field_name), default=0.0)
        features[field_name] = float(numeric_value)

    sbytes = float(features.get("sbytes", 0.0))
    dbytes = float(features.get("dbytes", 0.0))
    spkts = float(features.get("spkts", 0.0))
    dpkts = float(features.get("dpkts", 0.0))
    duration = float(features.get("dur", 0.0))
    packets_total = spkts + dpkts
    bytes_total = sbytes + dbytes

    features["bytes_total"] = bytes_total
    features["packets_total"] = packets_total
    features["sbytes_ratio"] = _safe_ratio(sbytes, max(bytes_total, 1.0))
    features["spkts_ratio"] = _safe_ratio(spkts, max(packets_total, 1.0))
    features["bytes_per_packet"] = _safe_ratio(bytes_total, max(packets_total, 1.0))
    features["packets_per_second"] = _safe_ratio(packets_total, max(duration, 1e-6))
    features["bytes_per_second"] = _safe_ratio(bytes_total, max(duration, 1e-6))

    proto = _safe_category(window.get("proto", "unknown"))
    service = _safe_category(window.get("service", "unknown"))
    state = _safe_category(window.get("state", "unknown"))
    features[f"proto={proto}"] = 1.0
    features[f"service={service}"] = 1.0
    features[f"state={state}"] = 1.0

    source_ip_map = _coerce_count_map(window.get("counts_by_source_ip"))
    if not source_ip_map and str(window.get("srcip", "")).strip():
        source_ip_map = {str(window["srcip"]).strip(): 1}

    destination_ip_map = _coerce_count_map(window.get("counts_by_destination_ip"))
    if not destination_ip_map and str(window.get("dstip", "")).strip():
        destination_ip_map = {str(window["dstip"]).strip(): 1}

    proto_map = _coerce_count_map(window.get("counts_by_proto"))
    if not proto_map:
        proto_map = {proto: 1}
    service_map = _coerce_count_map(window.get("counts_by_service"))
    if not service_map:
        service_map = {service: 1}
    state_map = _coerce_count_map(window.get("counts_by_state"))
    if not state_map:
        state_map = {state: 1}

    unique_source_ip_count = _coerce_int(window.get("unique_source_ip_count"), len(source_ip_map))
    unique_destination_ip_count = _coerce_int(window.get("unique_destination_ip_count"), len(destination_ip_map))
    unique_proto_count = _coerce_int(window.get("unique_proto_count"), len(proto_map))
    unique_service_count = _coerce_int(window.get("unique_service_count"), len(service_map))
    unique_state_count = _coerce_int(window.get("unique_state_count"), len(state_map))

    features["unique_source_ip_count"] = float(max(0, unique_source_ip_count))
    features["unique_destination_ip_count"] = float(max(0, unique_destination_ip_count))
    features["unique_proto_count"] = float(max(0, unique_proto_count))
    features["unique_service_count"] = float(max(0, unique_service_count))
    features["unique_state_count"] = float(max(0, unique_state_count))
    features["source_ip_spread"] = float(_safe_ratio(unique_source_ip_count, max(event_count, 1)))
    features["destination_ip_spread"] = float(_safe_ratio(unique_destination_ip_count, max(event_count, 1)))

    features.update(_map_stats("source_ip", source_ip_map))
    features.update(_map_stats("destination_ip", destination_ip_map))
    features.update(_map_stats("proto", proto_map))
    features.update(_map_stats("service", service_map))
    features.update(_map_stats("state", state_map))
    features.update(_sparse_count_features("proto", proto_map))
    features.update(_sparse_count_features("service", service_map))
    features.update(_sparse_count_features("state", state_map))
    return features


def window_to_feature_dict(window: dict[str, Any]) -> dict[str, float]:
    return _network_flow_window_to_feature_dict(window)


def _network_flow_window_to_text(window: dict[str, Any]) -> str:
    features = _network_flow_window_to_feature_dict(window)
    proto = _safe_category(window.get("proto", "unknown"))
    service = _safe_category(window.get("service", "unknown"))
    state = _safe_category(window.get("state", "unknown"))
    parts = [
        f"feature_schema={NETWORK_FLOW_FEATURE_SCHEMA}",
        f"event_count={int(features.get('event_count', 1.0))}",
        f"dur={features.get('dur', 0.0):.6f}",
        f"sbytes={features.get('sbytes', 0.0):.3f}",
        f"dbytes={features.get('dbytes', 0.0):.3f}",
        f"spkts={features.get('spkts', 0.0):.3f}",
        f"dpkts={features.get('dpkts', 0.0):.3f}",
        f"bytes_total={features.get('bytes_total', 0.0):.3f}",
        f"packets_total={features.get('packets_total', 0.0):.3f}",
        f"bytes_per_second={features.get('bytes_per_second', 0.0):.3f}",
        f"packets_per_second={features.get('packets_per_second', 0.0):.3f}",
        f"source_ip_spread={features.get('source_ip_spread', 0.0):.3f}",
        f"destination_ip_spread={features.get('destination_ip_spread', 0.0):.3f}",
        f"proto={proto}",
        f"service={service}",
        f"state={state}",
    ]
    for key in ("ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_dst_src_ltm", "is_sm_ips_ports"):
        if key in features:
            parts.append(f"{key}={features[key]:.3f}")
    return " ".join(parts)


def window_to_text(window: dict[str, Any]) -> str:
    return _network_flow_window_to_text(window)


def record_to_window_text(record: dict[str, Any]) -> tuple[dict[str, Any], str]:
    window = extract_window(record)
    if window:
        return window, window_to_text(window)
    return {}, str(record.get("text", "")) if isinstance(record, dict) else ""



def load_jsonl_records(data_file: str | Path | list[str] | tuple[str, ...], max_samples: int = 0) -> list[dict[str, Any]]:
    if isinstance(data_file, (list, tuple)):
        records: list[dict[str, Any]] = []
        for path in data_file:
            remaining = max_samples - len(records) if max_samples > 0 else 0
            if max_samples > 0 and remaining <= 0:
                break
            records.extend(load_jsonl_records(path, max_samples=remaining))
        return records

    records: list[dict[str, Any]] = []
    dataset_source = infer_dataset_source_name(data_file)
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples > 0 and len(records) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    records.append(stamp_record_dataset_source(parsed, dataset_source))
            except Exception:
                continue
    return records


def split_records(
    records: list[dict[str, Any]],
    *,
    test_size: float = 0.2,
    split_mode: str = "time",
    seed: int = 42,
) -> SplitRecordsResult:
    if not records:
        return SplitRecordsResult([], [], split_mode, split_mode)

    requested_split_mode = str(split_mode)
    actual_split_mode = requested_split_mode
    if split_mode == "time":
        ordered = sorted(
            records,
            key=lambda record: (
                _coerce_int(record.get("window_start_ms"), 0),
                _coerce_int(record.get("window_end_ms"), 0),
                str(record.get("id", "")),
            ),
        )
        split_index = max(1, min(len(ordered) - 1, int(round(len(ordered) * (1 - test_size)))))
        train = ordered[:split_index]
        test = ordered[split_index:]
        if not test or len({resolve_record_label(record) for record in train}) < 2 or len({resolve_record_label(record) for record in test}) < 2:
            actual_split_mode = "stratified"
        else:
            return SplitRecordsResult(train, test, requested_split_mode, actual_split_mode)

    labels = [resolve_record_label(record) for record in records]
    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    if actual_split_mode != requested_split_mode:
        logging.info(
            "Requested split_mode=%s fell back to %s for label-balanced evaluation.",
            requested_split_mode,
            actual_split_mode,
        )
    return SplitRecordsResult(list(train_records), list(test_records), requested_split_mode, actual_split_mode)


def resolve_record_label(record: dict[str, Any], threshold: int = 3) -> int:
    if not isinstance(record, dict):
        return 0

    window = extract_window(record)

    if "label" in record:
        label = record.get("label")
        if isinstance(label, bool):
            return int(label)
        if isinstance(label, (int, float)):
            return int(label != 0)
        if isinstance(label, str):
            normalized = label.strip().lower()
            if normalized in LABEL_TRUE_STRINGS:
                return 1
            if normalized in LABEL_FALSE_STRINGS:
                return 0

    metadata_label = resolve_label_from_metadata(window)
    if metadata_label is not None:
        return metadata_label

    return 0


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
        "positive_precision": float(precision[1]),
        "positive_recall": float(recall[1]),
        "positive_f1": float(f1[1]),
        "negative_support": int(support[0]),
        "positive_support": int(support[1]),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": int(len(y_true)),
    }

    if y_score is not None and len(set(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["average_precision"] = None
    return metrics


def find_best_threshold(
    scores: list[float],
    labels: list[int],
    metric: str = "macro_f1",
) -> tuple[float, dict[str, Any]]:
    """Return (threshold, metrics_at_threshold) maximising *metric* on training data.

    metric choices
    --------------
    ``"macro_f1"``   — average of positive-F1 and negative-F1.  Balances
                       detection rate against false-positive rate, which is
                       essential for operational deployments where analysts
                       triage every alert.  This is the default.

    ``"positive_f1"`` — maximises detection recall at the cost of high FP
                        rates.  Useful for very high-security environments
                        that accept alert fatigue in exchange for near-zero
                        missed attacks.
    """
    if not scores:
        return 0.5, {"positive_f1": 0.0, "positive_precision": 0.0, "positive_recall": 0.0}

    y_score = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.int32)

    if len(set(labels)) < 2:
        best_threshold = float(y_score.mean())
        return best_threshold, compute_metrics(labels, (y_score >= best_threshold).astype(int).tolist(), scores)

    # precision_recall_curve is O(n log n): computes all threshold candidates in one pass.
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_true, y_score)

    if len(thresholds_arr) == 0:
        best_threshold = 0.5
    else:
        # precision_arr[i]/recall_arr[i] correspond to thresholds_arr[i]; last entry has no threshold.
        p = precision_arr[:-1]  # positive precision
        r = recall_arr[:-1]     # positive recall

        denom_pos = p + r
        pos_f1 = np.where(denom_pos > 0, 2.0 * p * r / denom_pos, 0.0)

        if metric == "macro_f1":
            # Derive negative-class precision/recall algebraically from the positive-class PR curve.
            # At each threshold candidate:
            #   TP = r * n_pos
            #   FP = TP * (1 - p) / p   (when p > 0)
            #   TN = n_neg - FP
            #   FN = n_pos - TP
            n_pos = int(np.sum(y_true == 1))
            n_neg = int(np.sum(y_true == 0))

            tp = r * n_pos
            fp = np.where(p > 0, tp * (1.0 - p) / p, float(n_neg))
            tn = np.maximum(n_neg - fp, 0.0)
            fn = n_pos - tp

            neg_recall = tn / n_neg if n_neg > 0 else np.zeros_like(tn)
            with np.errstate(divide="ignore", invalid="ignore"):
                neg_precision = np.where(
                    (tn + fn) > 0, tn / (tn + fn), 0.0
                )
                denom_neg = neg_precision + neg_recall
                neg_f1 = np.where(denom_neg > 0, 2.0 * neg_precision * neg_recall / denom_neg, 0.0)

            objective = (pos_f1 + neg_f1) / 2.0
        else:
            # "positive_f1" or any unrecognised value — fall back to positive-only F1
            objective = pos_f1

        best_idx = int(np.argmax(objective))
        best_threshold = float(thresholds_arr[best_idx])

    best_preds = (y_score >= best_threshold).astype(int).tolist()
    return best_threshold, compute_metrics(labels, best_preds, scores)




def score_structured_inputs(
    model_dict: dict[str, Any],
    windows: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    structured_model = model_dict.get("structured_model")
    structured_vectorizer = model_dict.get("structured_vectorizer")
    structured_threshold = float(model_dict.get("structured_threshold", 0.5))
    structured_family_models = model_dict.get("structured_family_models") or {}
    structured_family_vectorizers = model_dict.get("structured_family_vectorizers") or {}
    structured_family_thresholds = model_dict.get("structured_family_thresholds") or {}

    if structured_model is None or structured_vectorizer is None:
        raise ValueError("Structured model bundle is incomplete")

    results: list[dict[str, Any]] = []
    for window in windows:
        resolved_window = window or {}
        family = resolve_window_threat_family(resolved_window)
        family_model = structured_family_models.get(family)
        family_vectorizer = structured_family_vectorizers.get(family)
        family_threshold = float(structured_family_thresholds.get(family, structured_threshold))

        active_model = family_model if family_model is not None else structured_model
        active_vectorizer = family_vectorizer if family_vectorizer is not None else structured_vectorizer
        active_threshold = family_threshold if family_model is not None else structured_threshold

        if not hasattr(active_model, "predict_proba"):
            raise ValueError("Structured model does not support probability scoring")

        feature_row = window_to_feature_dict(resolved_window)
        matrix = active_vectorizer.transform([feature_row])
        score = float(active_model.predict_proba(matrix)[0][1])
        results.append(
            {
                "label": int(score >= active_threshold),
                "score": score,
                "model": "structured_baseline",
                "threshold": float(active_threshold),
            }
        )
    return results
