import json
import shutil
import uuid
from pathlib import Path

from services.ml.ml_utils import (
    NETWORK_FLOW_FEATURE_SCHEMA,
    clean_fractured_threat_records,
    load_jsonl_records,
    resolve_record_dataset_source,
    resolve_record_label_quality_tier,
    resolve_record_sample_weight,
    resolve_record_label,
    resolve_window_threat_family,
    split_records,
    window_to_feature_dict,
    window_to_text,
)


def _make_fractured_attack_record() -> dict:
    return {
        "id": "fractured-attack",
        "label": 1,
        "label_source": "metadata",
        "window_start_ms": 1_700_000_100_000,
        "window_end_ms": 1_700_000_100_001,
        "window": {
            "window_start_ms": 1_700_000_100_000,
            "window_end_ms": 1_700_000_100_001,
            "event_count": 1,
            "login_event_count": 0,
            "success_event_count": 1,
            "failure_event_count": 0,
            "counts_by_user": {"alice": 1},
            "counts_by_source_ip": {"203.0.113.24": 1},
            "counts_by_user_agent": {"AWS CLI/2.15": 1},
            "counts_by_event_name": {"GetCallerIdentity": 1},
            "simulation_malicious_event_count": 1,
            "simulation_benign_event_count": 0,
            "simulation_counts_by_type": {"attack": 1},
            "simulation_counts_by_attack_family": {"session_hijack": 1},
            "simulation_counts_by_scenario": {"session_cookie_reuse": 1},
            "metadata": {
                "attack_family": "session_hijack",
                "scenario_name": "session_cookie_reuse",
                "primary_scenario": "session_cookie_reuse",
            },
        },
    }



def test_ml_utils_resolves_label_quality_tier_and_sample_weight():
    assert resolve_record_label_quality_tier({"label_source": "metadata"}) == "high"
    assert resolve_record_sample_weight({"label_source": "metadata"}) == 1.0
    assert resolve_record_label_quality_tier({"label_source": "dataset_ground_truth"}) == "high"
    assert resolve_record_sample_weight({"label_source": "dataset_ground_truth"}) == 1.0
    assert resolve_record_label_quality_tier({"label_source": "external_heuristic"}) == "medium"
    assert resolve_record_sample_weight({"label_source": "external_heuristic"}) == 0.5
    assert resolve_record_label_quality_tier({"label_source": "legacy_rule"}) == "low"
    assert resolve_record_sample_weight({"label_source": "legacy_rule"}) == 0.25
    assert resolve_record_label_quality_tier({"label": 1}) == "low"
    assert resolve_record_sample_weight({"label": 1}) == 0.25


def test_ml_utils_resolves_window_threat_family_from_unsw_attack_cat():
    recon_window = {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        "metadata": {"attack_cat": "Reconnaissance"},
    }
    generic_window = {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
    }

    assert resolve_window_threat_family(recon_window) == "network_reconnaissance"
    assert resolve_window_threat_family(generic_window) == "network_intrusion"


def test_load_jsonl_records_stamps_dataset_source_for_multiple_files():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"ml-utils-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    first = temp_dir / "alpha.jsonl"
    second = temp_dir / "beta.jsonl"
    first.write_text(json.dumps({"label": 0, "window": {"window_start_ms": 0, "metadata": {}}}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"label": 1, "window": {"window_start_ms": 1, "metadata": {}}}) + "\n", encoding="utf-8")

    try:
        records = load_jsonl_records([str(first), str(second)])

        assert len(records) == 2
        assert [resolve_record_dataset_source(record) for record in records] == ["alpha", "beta"]
        assert "dataset_source" not in records[0]["window"].get("metadata", {})
        assert "dataset_source" not in records[1]["window"].get("metadata", {})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_split_records_reports_actual_fallback_mode():
    records = [
        {"label": 0, "window_start_ms": 1, "window_end_ms": 2},
        {"label": 0, "window_start_ms": 3, "window_end_ms": 4},
        {"label": 1, "window_start_ms": 5, "window_end_ms": 6},
        {"label": 1, "window_start_ms": 7, "window_end_ms": 8},
    ]

    split_result = split_records(records, test_size=0.5, split_mode="time", seed=42)
    train_records, test_records = split_result

    assert split_result.actual_split_mode == "stratified"
    assert split_result.requested_split_mode == "time"
    assert len(train_records) == 2
    assert len(test_records) == 2


def test_clean_fractured_threat_records_drops_single_event_attack_windows(caplog):
    fractured_attack = _make_fractured_attack_record()
    benign_single_event = {
        "id": "benign-single-event",
        "label": 0,
        "window": {
            "event_count": 1,
            "counts_by_user": {"bob": 1},
            "counts_by_event_name": {"ListBuckets": 1},
        },
    }
    normal_attack = {
        "id": "non-fractured-attack",
        "label": 1,
        "window": {
            "event_count": 3,
            "counts_by_user": {"alice": 3},
            "simulation_malicious_event_count": 3,
            "simulation_counts_by_attack_family": {"cloud_recon": 3},
        },
    }

    with caplog.at_level("INFO"):
        cleaned = clean_fractured_threat_records(
            [fractured_attack, benign_single_event, normal_attack],
            policy="drop",
            min_event_count=2,
        )

    assert [record["id"] for record in cleaned] == ["benign-single-event", "non-fractured-attack"]
    assert "Dropped 1 fractured single-event threat windows." in caplog.messages


def test_clean_fractured_threat_records_relabels_single_event_attack_windows(caplog):
    with caplog.at_level("INFO"):
        cleaned = clean_fractured_threat_records(
            [_make_fractured_attack_record()],
            policy="relabel",
            min_event_count=2,
        )

    assert len(cleaned) == 1
    record = cleaned[0]
    assert record["label"] == 0
    assert resolve_record_label(record) == 0
    assert record["window"]["simulation_malicious_event_count"] == 0
    assert record["window"]["simulation_benign_event_count"] == 1
    assert record["window"]["simulation_counts_by_type"] == {"benign": 1}
    assert record["window"]["simulation_counts_by_attack_family"] == {}
    assert record["window"]["simulation_counts_by_scenario"] == {}
    assert "attack_family" not in record["window"]["metadata"]
    assert "scenario_name" not in record["window"]["metadata"]
    assert "primary_scenario" not in record["window"]["metadata"]
    assert "Relabeled 1 fractured single-event threat windows as benign." in caplog.messages


def test_network_flow_feature_schema_dispatch_and_text_serialization():
    window = {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        "event_count": 1,
        "dur": 2.0,
        "spkts": 10,
        "dpkts": 5,
        "sbytes": 1200,
        "dbytes": 300,
        "proto": "tcp",
        "service": "http",
        "state": "con",
        "srcip": "10.0.0.1",
        "dstip": "10.0.0.2",
        "counts_by_source_ip": {"10.0.0.1": 1},
        "counts_by_destination_ip": {"10.0.0.2": 1},
    }

    features = window_to_feature_dict(window)
    text = window_to_text(window)

    assert features["bytes_total"] == 1500.0
    assert features["packets_total"] == 15.0
    assert features["proto=tcp"] == 1.0
    assert "login_total" not in features
    assert "feature_schema=network_flow_v1" in text
    assert "proto=tcp" in text


def test_network_flow_records_without_label_default_to_zero():
    record = {
        "window": {
            "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        }
    }

    assert resolve_record_label(record) == 0
