import json
import shutil
import sys
import uuid
from pathlib import Path

import joblib

from scripts import train_baseline
from services.ml.ml_utils import NETWORK_FLOW_FEATURE_SCHEMA


def _make_unsw_record(label: int, index: int) -> dict:
    proto = "udp" if label else "tcp"
    service = "dns" if label else "http"
    sbytes = 6000 if label else 200
    dbytes = 150 if label else 1500
    spkts = 20 if label else 3
    dpkts = 2 if label else 12
    window = {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        "window_start_ms": 1_700_000_000_000 + index * 10,
        "window_end_ms": 1_700_000_000_005 + index * 10,
        "event_count": 5,
        "dur": 1.0,
        "sport": 4000 + index,
        "dsport": 53 if label else 443,
        "srcip": f"10.0.0.{index + 1}",
        "dstip": f"10.0.1.{index + 1}",
        "proto": proto,
        "service": service,
        "state": "int" if label else "con",
        "spkts": spkts,
        "dpkts": dpkts,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "counts_by_source_ip": {f"10.0.0.{index + 1}": 1},
        "counts_by_destination_ip": {f"10.0.1.{index + 1}": 1},
        "counts_by_proto": {proto: 1},
        "counts_by_service": {service: 1},
        "counts_by_state": {"int" if label else "con": 1},
        "metadata": {
            "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
            "attack_cat": "DoS" if label else "Normal",
        },
    }
    return {
        "id": f"unsw-{index}",
        "window_start_ms": window["window_start_ms"],
        "window_end_ms": window["window_end_ms"],
        "label": label,
        "label_source": "dataset_ground_truth",
        "window": window,
    }


def test_train_baseline_network_flow_disables_cloudtrail_rule_and_family_paths():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"train-baseline-unsw-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    source_train = temp_dir / "unsw_nb15_train.jsonl"
    source_test = temp_dir / "unsw_nb15_test.jsonl"
    out_path = temp_dir / "baseline.pkl"
    manifest_path = out_path.with_suffix(".manifest.json")

    train_records = [
        _make_unsw_record(0, 1),
        _make_unsw_record(1, 2),
        _make_unsw_record(0, 3),
        _make_unsw_record(1, 4),
    ]
    test_records = [
        _make_unsw_record(0, 5),
        _make_unsw_record(1, 6),
        _make_unsw_record(0, 7),
        _make_unsw_record(1, 8),
    ]
    source_train.write_text("\n".join(json.dumps(record) for record in train_records) + "\n", encoding="utf-8")
    source_test.write_text("\n".join(json.dumps(record) for record in test_records) + "\n", encoding="utf-8")

    try:
        argv = [
            "train_baseline.py",
            "--data",
            str(source_train),
            "--data",
            str(source_test),
            "--out",
            str(out_path),
            "--holdout-source",
            "unsw_nb15_test",
            "--split-mode",
            "stratified",
            "--fractured-threat-policy",
            "off",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            assert train_baseline.main() == 0
        finally:
            sys.argv = old_argv

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_bundle = joblib.load(out_path)
        assert manifest["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
        assert model_bundle["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
        assert model_bundle["structured_family_models"] == {}
        assert model_bundle["structured_family_vectorizers"] == {}
        assert model_bundle["structured_family_thresholds"] == {}
        assert manifest["metrics"]["rule_baseline"]["disabled"] is True
        assert manifest["holdout_source"] == "unsw_nb15_test"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
