import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import joblib

from scripts import train_baseline


def _make_record(label: int, srcip: str) -> dict:
    window = {
        "feature_schema": "network_flow_v1",
        "window_start_ms": 1_700_000_000_000,
        "window_end_ms": 1_700_000_010_000,
        "event_count": 3,
        "dur": 1.5 if label else 0.1,
        "sbytes": 10_000 if label else 100,
        "dbytes": 5_000 if label else 50,
        "spkts": 20 if label else 2,
        "dpkts": 10 if label else 1,
        "proto": "tcp" if label else "udp",
        "service": "http" if label else "-",
        "state": "con",
        "srcip": srcip,
        "dstip": "192.168.1.1",
        "counts_by_source_ip": {srcip: 1},
        "counts_by_destination_ip": {"192.168.1.1": 1},
        "metadata": {},
    }
    return {
        "id": f"{srcip}-{label}",
        "label": label,
        "label_source": "dataset_ground_truth",
        "window_start_ms": window["window_start_ms"],
        "window_end_ms": window["window_end_ms"],
        "window": window,
    }


def _make_fractured_attack_record(srcip: str) -> dict:
    window = {
        "feature_schema": "network_flow_v1",
        "window_start_ms": 1_700_000_100_000,
        "window_end_ms": 1_700_000_100_001,
        "event_count": 1,
        "dur": 0.001,
        "sbytes": 200,
        "dbytes": 0,
        "spkts": 1,
        "dpkts": 0,
        "proto": "tcp",
        "service": "-",
        "state": "INT",
        "srcip": srcip,
        "dstip": "192.168.1.1",
        "counts_by_source_ip": {srcip: 1},
        "counts_by_destination_ip": {"192.168.1.1": 1},
        "metadata": {"attack_cat": "Reconnaissance"},
    }
    return {
        "id": f"{srcip}-fractured-attack",
        "label": 1,
        "label_source": "dataset_ground_truth",
        "window_start_ms": window["window_start_ms"],
        "window_end_ms": window["window_end_ms"],
        "window": window,
    }


def test_split_records_for_training_respects_holdout_source():
    records = [
        {"label": 0, "dataset_source": "train_a"},
        {"label": 1, "dataset_source": "train_a"},
        {"label": 0, "dataset_source": "eval_b"},
        {"label": 1, "dataset_source": "eval_b"},
    ]

    train_records, test_records = train_baseline._split_records_for_training(
        records,
        test_size=0.2,
        split_mode="stratified",
        seed=42,
        holdout_source="eval_b",
    )

    assert {record["dataset_source"] for record in train_records} == {"train_a"}
    assert {record["dataset_source"] for record in test_records} == {"eval_b"}


def test_train_baseline_writes_multi_source_manifest():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"train-baseline-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    source_a = temp_dir / "source_a.jsonl"
    source_b = temp_dir / "source_b.jsonl"
    out_path = temp_dir / "baseline.pkl"
    manifest_path = out_path.with_suffix(".manifest.json")

    source_a_records = [
        _make_record(0, "10.0.0.1"),
        _make_record(1, "10.0.0.1"),
        _make_record(0, "10.0.0.2"),
        _make_record(1, "10.0.0.2"),
    ]
    source_b_records = [
        _make_record(0, "10.0.0.3"),
        _make_record(1, "10.0.0.3"),
        _make_record(0, "10.0.0.4"),
        _make_record(1, "10.0.0.4"),
    ]
    source_a.write_text("\n".join(json.dumps(record) for record in source_a_records) + "\n", encoding="utf-8")
    source_b.write_text("\n".join(json.dumps(record) for record in source_b_records) + "\n", encoding="utf-8")

    try:
        argv = [
            "train_baseline.py",
            "--data",
            str(source_a),
            "--data",
            str(source_b),
            "--out",
            str(out_path),
            "--holdout-source",
            "source_b",
            "--split-mode",
            "stratified",
            "--fractured-threat-policy",
            "drop",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            assert train_baseline.main() == 0
        finally:
            sys.argv = old_argv

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        joblib.load(out_path)
        assert manifest["feature_version"] == 7
        assert manifest["holdout_source"] == "source_b"
        assert manifest["dataset_source_counts"] == {"source_a": 4, "source_b": 4}
        assert manifest["train_source_counts"] == {"source_a": 4}
        assert manifest["test_source_counts"] == {"source_b": 4}
        assert len(manifest["data_paths"]) == 2
        assert manifest["quality_gate"]["promotion_ready"] is True
        assert manifest["quality_gate"]["blocker_codes"] == []
        assert "structured_family_train_counts" in manifest
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_baseline_blocks_promotion_when_holdout_labels_are_weak():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"train-baseline-weak-holdout-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    source_a = temp_dir / "source_a.jsonl"
    source_b = temp_dir / "source_b.jsonl"
    out_path = temp_dir / "baseline.pkl"
    manifest_path = out_path.with_suffix(".manifest.json")

    source_a_records = [
        _make_record(0, "10.0.0.1"),
        _make_record(1, "10.0.0.1"),
        _make_record(0, "10.0.0.2"),
        _make_record(1, "10.0.0.2"),
    ]
    source_b_records = [
        {key: value for key, value in _make_record(0, "10.0.0.3").items() if key != "label_source"},
        {key: value for key, value in _make_record(1, "10.0.0.3").items() if key != "label_source"},
        {key: value for key, value in _make_record(0, "10.0.0.4").items() if key != "label_source"},
        {key: value for key, value in _make_record(1, "10.0.0.4").items() if key != "label_source"},
    ]
    source_a.write_text("\n".join(json.dumps(record) for record in source_a_records) + "\n", encoding="utf-8")
    source_b.write_text("\n".join(json.dumps(record) for record in source_b_records) + "\n", encoding="utf-8")

    try:
        argv = [
            "train_baseline.py",
            "--data",
            str(source_a),
            "--data",
            str(source_b),
            "--out",
            str(out_path),
            "--holdout-source",
            "source_b",
            "--split-mode",
            "stratified",
            "--fractured-threat-policy",
            "drop",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            assert train_baseline.main() == 0
        finally:
            sys.argv = old_argv

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert manifest["quality_gate"]["promotion_ready"] is False
        assert "weak_holdout_labels" in manifest["quality_gate"]["blocker_codes"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_baseline_requires_holdout_source_for_multi_source_training():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"train-baseline-missing-holdout-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    source_a = temp_dir / "source_a.jsonl"
    source_b = temp_dir / "source_b.jsonl"

    source_a_records = [_make_record(0, "10.0.0.1"), _make_record(1, "10.0.0.1")]
    source_b_records = [_make_record(0, "10.0.0.2"), _make_record(1, "10.0.0.2")]
    source_a.write_text("\n".join(json.dumps(record) for record in source_a_records) + "\n", encoding="utf-8")
    source_b.write_text("\n".join(json.dumps(record) for record in source_b_records) + "\n", encoding="utf-8")

    try:
        argv = [
            "train_baseline.py",
            "--data",
            str(source_a),
            "--data",
            str(source_b),
            "--out",
            str(temp_dir / "baseline.pkl"),
            "--split-mode",
            "stratified",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            assert train_baseline.main() == 2
        finally:
            sys.argv = old_argv
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_baseline_blocks_runtime_bundle_overwrite_when_promotion_fails():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"train-baseline-runtime-block-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    source_a = temp_dir / "source_a.jsonl"
    source_b = temp_dir / "source_b.jsonl"
    models_dir = temp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "baseline.pkl"
    manifest_path = out_path.with_suffix(".manifest.json").resolve()

    source_a_records = [
        _make_record(0, "10.0.0.1"),
        _make_record(1, "10.0.0.1"),
        _make_record(0, "10.0.0.2"),
        _make_record(1, "10.0.0.2"),
    ]
    source_b_records = [
        {key: value for key, value in _make_record(0, "10.0.0.3").items() if key != "label_source"},
        {key: value for key, value in _make_record(1, "10.0.0.3").items() if key != "label_source"},
        {key: value for key, value in _make_record(0, "10.0.0.4").items() if key != "label_source"},
        {key: value for key, value in _make_record(1, "10.0.0.4").items() if key != "label_source"},
    ]
    source_a.write_text("\n".join(json.dumps(record) for record in source_a_records) + "\n", encoding="utf-8")
    source_b.write_text("\n".join(json.dumps(record) for record in source_b_records) + "\n", encoding="utf-8")

    old_cwd = Path.cwd()
    source_a_abs = source_a.resolve()
    source_b_abs = source_b.resolve()
    out_path_abs = out_path.resolve()
    try:
        os.chdir(temp_dir)
        argv = [
            "train_baseline.py",
            "--data",
            str(source_a_abs),
            "--data",
            str(source_b_abs),
            "--out",
            "models/baseline.pkl",
            "--holdout-source",
            "source_b",
            "--split-mode",
            "stratified",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            assert train_baseline.main() == 2
        finally:
            sys.argv = old_argv

        assert manifest_path.exists()
        assert not out_path_abs.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["quality_gate"]["promotion_ready"] is False
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_baseline_drops_fractured_attack_windows_before_split():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"train-baseline-fractured-drop-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    source_a = temp_dir / "source_a.jsonl"
    source_b = temp_dir / "source_b.jsonl"
    out_path = temp_dir / "baseline.pkl"
    manifest_path = out_path.with_suffix(".manifest.json")

    source_a_records = [
        _make_record(0, "10.0.0.1"),
        _make_record(1, "10.0.0.1"),
        _make_record(0, "10.0.0.2"),
        _make_record(1, "10.0.0.2"),
        _make_fractured_attack_record("10.0.0.1"),
    ]
    source_b_records = [
        _make_record(0, "10.0.0.3"),
        _make_record(1, "10.0.0.3"),
        _make_record(0, "10.0.0.4"),
        _make_record(1, "10.0.0.4"),
        _make_fractured_attack_record("10.0.0.3"),
    ]
    source_a.write_text("\n".join(json.dumps(record) for record in source_a_records) + "\n", encoding="utf-8")
    source_b.write_text("\n".join(json.dumps(record) for record in source_b_records) + "\n", encoding="utf-8")

    try:
        argv = [
            "train_baseline.py",
            "--data",
            str(source_a),
            "--data",
            str(source_b),
            "--out",
            str(out_path),
            "--holdout-source",
            "source_b",
            "--split-mode",
            "stratified",
            "--fractured-threat-policy",
            "drop",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            assert train_baseline.main() == 0
        finally:
            sys.argv = old_argv

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        model_bundle = joblib.load(out_path)
        assert manifest["n_samples"] == 8
        assert manifest["dataset_source_counts"] == {"source_a": 4, "source_b": 4}
        assert manifest["fractured_threat_policy"] == "drop"
        assert manifest["fractured_threat_min_events"] == 2
        assert model_bundle["fractured_threat_policy"] == "drop"
        assert model_bundle["fractured_threat_min_events"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
