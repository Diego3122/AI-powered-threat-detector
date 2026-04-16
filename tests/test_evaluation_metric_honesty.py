import json
import sys
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np

from scripts import evaluate, evaluate_credibility
from services.ml.ml_utils import NETWORK_FLOW_FEATURE_SCHEMA


class _DummyVectorizer:
    def transform(self, texts):
        return list(texts)


class _DummyModel:
    def predict_proba(self, matrix):
        return np.array([[0.4, 0.6] for _ in matrix], dtype=float)


class _SplitResult:
    def __init__(self, train_records, test_records, actual_split_mode):
        self.train_records = train_records
        self.test_records = test_records
        self.actual_split_mode = actual_split_mode

    def __iter__(self):
        yield self.train_records
        yield self.test_records


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _make_window(label: int, *, event_name: str, campaign_signature: str, attack_family: str) -> dict:
    return {
        "window_start_ms": 1_700_000_000_000 + label,
        "window_end_ms": 1_700_000_000_100 + label,
        "event_count": 3,
        "counts_by_user": {f"user-{label}": 3},
        "counts_by_login_user": {f"user-{label}": 3},
        "failed_no_mfa_by_user": {f"user-{label}": 2} if label else {},
        "counts_by_source_ip": {"198.51.100.10": 1},
        "counts_by_user_agent": {"AWS CLI/2.15": 1},
        "counts_by_event_name": {event_name: 1},
        "counts_by_login_result": {"Success": 1, "Failure": 1 if label else 0},
        "counts_by_mfa_used": {"Yes": 1, "No": 1 if label else 0},
        "simulation_malicious_event_count": 1 if label else 0,
        "simulation_benign_event_count": 0 if label else 1,
        "simulation_counts_by_type": {"attack": 1} if label else {"benign": 1},
        "simulation_counts_by_attack_family": {attack_family: 1} if label else {},
        "metadata": {
            "campaign_signature": campaign_signature,
            "scenario_signature": f"{attack_family}-scenario",
            "window_group_id": f"{campaign_signature}-group",
            "attack_family": attack_family,
            "primary_scenario": f"{attack_family}-scenario",
        },
    }


def _make_network_flow_window(label: int) -> dict:
    return {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        "window_start_ms": 1_700_100_000_000 + label,
        "window_end_ms": 1_700_100_000_100 + label,
        "event_count": 1,
        "dur": 0.001 if label else 0.01,
        "spkts": 10 if label else 2,
        "dpkts": 8 if label else 2,
        "sbytes": 4096 if label else 128,
        "dbytes": 2048 if label else 64,
        "proto": "tcp",
        "service": "http",
        "state": "con",
        "metadata": {
            "label": label,
            "attack_cat": "DoS" if label else "Normal",
        },
    }


def test_evaluate_uses_structured_labels_and_honest_split_mode(monkeypatch):
    root = Path("pytest_tmp_files")
    temp_dir = root / f"evaluate-metric-honesty-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    data_path = temp_dir / "dataset.jsonl"
    model_path = temp_dir / "baseline.pkl"
    out_metrics = temp_dir / "metrics.json"

    records = [
        {
            "id": "positive",
            "label": 1,
            "label_source": "metadata",
            "window": _make_window(1, event_name="GetCallerIdentity", campaign_signature="camp-a", attack_family="session_hijack"),
        },
        {
            "id": "negative",
            "label": 0,
            "label_source": "metadata",
            "window": _make_window(0, event_name="ListBuckets", campaign_signature="camp-b", attack_family="login_abuse"),
        },
    ]
    _write_records(data_path, records)
    model_path.write_text("stub", encoding="utf-8")

    split_result = _SplitResult(records, records, "stratified")

    monkeypatch.setattr(joblib, "load", lambda _: {"model": _DummyModel(), "vectorizer": _DummyVectorizer(), "structured_threshold": 0.5})
    monkeypatch.setattr(evaluate, "split_records", lambda *args, **kwargs: split_result)
    monkeypatch.setattr(
        evaluate,
        "score_structured_inputs",
        lambda model_dict, windows: [
            {
                "label": int(window.get("simulation_malicious_event_count", 0) > 0),
                "score": 0.1 if window.get("simulation_malicious_event_count", 0) else 0.9,
                "threshold": 0.2 if window.get("simulation_malicious_event_count", 0) else 0.8,
                "model": "structured_baseline",
            }
            for window in windows
        ],
    )

    argv = [
        "evaluate.py",
        "--model",
        str(model_path),
        "--data",
        str(data_path),
        "--out-metrics",
        str(out_metrics),
        "--split-mode",
        "time",
        "--split-fraction",
        "0.5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    try:
        assert evaluate.main() == 0

        metrics = json.loads(out_metrics.read_text(encoding="utf-8"))
        assert metrics["split_mode_requested"] == "time"
        assert metrics["split_mode"] == "stratified"
        assert metrics["comparisons"]["structured_baseline"]["positive_f1"] == 1.0
        assert metrics["comparisons"]["structured_baseline"]["threshold_mode"] == "per_window_family_thresholds"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evaluate_disables_cloudtrail_rule_baselines_for_network_flow(monkeypatch):
    root = Path("pytest_tmp_files")
    temp_dir = root / f"evaluate-network-flow-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    data_path = temp_dir / "dataset.jsonl"
    model_path = temp_dir / "baseline.pkl"
    out_metrics = temp_dir / "metrics.json"

    records = [
        {
            "id": "positive",
            "label_source": "dataset_ground_truth",
            "window": _make_network_flow_window(1),
        },
        {
            "id": "negative",
            "label_source": "dataset_ground_truth",
            "window": _make_network_flow_window(0),
        },
    ]
    _write_records(data_path, records)
    model_path.write_text("stub", encoding="utf-8")

    split_result = _SplitResult(records, records, "stratified")
    monkeypatch.setattr(joblib, "load", lambda _: {"model": _DummyModel(), "vectorizer": _DummyVectorizer(), "structured_threshold": 0.5})
    monkeypatch.setattr(evaluate, "split_records", lambda *args, **kwargs: split_result)
    monkeypatch.setattr(
        evaluate,
        "score_structured_inputs",
        lambda model_dict, windows: [
            {
                "label": int(window.get("metadata", {}).get("label", 0)),
                "score": 0.95 if window.get("metadata", {}).get("label", 0) else 0.05,
                "threshold": 0.5,
                "model": "structured_baseline",
            }
            for window in windows
        ],
    )

    argv = [
        "evaluate.py",
        "--model",
        str(model_path),
        "--data",
        str(data_path),
        "--out-metrics",
        str(out_metrics),
        "--split-mode",
        "time",
        "--split-fraction",
        "0.5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    try:
        assert evaluate.main() == 0

        metrics = json.loads(out_metrics.read_text(encoding="utf-8"))
        assert metrics["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
        assert "rule_baseline" not in metrics["comparisons"]
        assert "legacy_rule" not in metrics["comparisons"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evaluate_credibility_defaults_to_grouped_synthetic_cv(monkeypatch):
    root = Path("pytest_tmp_files")
    temp_dir = root / f"credibility-metric-honesty-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    data_path = temp_dir / "credibility.jsonl"
    out_path = temp_dir / "credibility_report.json"

    records = [
        {
            "id": "api-positive",
            "label": 1,
            "label_source": "metadata",
            "window": _make_window(1, event_name="GetCallerIdentity", campaign_signature="camp-a", attack_family="session_hijack"),
        },
        {
            "id": "api-negative",
            "label": 0,
            "label_source": "metadata",
            "window": _make_window(0, event_name="ListBuckets", campaign_signature="camp-a", attack_family="session_hijack"),
        },
        {
            "id": "login-positive",
            "label": 1,
            "label_source": "metadata",
            "window": _make_window(1, event_name="ConsoleLogin", campaign_signature="camp-b", attack_family="login_abuse"),
        },
        {
            "id": "login-negative",
            "label": 0,
            "label_source": "metadata",
            "window": _make_window(0, event_name="ConsoleLogin", campaign_signature="camp-b", attack_family="login_abuse"),
        },
    ]
    _write_records(data_path, records)

    captured = {}

    def fake_score_structured_inputs(model_dict, windows):
        captured["bundle_keys"] = set(model_dict)
        captured["family_models"] = set(model_dict.get("structured_family_models", {}))
        captured["family_thresholds"] = dict(model_dict.get("structured_family_thresholds", {}))
        results = []
        for window in windows:
            label = int(window.get("simulation_malicious_event_count", 0) > 0)
            results.append(
                {
                    "label": label,
                    "score": 0.9 if label else 0.1,
                    "threshold": 0.6 if label else 0.4,
                    "model": "structured_baseline",
                }
            )
        return results

    monkeypatch.setattr(evaluate_credibility, "score_structured_inputs", fake_score_structured_inputs)

    argv = [
        "evaluate_credibility.py",
        "--data",
        str(data_path),
        "--out",
        str(out_path),
        "--folds",
        "2",
        "--repeats",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    try:
        assert evaluate_credibility.main() == 0

        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["evaluation"]["group_by_requested"] == "auto"
        assert report["evaluation"]["group_by"] == "campaign_signature"
        assert report["evaluation"]["scheme"] == "stratified_group_kfold"
        assert "structured_family_models" in captured["bundle_keys"]
        assert "structured_family_thresholds" in captured["bundle_keys"]
        assert any("grouped CV resolved to campaign_signature" in warning for warning in report["warnings"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evaluate_credibility_disables_legacy_rule_metrics_for_network_flow(monkeypatch):
    root = Path("pytest_tmp_files")
    temp_dir = root / f"credibility-network-flow-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    data_path = temp_dir / "credibility.jsonl"
    out_path = temp_dir / "credibility_report.json"

    records = [
        {
            "id": "nf-positive-a",
            "label_source": "dataset_ground_truth",
            "window": _make_network_flow_window(1),
        },
        {
            "id": "nf-negative-a",
            "label_source": "dataset_ground_truth",
            "window": _make_network_flow_window(0),
        },
        {
            "id": "nf-positive-b",
            "label_source": "dataset_ground_truth",
            "window": _make_network_flow_window(1),
        },
        {
            "id": "nf-negative-b",
            "label_source": "dataset_ground_truth",
            "window": _make_network_flow_window(0),
        },
    ]
    _write_records(data_path, records)

    captured = {}

    def fake_score_structured_inputs(model_dict, windows):
        captured["feature_schema"] = model_dict.get("feature_schema")
        captured["family_models"] = dict(model_dict.get("structured_family_models", {}))
        captured["family_thresholds"] = dict(model_dict.get("structured_family_thresholds", {}))
        return [
            {
                "label": int(window.get("metadata", {}).get("label", 0)),
                "score": 0.9 if window.get("metadata", {}).get("label", 0) else 0.1,
                "threshold": 0.5,
                "model": "structured_baseline",
            }
            for window in windows
        ]

    monkeypatch.setattr(evaluate_credibility, "score_structured_inputs", fake_score_structured_inputs)

    argv = [
        "evaluate_credibility.py",
        "--data",
        str(data_path),
        "--out",
        str(out_path),
        "--folds",
        "2",
        "--repeats",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    try:
        assert evaluate_credibility.main() == 0

        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
        assert "legacy_rule_match_rate" not in report
        assert "rule_baseline" not in report["summary"]
        assert captured["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
        assert captured["family_models"] == {}
        assert captured["family_thresholds"] == {}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
