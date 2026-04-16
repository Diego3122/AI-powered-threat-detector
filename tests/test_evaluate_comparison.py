import json
import shutil
import sys
import uuid
from pathlib import Path

from scripts import evaluate_comparison
from services.ml.ml_utils import NETWORK_FLOW_FEATURE_SCHEMA


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _make_network_flow_window(label: int) -> dict:
    return {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        "window_start_ms": 1_700_200_000_000 + label,
        "window_end_ms": 1_700_200_000_100 + label,
        "event_count": 1,
        "dur": 0.002 if label else 0.01,
        "spkts": 12 if label else 3,
        "dpkts": 9 if label else 2,
        "sbytes": 6000 if label else 200,
        "dbytes": 4000 if label else 100,
        "proto": "tcp",
        "service": "http",
        "state": "con",
        "metadata": {"label": label, "attack_cat": "Exploits" if label else "Normal"},
    }


def test_evaluate_comparison_disables_rule_baseline_for_network_flow(monkeypatch):
    root = Path("pytest_tmp_files")
    temp_dir = root / f"evaluate-comparison-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    data_path = temp_dir / "dataset.jsonl"
    out_path = temp_dir / "comparison.json"

    records = [
        {"id": "positive", "label_source": "dataset_ground_truth", "window": _make_network_flow_window(1)},
        {"id": "negative", "label_source": "dataset_ground_truth", "window": _make_network_flow_window(0)},
    ]
    _write_records(data_path, records)

    monkeypatch.setattr(
        evaluate_comparison,
        "evaluate_structured",
        lambda *_args, **_kwargs: {
            "model": "structured_baseline",
            "accuracy": 1.0,
            "positive_f1": 1.0,
            "latency_ms": 1.0,
            "n_samples": 2,
        },
    )
    monkeypatch.setattr(
        evaluate_comparison,
        "evaluate_tfidf",
        lambda *_args, **_kwargs: {
            "model": "tfidf",
            "accuracy": 0.5,
            "positive_f1": 0.5,
            "latency_ms": 2.0,
            "n_samples": 2,
        },
    )
    monkeypatch.setattr(
        evaluate_comparison,
        "evaluate_distilbert",
        lambda *_args, **_kwargs: {
            "model": "distilbert",
            "accuracy": 0.5,
            "positive_f1": 0.5,
            "latency_ms": 20.0,
            "n_samples": 2,
        },
    )

    argv = [
        "evaluate_comparison.py",
        "--data",
        str(data_path),
        "--output",
        str(out_path),
        "--split-mode",
        "none",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    try:
        evaluate_comparison.main()
        output = json.loads(out_path.read_text(encoding="utf-8"))
        assert output["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
        assert output["rule_baseline"]["disabled"] is True
        assert output["summary"]["rule_f1_delta"] is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
