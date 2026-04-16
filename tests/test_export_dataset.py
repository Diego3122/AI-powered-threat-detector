import json
import shutil
import uuid
from pathlib import Path
from services.ml import export_dataset


def make_window(start, end, counts=None, failed=None):
    return {
        "window_start_ms": start,
        "window_end_ms": end,
        "counts_by_user": counts or {},
        "failed_no_mfa_by_user": failed or {},
    }


def test_label_from_window_uses_simulation_metadata_when_present():
    w_attack = {"simulation_malicious_event_count": 2, "simulation_benign_event_count": 0}
    assert export_dataset.label_from_window(w_attack, 3) == 1

    w_benign = {"simulation_malicious_event_count": 0, "simulation_benign_event_count": 2}
    assert export_dataset.label_from_window(w_benign, 3) == 0

    w_no_sim = {"counts_by_user": {"alice": 1}}
    assert export_dataset.label_from_window(w_no_sim, 3) == 0


def test_export_dataset_from_file():
    windows = [
        {"window_start_ms": 0, "window_end_ms": 10000, "simulation_malicious_event_count": 0, "simulation_benign_event_count": 1},
        {"window_start_ms": 10000, "window_end_ms": 20000, "simulation_malicious_event_count": 1, "simulation_benign_event_count": 0},
    ]

    temp_root = Path("pytest_tmp_files")
    temp_path = temp_root / f"export-dataset-{uuid.uuid4().hex}"
    temp_path.mkdir(parents=True, exist_ok=False)
    try:
        in_file = temp_path / "windows.jsonl"
        out_file = temp_path / "out.jsonl"
        with open(in_file, "w", encoding="utf-8") as f:
            for w in windows:
                f.write(json.dumps(w) + "\n")

        # Run export_dataset in file mode
        export_dataset_args = ["prog", "--input-file", str(in_file), "--out", str(out_file), "--max", "10"]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = export_dataset_args
            export_dataset.main()
        finally:
            sys.argv = old_argv

        data = [json.loads(line) for line in open(out_file, "r", encoding="utf-8")]
        assert len(data) == 2
        assert data[0]["label"] == 0
        assert data[1]["label"] == 1
        assert data[0]["label_quality_tier"] == "high"
        assert data[1]["label_quality_tier"] == "high"
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_export_dataset_prefers_simulation_metadata_labels():
    window = {
        "window_start_ms": 0,
        "window_end_ms": 10000,
        "counts_by_user": {"alice": 2},
        "failed_no_mfa_by_user": {"alice": 1},
        "simulation_malicious_event_count": 1,
        "simulation_benign_event_count": 2,
        "simulation_counts_by_type": {"attack": 1, "benign": 2},
        "simulation_counts_by_attack_family": {"password_spray": 1},
    }

    assert export_dataset.label_from_window(window, 3) == 1


def test_export_dataset_stamps_metadata_label_quality_tier():
    windows = [
        {
            "window_start_ms": 0,
            "window_end_ms": 10000,
            "counts_by_user": {"alice": 2},
            "failed_no_mfa_by_user": {"alice": 1},
            "simulation_malicious_event_count": 1,
            "simulation_benign_event_count": 0,
            "simulation_counts_by_type": {"attack": 1},
        }
    ]

    temp_root = Path("pytest_tmp_files")
    temp_path = temp_root / f"export-dataset-tier-{uuid.uuid4().hex}"
    temp_path.mkdir(parents=True, exist_ok=False)
    try:
        in_file = temp_path / "windows.jsonl"
        out_file = temp_path / "out.jsonl"
        with open(in_file, "w", encoding="utf-8") as f:
            for w in windows:
                f.write(json.dumps(w) + "\n")

        export_dataset_args = ["prog", "--input-file", str(in_file), "--out", str(out_file), "--max", "10"]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = export_dataset_args
            export_dataset.main()
        finally:
            sys.argv = old_argv

        data = [json.loads(line) for line in open(out_file, "r", encoding="utf-8")]
        assert data[0]["label_source"] == "dataset_ground_truth"
        assert data[0]["label_quality_tier"] == "high"
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
