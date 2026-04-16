import csv
import shutil
import uuid
from pathlib import Path

from scripts import build_unsw_nb15_dataset
from services.ml.ml_utils import NETWORK_FLOW_FEATURE_SCHEMA


def test_build_unsw_nb15_dataset_rows_uses_ground_truth_labels():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"unsw-adapter-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    csv_path = temp_dir / "UNSW_NB15_train.csv"

    rows = [
        {
            "id": "1",
            "Stime": "1700000000",
            "Ltime": "1700000002",
            "dur": "2.0",
            "srcip": "10.0.0.1",
            "dstip": "10.0.0.2",
            "sport": "1000",
            "dsport": "443",
            "proto": "tcp",
            "service": "http",
            "state": "CON",
            "spkts": "8",
            "dpkts": "4",
            "sbytes": "1200",
            "dbytes": "300",
            "attack_cat": "Normal",
            "label": "0",
        },
        {
            "id": "2",
            "Stime": "1700000010",
            "Ltime": "1700000011",
            "dur": "1.0",
            "srcip": "10.0.0.5",
            "dstip": "10.0.0.9",
            "sport": "4000",
            "dsport": "53",
            "proto": "udp",
            "service": "dns",
            "state": "INT",
            "spkts": "20",
            "dpkts": "2",
            "sbytes": "5000",
            "dbytes": "120",
            "attack_cat": "DoS",
            "label": "1",
        },
    ]

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        dataset_rows = build_unsw_nb15_dataset.build_dataset_rows(csv_path, dataset_source="unsw_nb15_train")
        assert len(dataset_rows) == 2
        assert [row["label"] for row in dataset_rows] == [0, 1]
        assert all(row["label_source"] == "dataset_ground_truth" for row in dataset_rows)
        assert all(row["label_quality_tier"] == "high" for row in dataset_rows)
        assert dataset_rows[0]["window"]["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
        assert dataset_rows[1]["window"]["counts_by_proto"] == {"udp": 1}
        assert "feature_schema=network_flow_v1" in dataset_rows[1]["text"]
        assert "attack_cat" not in dataset_rows[1]["text"].lower()
        assert "dos" not in dataset_rows[1]["text"].lower()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_unsw_nb15_dataset_skips_rows_without_labels():
    root = Path("pytest_tmp_files")
    temp_dir = root / f"unsw-adapter-missing-label-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    csv_path = temp_dir / "UNSW_NB15_train.csv"

    rows = [
        {"id": "1", "Stime": "1700000000", "dur": "1.0", "proto": "tcp", "service": "http", "state": "con", "label": ""},
        {"id": "2", "Stime": "1700000001", "dur": "1.0", "proto": "tcp", "service": "http", "state": "con", "label": "0"},
    ]

    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        dataset_rows = build_unsw_nb15_dataset.build_dataset_rows(csv_path, dataset_source="unsw_nb15_train")
        assert len(dataset_rows) == 1
        assert dataset_rows[0]["id"] == "2"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
