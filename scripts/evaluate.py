#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys

import joblib

from services.ml.ml_utils import (
    clean_fractured_threat_records,
    compute_metrics,
    extract_window,
    load_jsonl_records,
    record_to_window_text,
    resolve_record_label,
    resolve_window_feature_schema,
    score_structured_inputs,
    split_records,
)


def _collect_feature_schemas(records: list[dict]) -> set[str]:
    schemas: set[str] = set()
    for record in records:
        schemas.add(resolve_window_feature_schema(extract_window(record)))
    return schemas


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(description="Evaluate a trained threat detector model")
    ap.add_argument("--model", default="models/baseline.pkl", help="Saved model path")
    ap.add_argument("--data", default="data/regenerated/windows_dataset_sample.jsonl", help="Test dataset JSONL file")
    ap.add_argument("--out-metrics", default="models/metrics.json", help="Output metrics JSON")
    ap.add_argument("--test-size", type=int, default=0, help="Use only first N samples (0 = all)")
    ap.add_argument("--split-mode", choices=["time", "stratified", "none"], default="time", help="How to split the dataset before evaluation")
    ap.add_argument("--split-fraction", type=float, default=0.2, help="Hold-out fraction when split-mode is enabled")
    ap.add_argument(
        "--fractured-threat-policy",
        choices=["off", "drop", "relabel"],
        default="off",
        help="How to handle attack-labeled windows whose event_count is below the minimum threshold.",
    )
    ap.add_argument(
        "--fractured-threat-min-events",
        type=int,
        default=2,
        help="Minimum event_count required before an attack-labeled window is kept as-is.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)

    model_dict = joblib.load(args.model)
    print(f"Loaded model from {args.model}")
    quality_gate = model_dict.get("quality_gate")
    if isinstance(quality_gate, dict):
        state = "PASS" if quality_gate.get("promotion_ready") else "BLOCKED"
        print(f"Promotion gate: {state}")
        for blocker in quality_gate.get("blockers", []):
            print(f"  - {blocker}")

    records = load_jsonl_records(args.data, max_samples=args.test_size)
    records = clean_fractured_threat_records(
        records,
        policy=args.fractured_threat_policy,
        min_event_count=args.fractured_threat_min_events,
    )
    if len(records) == 0:
        print("ERROR: No test data loaded", file=sys.stderr)
        sys.exit(1)

    feature_schemas = _collect_feature_schemas(records)
    if len(feature_schemas) != 1:
        print(
            "ERROR: mixed feature schemas detected in the same evaluation run: "
            + ", ".join(sorted(feature_schemas)),
            file=sys.stderr,
        )
        return 2
    active_feature_schema = next(iter(feature_schemas))

    if args.split_mode == "none":
        train_records = records
        test_records = records
        actual_split_mode = "none"
    else:
        split_result = split_records(
            records,
            test_size=args.split_fraction,
            split_mode=args.split_mode,
            seed=42,
        )
        train_records, test_records = split_result
        actual_split_mode = split_result.actual_split_mode

    print(f"Loaded {len(records)} samples")
    print(f"Train: {len(train_records)}, Test: {len(test_records)} (split={actual_split_mode}; requested={args.split_mode})")
    print(f"Feature schema: {active_feature_schema}")

    test_texts = [record_to_window_text(record)[1] for record in test_records]
    test_windows = [record_to_window_text(record)[0] for record in test_records]
    test_labels = [resolve_record_label(record) for record in test_records]

    text_model = model_dict["model"]
    text_vectorizer = model_dict["vectorizer"]
    text_scores = text_model.predict_proba(text_vectorizer.transform(test_texts))[:, 1].tolist()
    text_pred = [1 if score >= 0.5 else 0 for score in text_scores]
    text_metrics = compute_metrics(test_labels, text_pred, text_scores)

    structured_results = score_structured_inputs(model_dict, test_windows)
    structured_scores = [result["score"] for result in structured_results]
    structured_pred = [int(result["label"]) for result in structured_results]
    structured_thresholds = sorted({float(result.get("threshold", model_dict.get("structured_threshold", 0.5))) for result in structured_results})
    structured_metrics = compute_metrics(test_labels, structured_pred, structured_scores)
    structured_metrics["threshold"] = structured_thresholds[0] if len(structured_thresholds) == 1 else None
    structured_metrics["threshold_mode"] = "per_window_family_thresholds" if len(structured_thresholds) > 1 else "global_threshold"
    structured_metrics["thresholds_used"] = structured_thresholds

    metrics = dict(structured_metrics)
    metrics["model"] = "structured_baseline"
    metrics["split_mode"] = actual_split_mode
    metrics["split_mode_requested"] = args.split_mode
    metrics["feature_schema"] = active_feature_schema
    metrics["feature_version"] = model_dict.get("feature_version", 1)
    if isinstance(quality_gate, dict):
        metrics["quality_gate"] = quality_gate
    metrics["comparisons"] = {
        "structured_baseline": structured_metrics,
        "text_tfidf": text_metrics,
    }

    print("\nStructured baseline metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Positive Precision: {metrics['positive_precision']:.4f}")
    print(f"  Positive Recall:    {metrics['positive_recall']:.4f}")
    print(f"  Positive F1:        {metrics['positive_f1']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    print("\nComparison:")
    print(f"  Structured F1:      {structured_metrics['positive_f1']:.4f}")
    print(f"  Text TF-IDF F1:     {text_metrics['positive_f1']:.4f}")

    os.makedirs(os.path.dirname(args.out_metrics) or ".", exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {args.out_metrics}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
