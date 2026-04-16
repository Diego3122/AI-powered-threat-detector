#!/usr/bin/env python
import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.ml.ml_utils import (
    NETWORK_FLOW_FEATURE_SCHEMA,
    clean_fractured_threat_records,
    compute_metrics,
    extract_window,
    find_best_threshold,
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


def evaluate_tfidf(model_path: str, texts: list[str], labels: list[int]) -> dict:
    print("\nEvaluating TF-IDF Model...")

    model_dict = joblib.load(model_path)
    model = model_dict["model"]
    vectorizer = model_dict["vectorizer"]

    X = vectorizer.transform(texts)
    start = time.time()
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    elapsed = time.time() - start
    latency_per_sample = (elapsed / len(texts)) * 1000 if texts else 0.0

    results = compute_metrics(labels, y_pred.tolist(), y_proba.tolist())
    results.update(
        {
            "model": "tfidf",
            "latency_ms": round(latency_per_sample, 2),
            "total_time_s": round(elapsed, 2),
            "n_samples": len(texts),
        }
    )
    return results


def evaluate_structured(model_path: str, windows: list[dict], labels: list[int]) -> dict:
    print("\nEvaluating Structured Baseline...")

    model_dict = joblib.load(model_path)
    start = time.time()
    structured_results = score_structured_inputs(model_dict, windows)
    elapsed = time.time() - start
    latency_per_sample = (elapsed / len(windows)) * 1000 if windows else 0.0

    y_pred = [result["label"] for result in structured_results]
    y_proba = [result["score"] for result in structured_results]
    results = compute_metrics(labels, y_pred, y_proba)
    results.update(
        {
            "model": "structured_baseline",
            "threshold": float(model_dict.get("structured_threshold", 0.5)),
            "latency_ms": round(latency_per_sample, 2),
            "total_time_s": round(elapsed, 2),
            "n_samples": len(windows),
        }
    )
    return results


def evaluate_distilbert(model_dir: str, texts: list[str], labels: list[int], device: str = "cpu") -> dict:
    print("\nEvaluating DistilBERT Model...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_probas: list[float] = []
    start = time.time()

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            proba = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
            pred = int(proba.argmax())

            all_preds.append(pred)
            all_probas.append(float(proba[1]))

    elapsed = time.time() - start
    latency_per_sample = (elapsed / len(texts)) * 1000 if texts else 0.0

    results = compute_metrics(labels, all_preds, all_probas)
    results.update(
        {
            "model": "distilbert",
            "latency_ms": round(latency_per_sample, 2),
            "total_time_s": round(elapsed, 2),
            "n_samples": len(texts),
        }
    )
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(description="Compare structured, TF-IDF, DistilBERT, and rule baselines")
    ap.add_argument("--data", default="data/regenerated/windows_dataset_sample.jsonl", help="Test data file (JSONL)")
    ap.add_argument("--tfidf-model", default="models/baseline.pkl", help="Baseline model path")
    ap.add_argument("--distilbert-model", default="models/distilbert_finetuned", help="DistilBERT model directory")
    ap.add_argument("--output", default="models/model_comparison.json", help="Output JSON file")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    ap.add_argument("--split-mode", choices=["time", "stratified", "none"], default="time", help="How to split the dataset before comparison")
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

    print("Loading test data...")
    records = load_jsonl_records(args.data)
    records = clean_fractured_threat_records(
        records,
        policy=args.fractured_threat_policy,
        min_event_count=args.fractured_threat_min_events,
    )
    if len(records) == 0:
        print("ERROR: No records loaded for comparison")
        return 2
    feature_schemas = _collect_feature_schemas(records)
    if len(feature_schemas) != 1:
        print(
            "ERROR: mixed feature schemas detected in the same comparison run: "
            + ", ".join(sorted(feature_schemas))
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

    print(f"Loaded {len(records)} samples for comparison")
    print(f"Train: {len(train_records)}, Test: {len(test_records)} (split={actual_split_mode}; requested={args.split_mode})")
    print(f"Feature schema: {active_feature_schema}")

    texts = [record_to_window_text(record)[1] for record in test_records]
    windows = [record_to_window_text(record)[0] for record in test_records]
    labels = [resolve_record_label(record) for record in test_records]

    structured_results = evaluate_structured(args.tfidf_model, windows, labels)
    tfidf_results = evaluate_tfidf(args.tfidf_model, texts, labels)
    distilbert_results = evaluate_distilbert(args.distilbert_model, texts, labels, device=args.device)

    rule_results = {
        "model": "rule_baseline",
        "disabled": True,
        "reason": "rule_baseline_removed",
        "latency_ms": 0.0,
        "total_time_s": 0.0,
        "n_samples": len(texts),
    }

    rule_f1_delta = None
    if not rule_results.get("disabled"):
        rule_f1_delta = round(structured_results["positive_f1"] - rule_results["positive_f1"], 4)

    comparison = {
        "feature_schema": active_feature_schema,
        "structured_baseline": structured_results,
        "tfidf": tfidf_results,
        "distilbert": distilbert_results,
        "rule_baseline": rule_results,
        "summary": {
            "accuracy_delta": round(structured_results["accuracy"] - tfidf_results["accuracy"], 4),
            "f1_delta": round(structured_results["positive_f1"] - tfidf_results["positive_f1"], 4),
            "rule_f1_delta": rule_f1_delta,
            "latency_ratio": round(distilbert_results["latency_ms"] / max(structured_results["latency_ms"], 0.001), 1),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"Accuracy Delta:      Structured {comparison['summary']['accuracy_delta']:+.4f} vs TF-IDF")
    print(f"Positive F1 Delta:   Structured {comparison['summary']['f1_delta']:+.4f} vs TF-IDF")
    if comparison["summary"]["rule_f1_delta"] is None:
        print("Rule F1 Delta:       N/A (rule baseline disabled for this feature schema)")
    else:
        print(f"Rule F1 Delta:       Structured {comparison['summary']['rule_f1_delta']:+.4f} vs Rule")
    print(f"Latency Ratio:       DistilBERT is {comparison['summary']['latency_ratio']:.1f}x slower than structured")
    print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()
