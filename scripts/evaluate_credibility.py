#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from services.ml.ml_utils import (
    NETWORK_FLOW_FEATURE_SCHEMA,
    clean_fractured_threat_records,
    compute_metrics,
    extract_window,
    find_best_threshold,
    load_jsonl_records,
    record_to_window_text,
    resolve_record_label,
    resolve_record_sample_weight,
    resolve_window_feature_schema,
    resolve_window_threat_family,
    score_structured_inputs,
    window_to_feature_dict,
)


def _collect_feature_schemas(records: list[dict[str, Any]]) -> set[str]:
    schemas: set[str] = set()
    for record in records:
        schemas.add(resolve_window_feature_schema(extract_window(record)))
    return schemas


def _make_text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def _make_linear_model(seed: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        random_state=seed,
        class_weight="balanced",
        solver="liblinear",
        C=1.5,
    )


def _fit_text_model(texts: list[str], labels: list[int], seed: int) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = _make_text_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    model = _make_linear_model(seed)
    model.fit(matrix, labels)
    return vectorizer, model


def _fit_structured_model(
    features: list[dict[str, float]],
    labels: list[int],
    seed: int,
) -> tuple[DictVectorizer, LogisticRegression]:
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(features)
    model = _make_linear_model(seed)
    model.fit(matrix, labels)
    return vectorizer, model




def _default_group_by(records: list[dict[str, Any]]) -> str:
    prioritized_keys = (
        "campaign_signature",
        "scenario_signature",
        "window_group_id",
        "actor_signature",
        "dominant_actor_id",
    )
    for key in prioritized_keys:
        for record in records:
            window = extract_window(record)
            metadata = window.get("metadata") if isinstance(window, dict) else {}
            if not isinstance(metadata, dict):
                continue
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return key
    return "none"


def _summarize_metric(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}

    avg = mean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * (std / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "mean": round(avg, 4),
        "std": round(std, 4),
        "ci95_low": round(avg - margin, 4),
        "ci95_high": round(avg + margin, 4),
    }


def _summarize_results(metrics_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_names = ("positive_precision", "positive_recall", "positive_f1", "balanced_accuracy", "roc_auc", "average_precision")
    for model_name, fold_metrics in metrics_by_model.items():
        model_summary: dict[str, Any] = {
            "folds": len(fold_metrics),
        }
        for metric_name in metric_names:
            values = [float(metrics[metric_name]) for metrics in fold_metrics if metrics.get(metric_name) is not None]
            model_summary[metric_name] = _summarize_metric(values)
        summary[model_name] = model_summary
    return summary


def _label_source_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        source = str(record.get("label_source", "unknown")).strip() or "unknown"
        counts[source] += 1
    return dict(sorted(counts.items()))


def _mean_metric(summary: dict[str, Any], model_name: str, metric_name: str) -> float | None:
    try:
        value = summary[model_name][metric_name]["mean"]
    except Exception:
        return None
    if value is None:
        return None
    return float(value)


def _extract_group(record: dict[str, Any], group_by: str) -> str:
    if group_by == "none":
        return ""

    window = record.get("window") if isinstance(record.get("window"), dict) else record
    metadata = window.get("metadata") if isinstance(window, dict) and isinstance(window.get("metadata"), dict) else {}
    if group_by == "primary_scenario":
        return str(metadata.get("primary_scenario", "unknown"))
    if group_by == "scenario_signature":
        signature = metadata.get("scenario_signature")
        if signature:
            return str(signature)
        component_scenarios = metadata.get("component_scenarios")
        if isinstance(component_scenarios, list) and component_scenarios:
            return "|".join(sorted(str(value) for value in component_scenarios))
        return "unknown"
    if group_by == "window_group_id":
        return str(metadata.get("window_group_id", "unknown"))
    if group_by == "campaign_signature":
        return str(metadata.get("campaign_signature", "unknown"))
    if group_by == "dominant_actor_id":
        return str(metadata.get("dominant_actor_id", "unknown"))
    if group_by == "actor_signature":
        return str(metadata.get("actor_signature", "unknown"))
    raise ValueError(f"Unsupported group_by value: {group_by}")


def _iter_splits(
    texts: list[str],
    labels: list[int],
    records: list[dict[str, Any]],
    *,
    folds: int,
    repeats: int,
    seed: int,
    group_by: str,
):
    if group_by == "none":
        splitter = RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=seed,
        )
        yield from splitter.split(texts, labels)
        return

    groups = [_extract_group(record, group_by) for record in records]
    unique_groups = len(set(groups))
    if unique_groups < folds:
        raise ValueError(f"group_by={group_by} produced only {unique_groups} unique groups; need at least {folds}")

    for repeat_index in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + repeat_index,
        )
        yield from splitter.split(texts, labels, groups=groups)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(
        description=(
            "Run repeated cross-validated evaluation for the current baselines and "
            "report a leakage audit instead of relying on a single split."
        )
    )
    ap.add_argument("--data", default="data/regenerated/windows_dataset_sample.jsonl", help="Input dataset JSONL file")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--repeats", type=int, default=3, help="Number of CV repeats")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out", default="models/credibility_report.json", help="Output JSON file")
    ap.add_argument("--max-samples", type=int, default=0, help="Limit records loaded from disk")
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
    ap.add_argument(
        "--group-by",
        choices=[
            "auto",
            "none",
            "primary_scenario",
            "scenario_signature",
            "window_group_id",
            "campaign_signature",
            "dominant_actor_id",
            "actor_signature",
        ],
        default="auto",
        help="Optionally keep related synthetic windows together during CV",
    )
    args = ap.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    records = load_jsonl_records(args.data, max_samples=args.max_samples)
    records = clean_fractured_threat_records(
        records,
        policy=args.fractured_threat_policy,
        min_event_count=args.fractured_threat_min_events,
    )
    if not records:
        raise ValueError(f"No records found in {args.data}")

    labels = [resolve_record_label(record) for record in records]
    if len(set(labels)) < 2:
        raise ValueError("Credibility evaluation requires at least two classes")
    feature_schemas = _collect_feature_schemas(records)
    if len(feature_schemas) != 1:
        raise ValueError(
            "Mixed feature schemas detected in the same credibility run: "
            + ", ".join(sorted(feature_schemas))
        )
    active_feature_schema = next(iter(feature_schemas))

    texts = [record_to_window_text(record)[1] for record in records]
    windows = [record_to_window_text(record)[0] for record in records]
    structured_features = [window_to_feature_dict(window) for window in windows]
    metrics_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    effective_group_by = _default_group_by(records) if args.group_by == "auto" else args.group_by

    for fold_index, (train_idx, test_idx) in enumerate(
        _iter_splits(
            texts,
            labels,
            records,
            folds=args.folds,
            repeats=args.repeats,
            seed=args.seed,
            group_by=effective_group_by,
        ),
        start=1,
    ):
        train_texts = [texts[index] for index in train_idx]
        test_texts = [texts[index] for index in test_idx]
        train_features = [structured_features[index] for index in train_idx]
        train_labels = [labels[index] for index in train_idx]
        test_labels = [labels[index] for index in test_idx]
        train_records = [records[index] for index in train_idx]
        test_records = [records[index] for index in test_idx]
        test_windows = [windows[index] for index in test_idx]

        text_vectorizer, text_model = _fit_text_model(train_texts, train_labels, args.seed + fold_index)
        text_scores = text_model.predict_proba(text_vectorizer.transform(test_texts))[:, 1].tolist()
        metrics_by_model["text_tfidf"].append(
            compute_metrics(test_labels, [1 if score >= 0.5 else 0 for score in text_scores], text_scores)
        )

        structured_vectorizer, structured_model = _fit_structured_model(train_features, train_labels, args.seed + fold_index)
        structured_train_scores = structured_model.predict_proba(structured_vectorizer.transform(train_features))[:, 1].tolist()
        structured_threshold, _ = find_best_threshold(structured_train_scores, train_labels)
        structured_bundle = {
            "model": structured_model,
            "vectorizer": structured_vectorizer,
            "structured_model": structured_model,
            "structured_vectorizer": structured_vectorizer,
            "structured_family_models": {},
            "structured_family_vectorizers": {},
            "structured_family_thresholds": {},
            "structured_family_train_counts": {},
            "structured_threshold": float(structured_threshold),
            "feature_schema": active_feature_schema,
            "feature_version": 7,
            "fractured_threat_policy": args.fractured_threat_policy,
            "fractured_threat_min_events": int(args.fractured_threat_min_events),
        }
        structured_results = score_structured_inputs(structured_bundle, test_windows)
        structured_test_scores = [float(result["score"]) for result in structured_results]
        structured_test_preds = [int(result["label"]) for result in structured_results]
        metrics_by_model["structured_baseline"].append(
            compute_metrics(
                test_labels,
                structured_test_preds,
                structured_test_scores,
            )
        )

        always_normal_scores = [0.0 for _ in test_labels]
        metrics_by_model["always_normal"].append(
            compute_metrics(test_labels, [0 for _ in test_labels], always_normal_scores)
        )

    report = {
        "data_path": args.data,
        "n_samples": len(records),
        "feature_schema": active_feature_schema,
        "label_counts": dict(sorted(Counter(labels).items())),
        "label_source_counts": _label_source_counts(records),
        "evaluation": {
            "scheme": "stratified_group_kfold" if effective_group_by != "none" else "repeated_stratified_kfold",
            "folds": args.folds,
            "repeats": args.repeats,
            "seed": args.seed,
            "group_by_requested": args.group_by,
            "group_by": effective_group_by,
        },
        "warnings": [],
        "summary": _summarize_results(metrics_by_model),
    }

    if report["label_source_counts"].keys() <= {"legacy_rule", "unknown"}:
        report["warnings"].append("Dataset labels are heuristic or unknown; headline metrics should be treated as weak-label results.")

    structured_f1_mean = _mean_metric(report["summary"], "structured_baseline", "positive_f1")
    if structured_f1_mean is not None and structured_f1_mean >= 0.99:
        report["warnings"].append(
            "Near-perfect structured-model performance suggests the dataset may still be too easy; validate on a harder or external holdout."
        )
    if args.group_by == "auto" and effective_group_by != "none":
        report["warnings"].append(f"Synthetic metadata detected; grouped CV resolved to {effective_group_by}.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote credibility report to {out_path}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for model_name, model_summary in report["summary"].items():
        positive_f1 = model_summary["positive_f1"]
        print(
            f"{model_name}: positive_f1 mean={positive_f1['mean']} "
            f"ci95=[{positive_f1['ci95_low']}, {positive_f1['ci95_high']}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
