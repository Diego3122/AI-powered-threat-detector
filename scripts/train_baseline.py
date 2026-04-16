#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from services.ml.ml_utils import (
    NETWORK_FLOW_FEATURE_SCHEMA,
    clean_fractured_threat_records,
    compute_metrics,
    extract_window,
    find_best_threshold,
    load_jsonl_records,
    record_to_window_text,
    resolve_record_dataset_source,
    resolve_record_sample_weight,
    resolve_record_label,
    resolve_window_feature_schema,
    resolve_window_threat_family,
    split_records,
    window_to_feature_dict,
)

WEAK_LABEL_SOURCES = {"legacy_rule", "unknown", "external_heuristic"}
MIN_HOLDOUT_NEGATIVE_RECALL = 0.40


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


def _fit_text_model(
    texts: list[str],
    labels: list[int],
    seed: int,
    sample_weight: list[float] | None = None,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = _make_text_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    model = _make_linear_model(seed)
    model.fit(matrix, labels, sample_weight=sample_weight)
    return vectorizer, model


def _fit_structured_model(
    features: list[dict[str, float]],
    labels: list[int],
    seed: int,
    sample_weight: list[float] | None = None,
) -> tuple[DictVectorizer, LogisticRegression]:
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(features)
    model = _make_linear_model(seed)
    model.fit(matrix, labels, sample_weight=sample_weight)
    return vectorizer, model




def _score_structured_records(
    records: list[dict[str, object]],
    *,
    fallback_model: LogisticRegression,
    fallback_vectorizer: DictVectorizer,
    fallback_threshold: float,
    family_models: dict[str, LogisticRegression],
    family_vectorizers: dict[str, DictVectorizer],
    family_thresholds: dict[str, float],
) -> tuple[list[float], list[int], list[str]]:
    scores: list[float] = []
    predictions: list[int] = []
    families: list[str] = []

    for record in records:
        window = extract_window(record)
        family = resolve_window_threat_family(window)
        model = family_models.get(family, fallback_model)
        vectorizer = family_vectorizers.get(family, fallback_vectorizer)
        threshold = float(family_thresholds.get(family, fallback_threshold))
        feature_row = window_to_feature_dict(window)
        score = float(model.predict_proba(vectorizer.transform([feature_row]))[0][1])
        scores.append(score)
        predictions.append(int(score >= threshold))
        families.append(family)

    return scores, predictions, families


def _collect_label_sources(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        source = str(record.get("label_source", "unknown")).strip() or "unknown"
        counts[source] += 1
    return dict(sorted(counts.items()))


def _normalize_data_paths(values: list[str] | None) -> list[str]:
    if not values:
        return ["data/regenerated/windows_dataset_sample.jsonl"]
    return values


def _collect_dataset_source_counts(records: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[resolve_record_dataset_source(record)] += 1
    return dict(sorted(counts.items()))


def _collect_label_sources_by_dataset(records: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = {}
    for record in records:
        dataset_source = resolve_record_dataset_source(record)
        label_source = str(record.get("label_source", "unknown")).strip() or "unknown"
        grouped.setdefault(dataset_source, Counter())[label_source] += 1
    return {
        dataset_source: dict(sorted(counter.items()))
        for dataset_source, counter in sorted(grouped.items())
    }


def _collect_feature_schemas(records: list[dict]) -> set[str]:
    schemas: set[str] = set()
    for record in records:
        window = extract_window(record)
        schemas.add(resolve_window_feature_schema(window))
    return schemas


def _build_quality_gate(
    *,
    holdout_source: str | None,
    dataset_source_counts: dict[str, int],
    label_sources_by_dataset: dict[str, dict[str, int]],
    structured_metrics: dict[str, float],
    min_holdout_positive_f1: float,
    min_holdout_negative_recall: float,
) -> dict[str, object]:
    blocker_codes: list[str] = []
    blockers: list[str] = []

    holdout_label_sources = label_sources_by_dataset.get(holdout_source or "", {})
    holdout_positive_f1 = float(structured_metrics.get("positive_f1", 0.0))
    holdout_negative_recall = float(structured_metrics.get("negative_recall", 0.0))

    if len(dataset_source_counts) < 2:
        blocker_codes.append("single_source_dataset")
        blockers.append("Training corpus includes only one dataset source.")

    if not holdout_source:
        blocker_codes.append("no_holdout_source")
        blockers.append("No held-out dataset source was reserved for cross-source evaluation.")
    else:
        if holdout_positive_f1 < min_holdout_positive_f1:
            blocker_codes.append("holdout_structured_positive_f1_below_threshold")
            blockers.append(
                f"Held-out structured positive F1 {holdout_positive_f1:.4f} is below the promotion threshold "
                f"{min_holdout_positive_f1:.4f}."
            )
        if holdout_negative_recall < min_holdout_negative_recall:
            blocker_codes.append("holdout_negative_recall_below_threshold")
            blockers.append(
                f"Held-out negative recall {holdout_negative_recall:.4f} is below the promotion threshold "
                f"{min_holdout_negative_recall:.4f}."
            )

        if not holdout_label_sources:
            blocker_codes.append("missing_holdout_label_provenance")
            blockers.append("Held-out source is missing label provenance details.")
        elif set(holdout_label_sources) <= WEAK_LABEL_SOURCES:
            blocker_codes.append("weak_holdout_labels")
            blockers.append("Held-out source uses only weak or heuristic labels.")

    return {
        "promotion_ready": not blocker_codes,
        "blocker_codes": blocker_codes,
        "blockers": blockers,
        "criteria": {
            "requires_holdout_source": True,
            "min_holdout_positive_f1": float(min_holdout_positive_f1),
            "min_holdout_negative_recall": float(min_holdout_negative_recall),
            "strong_holdout_labels_required": True,
        },
        "observed": {
            "dataset_source_count": len(dataset_source_counts),
            "holdout_source": holdout_source,
            "holdout_label_sources": holdout_label_sources,
            "holdout_structured_positive_f1": holdout_positive_f1,
            "holdout_negative_recall": holdout_negative_recall,
        },
    }


def _split_records_for_training(
    records: list[dict[str, object]],
    *,
    test_size: float,
    split_mode: str,
    seed: int,
    holdout_source: str | None,
) -> tuple[list[dict], list[dict]]:
    if not holdout_source:
        return split_records(records, test_size=test_size, split_mode=split_mode, seed=seed)

    normalized_holdout = holdout_source.strip()
    train_records = [record for record in records if resolve_record_dataset_source(record) != normalized_holdout]
    test_records = [record for record in records if resolve_record_dataset_source(record) == normalized_holdout]
    if not train_records:
        raise ValueError(f"--holdout-source={normalized_holdout} left no training records")
    if not test_records:
        known_sources = ", ".join(sorted(_collect_dataset_source_counts(records)))
        raise ValueError(
            f"--holdout-source={normalized_holdout} did not match any loaded records. "
            f"Known sources: {known_sources or 'none'}"
        )

    train_labels = {resolve_record_label(record) for record in train_records}
    test_labels = {resolve_record_label(record) for record in test_records}
    if len(train_labels) < 2:
        raise ValueError(f"Training split for holdout source {normalized_holdout} does not contain both classes")
    if len(test_labels) < 2:
        raise ValueError(f"Holdout source {normalized_holdout} does not contain both classes")
    return train_records, test_records


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(description="Train a direct structured threat detector baseline")
    ap.add_argument(
        "--data",
        action="append",
        default=None,
        help="Input dataset JSONL file. Repeat --data to train on multiple sources.",
    )
    ap.add_argument("--out", default="models/baseline.pkl", help="Output model path")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set")
    ap.add_argument("--split-mode", choices=["time", "stratified"], default="time", help="How to split the dataset before evaluation")
    ap.add_argument("--max-samples", type=int, default=0, help="Max samples to load (0 = all)")
    ap.add_argument("--holdout-source", default=None, help="Optional dataset_source to reserve fully for evaluation")
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
        "--promotion-min-holdout-positive-f1",
        type=float,
        default=0.6,
        help="Minimum held-out structured positive F1 required before marking the artifact promotion_ready",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    holdout_source = args.holdout_source.strip() if args.holdout_source and args.holdout_source.strip() else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_paths = _normalize_data_paths(args.data)
    records = load_jsonl_records(data_paths, max_samples=args.max_samples)
    records = clean_fractured_threat_records(
        records,
        policy=args.fractured_threat_policy,
        min_event_count=args.fractured_threat_min_events,
    )
    if len(records) == 0:
        print("ERROR: No data loaded", file=sys.stderr)
        sys.exit(1)

    labels = [resolve_record_label(record) for record in records]
    label_0_count = sum(1 for label in labels if label == 0)
    label_1_count = sum(1 for label in labels if label == 1)
    feature_schemas = _collect_feature_schemas(records)
    if len(feature_schemas) != 1:
        print(
            "ERROR: mixed feature schemas detected in the same training run: "
            + ", ".join(sorted(feature_schemas)),
            file=sys.stderr,
        )
        return 2
    active_feature_schema = next(iter(feature_schemas))

    print(f"Loaded {len(records)} samples")
    print(f"Class distribution: {label_0_count} label=0, {label_1_count} label=1")
    print(f"Feature schema: {active_feature_schema}")
    if label_0_count == 0 or label_1_count == 0:
        print("ERROR: Dataset has only one class. Use a balanced dataset or generated metadata labels.", file=sys.stderr)
        sys.exit(1)

    label_source_counts = _collect_label_sources(records)
    dataset_source_counts = _collect_dataset_source_counts(records)
    label_sources_by_dataset = _collect_label_sources_by_dataset(records)
    if len(dataset_source_counts) > 1 and not holdout_source:
        known_sources = ", ".join(sorted(dataset_source_counts))
        print(
            "ERROR: --holdout-source is required when more than one dataset source is loaded. "
            f"Loaded sources: {known_sources}",
            file=sys.stderr,
        )
        return 2
    if set(label_source_counts) <= {"legacy_rule", "unknown"}:
        print(
            "WARNING: dataset labels are mostly heuristic or unknown. "
            "Metrics may overstate real-world quality.",
            file=sys.stderr,
        )
    if any(set(source_counts) <= {"legacy_rule", "unknown"} for source_counts in label_sources_by_dataset.values()):
        weak_sources = [
            dataset_source
            for dataset_source, source_counts in label_sources_by_dataset.items()
            if set(source_counts) <= {"legacy_rule", "unknown"}
        ]
        print(
            "WARNING: some dataset sources are weakly labeled: "
            + ", ".join(sorted(weak_sources)),
            file=sys.stderr,
        )

    train_records, test_records = _split_records_for_training(
        records,
        test_size=args.test_size,
        split_mode=args.split_mode,
        seed=args.seed,
        holdout_source=holdout_source,
    )
    train_texts = [record_to_window_text(record)[1] for record in train_records]
    train_windows = [record_to_window_text(record)[0] for record in train_records]
    train_features = [window_to_feature_dict(window) for window in train_windows]
    train_labels = [resolve_record_label(record) for record in train_records]
    train_sample_weights = [resolve_record_sample_weight(record) for record in train_records]
    test_texts = [record_to_window_text(record)[1] for record in test_records]
    test_windows = [record_to_window_text(record)[0] for record in test_records]
    test_labels = [resolve_record_label(record) for record in test_records]

    split_descriptor = f"holdout:{holdout_source}" if holdout_source else args.split_mode
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)} (split={split_descriptor})")
    print(f"Dataset sources: {dataset_source_counts}")

    print("Training text baseline...")
    text_vectorizer, text_model = _fit_text_model(
        train_texts,
        train_labels,
        args.seed,
        sample_weight=train_sample_weights,
    )
    text_test_scores = text_model.predict_proba(text_vectorizer.transform(test_texts))[:, 1].tolist()
    text_threshold = 0.5

    print("Training structured baseline...")
    structured_vectorizer, structured_model = _fit_structured_model(
        train_features,
        train_labels,
        args.seed,
        sample_weight=train_sample_weights,
    )
    structured_train_scores = structured_model.predict_proba(structured_vectorizer.transform(train_features))[:, 1].tolist()
    # macro_f1 balances detection recall against false-positive rate; positive_f1
    # maximises recall at the cost of ~49 % FP rate on the UNSW benign class.
    structured_threshold, _ = find_best_threshold(structured_train_scores, train_labels, metric="macro_f1")
    structured_family_models: dict = {}
    structured_family_vectorizers: dict = {}
    structured_family_thresholds: dict = {}
    structured_family_train_counts: dict = {}
    structured_test_scores, structured_test_preds, _ = _score_structured_records(
        test_records,
        fallback_model=structured_model,
        fallback_vectorizer=structured_vectorizer,
        fallback_threshold=structured_threshold,
        family_models=structured_family_models,
        family_vectorizers=structured_family_vectorizers,
        family_thresholds=structured_family_thresholds,
    )

    rule_threshold = 0.5
    rule_metrics: dict = {
        "disabled": True,
        "reason": "rule_baseline_removed",
        "feature_schema": active_feature_schema,
    }

    text_metrics = compute_metrics(test_labels, [1 if score >= text_threshold else 0 for score in text_test_scores], text_test_scores)
    structured_metrics = compute_metrics(
        test_labels,
        structured_test_preds,
        structured_test_scores,
    )
    structured_metrics["threshold"] = float(structured_threshold)

    print(f"Text baseline positive F1:       {text_metrics['positive_f1']:.4f}")
    print(f"Structured baseline positive F1: {structured_metrics['positive_f1']:.4f}")

    feature_version = 6 if active_feature_schema != NETWORK_FLOW_FEATURE_SCHEMA else 7
    feature_text = (
        "window_summary_v6_sparse_event_semantics"
        if active_feature_schema != NETWORK_FLOW_FEATURE_SCHEMA
        else "network_flow_v1_unsw_ground_truth"
    )

    model_dict = {
        "model": text_model,
        "vectorizer": text_vectorizer,
        "structured_model": structured_model,
        "structured_vectorizer": structured_vectorizer,
        "structured_family_models": structured_family_models,
        "structured_family_vectorizers": structured_family_vectorizers,
        "structured_family_thresholds": structured_family_thresholds,
        "model_type": "structured_baseline",
        "default_model_type": "structured_baseline",
        "feature_schema": active_feature_schema,
        "feature_version": feature_version,
        "feature_text": feature_text,
        "split_mode": args.split_mode,
        "fractured_threat_policy": args.fractured_threat_policy,
        "fractured_threat_min_events": int(args.fractured_threat_min_events),
        "data_paths": data_paths,
        "dataset_source_counts": dataset_source_counts,
        "train_source_counts": _collect_dataset_source_counts(train_records),
        "test_source_counts": _collect_dataset_source_counts(test_records),
        "holdout_source": holdout_source,
        "rule_threshold": float(rule_threshold),
        "structured_threshold": float(structured_threshold),
        "label_source_counts": label_source_counts,
        "label_source_counts_by_dataset": label_sources_by_dataset,
        "structured_family_train_counts": structured_family_train_counts,
    }
    quality_gate = _build_quality_gate(
        holdout_source=holdout_source,
        dataset_source_counts=dataset_source_counts,
        label_sources_by_dataset=label_sources_by_dataset,
        structured_metrics=structured_metrics,
        min_holdout_positive_f1=args.promotion_min_holdout_positive_f1,
        min_holdout_negative_recall=MIN_HOLDOUT_NEGATIVE_RECALL,
    )
    model_dict["quality_gate"] = quality_gate

    manifest = {
        "model_type": "structured_baseline",
        "data_paths": data_paths,
        "n_samples": len(records),
        "split_mode": args.split_mode,
        "evaluation_split": split_descriptor,
        "feature_schema": active_feature_schema,
        "feature_version": model_dict["feature_version"],
        "feature_text": model_dict["feature_text"],
        "fractured_threat_policy": args.fractured_threat_policy,
        "fractured_threat_min_events": int(args.fractured_threat_min_events),
        "structured_threshold": model_dict["structured_threshold"],
        "structured_family_thresholds": structured_family_thresholds,
        "structured_family_train_counts": structured_family_train_counts,
        "rule_threshold": model_dict["rule_threshold"],
        "label_source_counts": label_source_counts,
        "label_source_counts_by_dataset": label_sources_by_dataset,
        "dataset_source_counts": dataset_source_counts,
        "train_source_counts": model_dict["train_source_counts"],
        "test_source_counts": model_dict["test_source_counts"],
        "holdout_source": holdout_source,
        "quality_gate": quality_gate,
        "metrics": {
            "structured_baseline": structured_metrics,
            "text_tfidf": text_metrics,
            "rule_baseline": rule_metrics,
        },
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    runtime_bundle_path = Path("models/baseline.pkl").resolve()
    if out_path.resolve() == runtime_bundle_path and not quality_gate["promotion_ready"]:
        print(
            "ERROR: refusing to overwrite models/baseline.pkl because the promotion gate is blocked.",
            file=sys.stderr,
        )
        for blocker in quality_gate["blockers"]:
            print(f"  - {blocker}", file=sys.stderr)
        return 2

    joblib.dump(model_dict, args.out)
    print(f"Model saved to {args.out}")

    if quality_gate["promotion_ready"]:
        print("Promotion gate: PASS")
    else:
        print("Promotion gate: BLOCKED")
        for blocker in quality_gate["blockers"]:
            print(f"  - {blocker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
