#!/usr/bin/env python
"""
Register a trained model manifest into the threat-detector database.

Reads the manifest JSON produced by train_baseline.py (stored alongside
the .pkl file) and upserts a row into the models table so that the API
and dashboard reflect the currently active model.

Usage:
    python scripts/register_model.py --manifest models/baseline_unsw.manifest.json
    python scripts/register_model.py --manifest models/baseline_unsw.manifest.json --activate

Environment:
    DATABASE_URL  PostgreSQL connection string (default: localhost dev URL)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.database.models import Base, Model, get_db_url, get_session


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _model_type_from_manifest(manifest: dict) -> str:
    raw = manifest.get("model_type", "structured_baseline")
    # normalise legacy names
    if raw in {"tfidf", "tfidf_text"}:
        return "tfidf_fallback"
    if raw in {"council", "ensemble", "structured"}:
        return "structured_baseline"
    return raw


def _version_from_manifest(manifest: dict) -> str:
    parts: list[str] = []
    fv = manifest.get("feature_version")
    if fv is not None:
        parts.append(f"v{fv}")
    schema = manifest.get("feature_text") or manifest.get("feature_schema")
    if schema:
        parts.append(schema)
    return "_".join(parts) if parts else "unknown"


def register(manifest_path: Path, db_url: str, activate: bool) -> None:
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = _load_manifest(manifest_path)
    model_type = _model_type_from_manifest(manifest)
    version = _version_from_manifest(manifest)

    metrics = manifest.get("metrics", {})
    holdout_key = "holdout"

    def _metric(tier: str, key: str) -> float | None:
        block = metrics.get(tier, {})
        holdout = block.get(holdout_key, {})
        if isinstance(holdout, dict) and key in holdout:
            val = holdout[key]
        elif key in block:
            val = block[key]
        else:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # Prefer structured metrics; fall back to text/tfidf
    f1 = _metric("structured_baseline", "positive_f1") or _metric("tfidf_fallback", "positive_f1")
    roc_auc = _metric("structured_baseline", "roc_auc") or _metric("tfidf_fallback", "roc_auc")
    accuracy = _metric("structured_baseline", "accuracy") or _metric("tfidf_fallback", "accuracy")
    n_samples = manifest.get("n_samples")

    gate = manifest.get("quality_gate", {})
    promotion_ready = gate.get("promotion_ready", False)
    if not promotion_ready:
        print(
            f"WARNING: model did not pass promotion gate "
            f"(promotion_ready={promotion_ready}). Registering anyway.",
            file=sys.stderr,
        )

    session = get_session(db_url)
    try:
        # Deactivate all existing models of this type when activating
        if activate:
            existing = session.query(Model).filter(Model.model_type == model_type).all()
            for m in existing:
                m.active = False
            session.flush()

        model_record = Model(
            model_type=model_type,
            version=version,
            accuracy=accuracy,
            f1_score=f1,
            roc_auc=roc_auc,
            n_samples=n_samples,
            active=activate,
            model_metadata=json.dumps({
                "manifest_path": str(manifest_path),
                "promotion_ready": promotion_ready,
                "feature_schema": manifest.get("feature_text") or manifest.get("feature_schema"),
                "data_paths": manifest.get("data_paths", []),
                "holdout_source": gate.get("holdout_source"),
            }),
        )
        session.add(model_record)
        session.commit()

        status = "ACTIVE" if activate else "registered (inactive)"
        print(
            f"Model registered [{status}]: type={model_type} version={version} "
            f"f1={f1:.4f if f1 else 'N/A'} roc_auc={roc_auc:.4f if roc_auc else 'N/A'} "
            f"n_samples={n_samples} id={model_record.id}"
        )
    except Exception as exc:
        session.rollback()
        print(f"ERROR: Failed to register model: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()


def main() -> None:
    db_url_default = os.getenv("DATABASE_URL", get_db_url())

    ap = argparse.ArgumentParser(description="Register a trained model manifest into the DB")
    ap.add_argument("--manifest", required=True, help="Path to .manifest.json file")
    ap.add_argument(
        "--activate",
        action="store_true",
        default=False,
        help="Mark this model as active (deactivates other models of the same type)",
    )
    ap.add_argument("--db-url", default=db_url_default, help="Database URL")
    args = ap.parse_args()

    register(Path(args.manifest), args.db_url, args.activate)


if __name__ == "__main__":
    main()
