#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

from alembic import command
from alembic.config import Config

from services.database.models import get_db_url, get_session
from services.database.db_service import ModelService


def redact_db_url(db_url: str) -> str:
    return re.sub(r":([^:@/]+)@", ":****@", db_url, count=1)


def run_migrations(db_url: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(repo_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(repo_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(alembic_cfg, "head")


def seed_default_models(db_url: str) -> None:
    session = get_session(db_url)
    try:
        model_service = ModelService(session)
        existing_models = model_service.get_models()
        if existing_models:
            print(f"INFO: Models already exist ({len(existing_models)} found)")
            active_models = [m for m in existing_models if m.active]
            if not active_models:
                distilbert = next((m for m in existing_models if m.model_type == "distilbert"), None)
                if distilbert:
                    model_service.set_active_model(distilbert.id)
                    print("INFO: Activated DistilBERT model")
                else:
                    tfidf = next((m for m in existing_models if m.model_type == "tfidf_fallback"), None)
                    if tfidf:
                        model_service.set_active_model(tfidf.id)
                        print("INFO: Activated TF-IDF model")
            return

        tfidf_model = model_service.register_model(
            model_type="tfidf_fallback",
            version="1.0",
            accuracy=0.774,
            f1_score=0.6921,
            roc_auc=0.8903,
            n_samples=500,
            metadata=json.dumps({"vectorizer": "tfidf", "classifier": "logistic_regression"}),
        )
        print(f"INFO: Registered TF-IDF model (ID: {tfidf_model.id})")

        distilbert_model = model_service.register_model(
            model_type="distilbert",
            version="1.0",
            accuracy=0.954,
            f1_score=0.9204,
            roc_auc=0.9981,
            n_samples=500,
            metadata=json.dumps({"model": "distilbert-base-uncased", "fine_tuned": True}),
        )
        print(f"INFO: Registered DistilBERT model (ID: {distilbert_model.id})")

        model_service.set_active_model(distilbert_model.id)
        print("INFO: Set DistilBERT as active model")
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize threat detector database")
    parser.add_argument("--db-url", default=None, help="Database URL")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--db-name", default="threat_detector")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="postgres")
    args = parser.parse_args()

    if args.db_url:
        db_url = args.db_url
    else:
        db_url = get_db_url(
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
        )

    print(f"Initializing database at {redact_db_url(db_url)}...")

    try:
        run_migrations(db_url)
        print("INFO: Database migrations applied")
        seed_default_models(db_url)
        print("INFO: Database initialization complete")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
