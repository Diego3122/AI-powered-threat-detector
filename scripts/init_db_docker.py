#!/usr/bin/env python
import os
import re
import time
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from services.database.models import get_session
from services.database.db_service import ModelService


def redact_db_url(db_url: str) -> str:
    return re.sub(r":([^:@/]+)@", ":****@", db_url, count=1)


def wait_for_db(db_url: str, max_retries: int = 30) -> bool:
    for i in range(max_retries):
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("INFO: Database is ready")
            return True
        except OperationalError:
            if i < max_retries - 1:
                print(f"INFO: Waiting for database... ({i + 1}/{max_retries})")
                time.sleep(2)
            else:
                print("ERROR: Database not available after max retries")
                return False
    return False


def run_migrations(db_url: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    alembic_cfg = Config(str(repo_root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(repo_root / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(alembic_cfg, "head")


def seed_default_models(db_url: str) -> None:
    """Register the UNSW structured baseline from its manifest, if not already active."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "models" / "baseline_unsw.manifest.json"

    if not manifest_path.exists():
        print(f"WARNING: Manifest not found at {manifest_path}, skipping model registration")
        return

    session = get_session(db_url)
    try:
        model_service = ModelService(session)
        existing = model_service.get_models()
        structured = [m for m in existing if m.model_type == "structured_baseline"]
        if structured and any(m.active for m in structured):
            print(f"INFO: Active structured_baseline already registered (id={structured[0].id}), skipping")
            return
    finally:
        session.close()

    from scripts.register_model import register
    register(manifest_path, db_url, activate=True)
    print("INFO: Registered structured_baseline from UNSW manifest as active model")


def main() -> int:
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@postgres:5432/threat_detector",
    )

    print(f"INFO: Initializing database at {redact_db_url(db_url)}")

    if not wait_for_db(db_url):
        return 1

    try:
        run_migrations(db_url)
        print("INFO: Database migrations applied")
        seed_default_models(db_url)
        print("INFO: Database initialization complete")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
