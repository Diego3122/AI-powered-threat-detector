"""
Integration test: alert_router → API → DB → read-back.

Tests the full data path without Kafka or the model server:
  1. Detector builds an alert dict (replicated inline).
  2. Alert router POSTs it to /api/internal/alerts/ingest.
  3. API persists it in a real SQLite database.
  4. Authenticated user reads it back via /api/alerts.
  5. Analyst creates an investigation and updates its status.
"""

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from services.api import alerts_api
from services.api.auth import TokenData
from services.api.security import rate_limiter
from services.database.models import Base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session():
    """Fresh SQLite in-memory database with the full schema for each test."""
    # StaticPool ensures every engine.connect() returns the *same* underlying
    # connection, so the in-memory schema created by create_all() persists for
    # all ORM operations within the test.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def api_client(db_session):
    """TestClient wired to the in-memory database; resets rate-limiter buckets."""
    rate_limiter._buckets.clear()

    def _get_db_override():
        yield db_session

    alerts_api.app.dependency_overrides[alerts_api.get_db] = _get_db_override
    client = TestClient(alerts_api.app, raise_server_exceptions=True)
    yield client
    alerts_api.app.dependency_overrides.clear()


def _viewer_user():
    return TokenData(username="viewer", roles=["viewer"])


def _analyst_user():
    return TokenData(username="analyst", roles=["analyst"])


# ---------------------------------------------------------------------------
# Helper: build the same alert payload the alert_router sends
# ---------------------------------------------------------------------------

def _make_alert_payload(score: float = 0.87, threshold: float = 0.5) -> dict:
    window = {
        "feature_schema": "network_flow_v1",
        "window_start_ms": 1_700_000_000_000,
        "window_end_ms": 1_700_000_010_000,
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "sbytes": 14_000,
        "dbytes": 3_200,
        "srcip": "10.0.0.99",
        "dstip": "192.168.1.1",
    }
    return {
        "timestamp": int(time.time() * 1000),
        "window_id": f"{window['window_start_ms']}-{window['window_end_ms']}",
        "model_type": "structured_baseline",
        "model_score": score,
        "threshold": threshold,
        "model_label": 1,
        "feature_schema": "network_flow_v1",
        "explanation_summary": (
            f"structured_baseline flagged anomalous network flow "
            f"(proto=tcp, service=http, state=FIN, sbytes=14000, dbytes=3200, score={score:.2f})"
        ),
        "window": window,
    }


INTERNAL_KEY = "dev-internal-api-key"
INTERNAL_HEADERS = {"X-Internal-API-Key": INTERNAL_KEY}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInternalIngest:
    def test_ingest_creates_alert_in_db(self, api_client):
        payload = _make_alert_payload()
        response = api_client.post(
            "/api/internal/alerts/ingest",
            json=payload,
            headers=INTERNAL_HEADERS,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model_type"] == "structured_baseline"
        assert body["model_score"] == pytest.approx(0.87)
        assert body["triggered"] is True
        assert body["feature_schema"] == "network_flow_v1"
        assert "flagged anomalous" in body["explanation_summary"]

    def test_ingest_below_threshold_not_triggered(self, api_client):
        payload = _make_alert_payload(score=0.3, threshold=0.5)
        response = api_client.post(
            "/api/internal/alerts/ingest",
            json=payload,
            headers=INTERNAL_HEADERS,
        )
        assert response.status_code == 200
        assert response.json()["triggered"] is False

    def test_ingest_requires_internal_key(self, api_client):
        response = api_client.post(
            "/api/internal/alerts/ingest",
            json=_make_alert_payload(),
            headers={"X-Internal-API-Key": "wrong-key"},
        )
        assert response.status_code == 401


class TestAlertReadback:
    def test_list_alerts_returns_ingested_alert(self, api_client):
        # Ingest an alert
        payload = _make_alert_payload()
        ingest_resp = api_client.post(
            "/api/internal/alerts/ingest",
            json=payload,
            headers=INTERNAL_HEADERS,
        )
        assert ingest_resp.status_code == 200
        created_id = ingest_resp.json()["id"]

        # Read it back as an authenticated viewer
        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _viewer_user
        list_resp = api_client.get("/api/alerts")
        assert list_resp.status_code == 200
        alerts = list_resp.json()
        assert any(a["id"] == created_id for a in alerts)

    def test_get_alert_by_id_returns_all_fields(self, api_client):
        payload = _make_alert_payload()
        created = api_client.post(
            "/api/internal/alerts/ingest",
            json=payload,
            headers=INTERNAL_HEADERS,
        ).json()

        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _viewer_user
        resp = api_client.get(f"/api/alerts/{created['id']}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == created["id"]
        assert body["window_id"] == payload["window_id"]
        assert body["feature_schema"] == "network_flow_v1"
        assert body["explanation_summary"] is not None

    def test_triggered_filter_excludes_below_threshold(self, api_client):
        # Ingest one triggered and one not-triggered
        api_client.post(
            "/api/internal/alerts/ingest",
            json=_make_alert_payload(score=0.9, threshold=0.5),
            headers=INTERNAL_HEADERS,
        )
        api_client.post(
            "/api/internal/alerts/ingest",
            json=_make_alert_payload(score=0.2, threshold=0.5),
            headers=INTERNAL_HEADERS,
        )

        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _viewer_user
        resp = api_client.get("/api/alerts?triggered=true")
        assert resp.status_code == 200
        alerts = resp.json()
        assert all(a["triggered"] for a in alerts)
        assert len(alerts) >= 1


class TestInvestigationWorkflow:
    def _ingest_and_get_id(self, api_client) -> int:
        resp = api_client.post(
            "/api/internal/alerts/ingest",
            json=_make_alert_payload(),
            headers=INTERNAL_HEADERS,
        )
        assert resp.status_code == 200
        return resp.json()["id"]

    def test_create_investigation_open_status(self, api_client):
        alert_id = self._ingest_and_get_id(api_client)

        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _analyst_user
        resp = api_client.post(
            f"/api/alerts/{alert_id}/investigations",
            json={"status": "open", "notes": "Initial triage"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["alert_id"] == alert_id
        assert body["status"] == "open"
        assert body["notes"] == "Initial triage"
        assert body["user_id"] == "analyst"

    def test_investigation_lifecycle(self, api_client):
        alert_id = self._ingest_and_get_id(api_client)
        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _analyst_user

        # Create as open
        create_resp = api_client.post(
            f"/api/alerts/{alert_id}/investigations",
            json={"status": "open"},
        )
        assert create_resp.status_code == 200
        inv_id = create_resp.json()["id"]

        # Escalate to investigating
        update_resp = api_client.put(
            f"/api/investigations/{inv_id}",
            json={"status": "investigating", "notes": "Confirmed suspicious traffic pattern"},
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["status"] == "investigating"
        assert "suspicious" in update_resp.json()["notes"]

        # Resolve
        resolve_resp = api_client.put(
            f"/api/investigations/{inv_id}",
            json={"status": "resolved"},
        )
        assert resolve_resp.status_code == 200
        assert resolve_resp.json()["status"] == "resolved"

    def test_get_alert_investigations(self, api_client):
        alert_id = self._ingest_and_get_id(api_client)
        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _analyst_user

        api_client.post(
            f"/api/alerts/{alert_id}/investigations",
            json={"status": "investigating", "notes": "Looking into srcip 10.0.0.99"},
        )

        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _viewer_user
        resp = api_client.get(f"/api/alerts/{alert_id}/investigations")
        assert resp.status_code == 200
        investigations = resp.json()
        assert len(investigations) == 1
        assert investigations[0]["alert_id"] == alert_id

    def test_mark_false_positive(self, api_client):
        alert_id = self._ingest_and_get_id(api_client)
        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _analyst_user

        create_resp = api_client.post(
            f"/api/alerts/{alert_id}/investigations",
            json={"status": "open"},
        )
        inv_id = create_resp.json()["id"]

        fp_resp = api_client.put(
            f"/api/investigations/{inv_id}",
            json={"status": "false_positive", "notes": "Benign port scan from monitoring tool"},
        )
        assert fp_resp.status_code == 200
        assert fp_resp.json()["status"] == "false_positive"

    def test_investigation_on_missing_alert_returns_404(self, api_client):
        alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _analyst_user
        resp = api_client.post(
            "/api/alerts/99999/investigations",
            json={"status": "open"},
        )
        assert resp.status_code == 404


class TestSSEEndpoint:
    def test_stream_requires_auth(self, api_client):
        # No auth override — should get 401
        resp = api_client.get("/api/alerts/stream", params={"since_id": 0})
        assert resp.status_code == 401

    def test_stream_route_exists_and_enforces_auth(self, api_client):
        # No auth override — endpoint should reject with 401, not 404.
        # (TestClient buffers the full body for streaming responses, so we
        # cannot read an infinite SSE stream here. Auth behaviour is sufficient
        # to confirm routing is wired up correctly.)
        resp = api_client.get("/api/alerts/stream", params={"since_id": 0})
        assert resp.status_code == 401
