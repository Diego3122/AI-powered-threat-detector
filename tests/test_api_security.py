from types import SimpleNamespace

from fastapi.testclient import TestClient

from services.api import alerts_api
from services.api.auth import TokenData
from services.api.security import get_client_ip, normalize_text, rate_limiter
from services.database.models import get_db_url


client = TestClient(alerts_api.app)


def setup_function():
    rate_limiter._buckets.clear()


def _override_viewer_user():
    return TokenData(username="viewer", roles=["viewer"])


def _fail_get_db():
    raise AssertionError("get_db should not be called for this request")


def test_login_rejects_unsafe_username():
    response = client.post(
        "/api/auth/login",
        json={"username": "admin user", "password": "bad-password"},
    )

    assert response.status_code == 422
    assert "username" in response.json()["detail"]


def test_login_rate_limit_is_enforced():
    for _ in range(5):
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        assert response.status_code == 401

    blocked = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "wrong-password"},
    )

    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "Rate limit exceeded"


def test_alert_filter_rejects_unsafe_query_input():
    alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _override_viewer_user
    alerts_api.app.dependency_overrides[alerts_api.get_db] = _fail_get_db

    try:
        response = client.get(
            "/api/alerts",
            params={"model_type": "distilbert<script>"},
        )
    finally:
        alerts_api.app.dependency_overrides.clear()

    assert response.status_code == 422
    assert "model_type" in response.json()["detail"]


def test_alert_list_requires_auth():
    alerts_api.app.dependency_overrides[alerts_api.get_db] = _fail_get_db

    try:
        response = client.get("/api/alerts")
    finally:
        alerts_api.app.dependency_overrides.clear()

    assert response.status_code == 401


def test_alert_list_allows_viewer_read(monkeypatch):
    def fake_get_db():
        yield object()

    fake_alert = SimpleNamespace(
        id=1,
        timestamp=1700000000000,
        window_id="1700000000000-1700000010000",
        model_type="distilbert",
        model_score=0.95,
        threshold=0.5,
        triggered=True,
        created_at="2026-01-01T00:00:00",
    )

    monkeypatch.setattr(alerts_api, "AlertService", lambda db: SimpleNamespace(get_alerts=lambda **kwargs: [fake_alert]))
    alerts_api.app.dependency_overrides[alerts_api.get_db] = fake_get_db
    alerts_api.app.dependency_overrides[alerts_api.get_current_user] = _override_viewer_user

    try:
        response = client.get("/api/alerts")
    finally:
        alerts_api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()[0]["window_id"] == "1700000000000-1700000010000"


def test_audit_logs_require_auth():
    alerts_api.app.dependency_overrides[alerts_api.get_db] = _fail_get_db

    try:
        response = client.get("/api/audit/logs")
    finally:
        alerts_api.app.dependency_overrides.clear()

    assert response.status_code == 401


def test_normalize_text_strips_control_characters():
    sanitized = normalize_text("  hello\x00world\r\n", field_name="text")

    assert sanitized == "helloworld"


def test_internal_alert_ingest_requires_shared_key():
    alerts_api.app.dependency_overrides[alerts_api.get_db] = _fail_get_db

    try:
        response = client.post(
            "/api/internal/alerts/ingest",
            json={
                "timestamp": 1700000000000,
                "window_id": "1700000000000-1700000010000",
                "model_type": "distilbert",
                "model_score": 0.95,
                "threshold": 0.5,
            },
        )
    finally:
        alerts_api.app.dependency_overrides.clear()

    assert response.status_code == 401


def test_default_db_url_uses_psycopg_driver():
    assert get_db_url().startswith("postgresql+psycopg://")


def test_internal_alert_ingest_accepts_valid_payload(monkeypatch):
    captured = {}

    def fake_get_db():
        yield object()

    def fake_create_alert(**kwargs):
        captured["alert_kwargs"] = kwargs
        return SimpleNamespace(
            id=7,
            timestamp=kwargs["timestamp"],
            window_id=kwargs["window_id"],
            model_type=kwargs["model_type"],
            model_score=kwargs["model_score"],
            threshold=kwargs["threshold"],
            triggered=kwargs["triggered"],
            created_at="2026-01-01T00:00:00",
        )

    def fake_log_action(**kwargs):
        captured["audit_kwargs"] = kwargs
        return None

    monkeypatch.setattr(alerts_api, "AlertService", lambda db: SimpleNamespace(create_alert=fake_create_alert))
    monkeypatch.setattr(alerts_api, "AuditService", lambda db: SimpleNamespace(log_action=fake_log_action))
    alerts_api.app.dependency_overrides[alerts_api.get_db] = fake_get_db

    try:
        response = client.post(
            "/api/internal/alerts/ingest",
            headers={"X-Internal-API-Key": "dev-internal-api-key"},
            json={
                "timestamp": 1700000000000,
                "window_id": "1700000000000-1700000010000",
                "model_type": "distilbert",
                "model_score": 0.95,
                "threshold": 0.5,
                "model_label": 1,
                "explanation_summary": "distilbert flagged repeated failed logins without MFA for alice=3",
                "window": {
                    "window_start_ms": 1700000000000,
                    "window_end_ms": 1700000010000,
                    "failed_no_mfa_by_user": {"alice": 3},
                },
            },
        )
    finally:
        alerts_api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["alert_kwargs"]["triggered"] is True
    assert captured["alert_kwargs"]["window_id"] == "1700000000000-1700000010000"
    assert captured["audit_kwargs"]["action"] == "model_alert_ingested"


def test_internal_alert_ingest_accepts_structured_model_type(monkeypatch):
    captured = {}

    def fake_get_db():
        yield object()

    def fake_create_alert(**kwargs):
        captured["alert_kwargs"] = kwargs
        return SimpleNamespace(
            id=8,
            timestamp=kwargs["timestamp"],
            window_id=kwargs["window_id"],
            model_type=kwargs["model_type"],
            model_score=kwargs["model_score"],
            threshold=kwargs["threshold"],
            triggered=kwargs["triggered"],
            created_at="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(alerts_api, "AlertService", lambda db: SimpleNamespace(create_alert=fake_create_alert))
    monkeypatch.setattr(alerts_api, "AuditService", lambda db: SimpleNamespace(log_action=lambda **kwargs: None))
    alerts_api.app.dependency_overrides[alerts_api.get_db] = fake_get_db

    try:
        response = client.post(
            "/api/internal/alerts/ingest",
            headers={"X-Internal-API-Key": "dev-internal-api-key"},
            json={
                "timestamp": 1700000000000,
                "window_id": "1700000000000-1700000010000",
                "model_type": "structured_baseline",
                "model_score": 0.91,
                "threshold": 0.5,
                "model_label": 1,
            },
        )
    finally:
        alerts_api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["alert_kwargs"]["model_type"] == "structured_baseline"


def test_trusted_host_middleware_blocks_unexpected_host():
    response = client.get("/api/alerts", headers={"host": "evil.example"})

    assert response.status_code == 400


def test_get_client_ip_ignores_forwarded_for_from_public_client():
    request = SimpleNamespace(
        headers={"x-forwarded-for": "198.51.100.10"},
        client=SimpleNamespace(host="8.8.8.8"),
    )

    assert get_client_ip(request) == "8.8.8.8"


def test_get_client_ip_accepts_forwarded_for_from_private_proxy():
    request = SimpleNamespace(
        headers={"x-forwarded-for": "198.51.100.10"},
        client=SimpleNamespace(host="172.18.0.5"),
    )

    assert get_client_ip(request) == "198.51.100.10"
