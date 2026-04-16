from fastapi.testclient import TestClient

from services.model_server import app as model_server_app


client = TestClient(model_server_app.app)


def _window_request_payload() -> dict:
    return {
        "text": "login_total=5 failed_no_mfa_total=3 signal_failed_no_mfa_present signal_mfa_bypass_activity",
        "window": {
            "window_start_ms": 1700000000000,
            "window_end_ms": 1700000010000,
            "counts_by_user": {"alice": 5},
            "failed_no_mfa_by_user": {"alice": 3},
        },
    }


def test_model_server_health_reports_active_model():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["active_model"] == model_server_app._active_model


def test_model_info_reports_available_models():
    response = client.get("/model/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_model"] == model_server_app._active_model
    assert set(payload["available_models"]) == set(model_server_app.model_info().available_models)


def test_score_batch_rejects_empty_payload_with_422():
    response = client.post("/score_batch", json=[])

    assert response.status_code == 422
    assert response.json()["detail"] == "texts cannot be empty"


def test_score_batch_rejects_invalid_text_with_422():
    response = client.post("/score_batch", json=["   "])

    assert response.status_code == 422
    assert "text cannot be empty" in response.json()["detail"]


def test_score_batch_rejects_oversized_batches_with_422():
    response = client.post("/score_batch", json=["signal"] * (model_server_app.MAX_BATCH_SIZE + 1))

    assert response.status_code == 422
    assert response.json()["detail"] == f"Batch size exceeds max of {model_server_app.MAX_BATCH_SIZE}"


def test_score_returns_runtime_prediction_shape():
    response = client.post("/score", json=_window_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] in {"structured_baseline", "tfidf_fallback", "distilbert"}
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["label"] in {0, 1}
    assert "threshold" in payload


def test_score_requires_window_when_structured_model_is_active():
    response = client.post("/score", json={"text": "signal_only"})

    if model_server_app._active_model == "structured_baseline":
        assert response.status_code == 400
        assert "requires a window payload" in response.json()["detail"]
    else:
        assert response.status_code == 200


def test_score_batch_matches_single_score_path_for_window_payloads():
    payload = _window_request_payload()
    single_response = client.post("/score", json=payload)
    batch_response = client.post("/score_batch", json={"items": [payload, payload]})

    assert single_response.status_code == 200
    assert batch_response.status_code == 200

    single_prediction = single_response.json()
    batch_predictions = batch_response.json()
    assert len(batch_predictions) == 2
    for batch_prediction in batch_predictions:
        assert batch_prediction["model"] == single_prediction["model"]
        assert batch_prediction["label"] == single_prediction["label"]
        assert batch_prediction["score"] == single_prediction["score"]
        assert batch_prediction["threshold"] == single_prediction["threshold"]


def test_score_batch_accepts_structured_window_only_items():
    payload = {
        "window": {
            "feature_schema": "network_flow_v1",
            "window_start_ms": 1700000000000,
            "window_end_ms": 1700000010000,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "sbytes": 1200,
            "dbytes": 400,
        }
    }
    response = client.post("/score_batch", json={"items": [payload]})

    if model_server_app._active_model == "structured_baseline":
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["model"] == "structured_baseline"
        assert body[0]["text"].startswith("feature_schema=network_flow_v1")
    else:
        assert response.status_code in {200, 400}


def test_score_batch_string_payload_is_rejected_when_structured_model_is_active():
    response = client.post("/score_batch", json=["signal_only"])

    if model_server_app._active_model == "structured_baseline":
        assert response.status_code == 400
        assert "requires a window payload" in response.json()["detail"]
    else:
        assert response.status_code == 200


def test_compare_includes_active_model():
    response = client.post("/compare", json=_window_request_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_model"] == model_server_app._active_model
    assert "structured_baseline" in payload["models"]


def test_compare_accepts_window_only_payload():
    payload = {
        "window": {
            "window_start_ms": 1700000000000,
            "window_end_ms": 1700000010000,
            "counts_by_user": {"alice": 5},
            "failed_no_mfa_by_user": {"alice": 3},
        }
    }
    response = client.post("/compare", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["models"]
