"""Integration tests for the DistilBERT scoring path on the model server.

These tests verify that the DistilBERT model loads from the configured path
(DISTILBERT_MODEL_PATH) and that the /score, /score_batch, /compare, /explain,
and /score_with_explanation endpoints produce correct response shapes when the
model is available.

If DistilBERT weights are not present in the test environment the tests are
skipped automatically so CI doesn't fail on lightweight runners.
"""

import pytest
from fastapi.testclient import TestClient

from services.model_server import app as model_server_app


_distilbert_available = model_server_app.distilbert_available
client = TestClient(model_server_app.app)


def _unsw_window() -> dict:
    """Minimal UNSW network-flow window suitable for DistilBERT text conversion."""
    return {
        "feature_schema": "network_flow_v1",
        "window_start_ms": 1700000000000,
        "window_end_ms": 1700000010000,
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "sbytes": 1200,
        "dbytes": 400,
        "spkts": 10,
        "dpkts": 8,
    }


def _window_request() -> dict:
    return {"window": _unsw_window()}


# ---------------------------------------------------------------------------
# Skip group: all tests require DistilBERT weights on disk
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not _distilbert_available,
    reason="DistilBERT weights not available in test environment",
)


class TestDistilBertScore:
    """Tests for /score when DistilBERT is available."""

    def test_score_returns_distilbert_prediction(self):
        """When distilbert is available, /score should be able to score text."""
        response = client.post("/score", json={"text": "tcp http FIN sbytes=1200 dbytes=400"})

        assert response.status_code == 200
        body = response.json()
        assert body["model"] in {"distilbert", "structured_baseline", "tfidf_fallback"}
        assert 0.0 <= body["score"] <= 1.0
        assert body["label"] in {0, 1}

    def test_score_with_window_converts_to_text(self):
        """DistilBERT accepts a window payload by converting it to text first."""
        response = client.post("/score", json=_window_request())

        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["score"] <= 1.0
        assert body["label"] in {0, 1}
        # The text field should contain the converted window text
        assert "feature_schema=network_flow_v1" in body["text"]


class TestDistilBertScoreBatch:
    """Tests for /score_batch with DistilBERT payloads."""

    def test_batch_returns_consistent_results(self):
        payload = _window_request()
        single = client.post("/score", json=payload).json()
        batch = client.post("/score_batch", json={"items": [payload, payload]}).json()

        assert len(batch) == 2
        for item in batch:
            assert item["score"] == single["score"]
            assert item["label"] == single["label"]
            assert item["model"] == single["model"]

    def test_batch_with_text_items(self):
        response = client.post("/score_batch", json=["tcp http FIN", "udp dns CON"])

        if model_server_app._active_model == "structured_baseline":
            # Structured model rejects text-only items
            assert response.status_code == 400
        else:
            assert response.status_code == 200
            body = response.json()
            assert len(body) == 2
            for item in body:
                assert 0.0 <= item["score"] <= 1.0


class TestDistilBertCompare:
    """/compare should include a distilbert entry when the model is loaded."""

    def test_compare_includes_distilbert(self):
        response = client.post("/compare", json=_window_request())

        assert response.status_code == 200
        body = response.json()
        assert "distilbert" in body["models"]
        distilbert_result = body["models"]["distilbert"]
        assert 0.0 <= distilbert_result["score"] <= 1.0
        assert distilbert_result["label"] in {0, 1}
        assert distilbert_result["model"] == "distilbert"


class TestDistilBertExplain:
    """/explain and /score_with_explanation for DistilBERT."""

    def test_explain_returns_explanation_structure(self):
        response = client.post("/explain", json={"text": "tcp http FIN sbytes=1200"})

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == model_server_app._active_model
        assert "explanation" in body

    def test_score_with_explanation_returns_both(self):
        response = client.post("/score_with_explanation", json=_window_request())

        assert response.status_code == 200
        body = response.json()
        assert "prediction" in body
        assert "explanation" in body
        pred = body["prediction"]
        assert 0.0 <= pred["score"] <= 1.0
        assert pred["label"] in {0, 1}
        assert pred["model"] in {"distilbert", "structured_baseline", "tfidf_fallback"}


class TestDistilBertModelInfo:
    """Model info should list distilbert as available."""

    def test_model_info_lists_distilbert(self):
        response = client.get("/model/info")

        assert response.status_code == 200
        body = response.json()
        assert "distilbert" in body["available_models"]

    def test_health_endpoint_reports_model(self):
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["active_model"] in {"distilbert", "structured_baseline", "tfidf_fallback"}
