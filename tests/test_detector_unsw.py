"""
Tests for UNSW-NB15 (network_flow_v1) support in the detector service.
Verifies schema-aware explanation dispatch and correct text extraction.
"""
from services.detector import detector
from services.ml.ml_utils import NETWORK_FLOW_FEATURE_SCHEMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unsw_window(**overrides):
    base = {
        "feature_schema": NETWORK_FLOW_FEATURE_SCHEMA,
        "window_start_ms": 1700000000000,
        "window_end_ms": 1700000010000,
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "sbytes": 1200,
        "dbytes": 400,
        "spkts": 8,
        "dpkts": 5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _build_explanation: network flow dispatch
# ---------------------------------------------------------------------------

def test_build_explanation_returns_network_flow_text_for_unsw():
    window = _make_unsw_window()
    explanation = detector._build_explanation(window, "structured_baseline", 0.87)

    assert "network flow" in explanation
    assert "proto=tcp" in explanation
    assert "service=http" in explanation
    assert "state=FIN" in explanation
    assert "sbytes=1200" in explanation
    assert "dbytes=400" in explanation
    assert "score=0.87" in explanation
    # must NOT contain CloudTrail-specific language
    assert "mfa" not in explanation.lower()
    assert "login" not in explanation.lower()


def test_build_explanation_includes_attack_cat_when_present():
    window = _make_unsw_window(metadata={"attack_cat": "Reconnaissance"})
    explanation = detector._build_explanation(window, "structured_baseline", 0.92)

    assert "[Reconnaissance]" in explanation


def test_build_explanation_omits_attack_cat_bracket_when_absent():
    window = _make_unsw_window()
    explanation = detector._build_explanation(window, "structured_baseline", 0.75)

    assert "[" not in explanation


# ---------------------------------------------------------------------------
# _build_alert: integrates network flow explanation
# ---------------------------------------------------------------------------

def test_build_alert_for_unsw_window_has_network_flow_explanation():
    window = _make_unsw_window(proto="udp", service="dns", state="INT", sbytes=80, dbytes=120)
    result = {"label": 1, "score": 0.82, "model": "structured_baseline", "threshold": 0.5}

    alert = detector._build_alert(window, "unsw-001", result, 0.5)

    assert alert["model_label"] == 1
    assert alert["model_score"] == 0.82
    assert "network flow" in alert["explanation_summary"]
    assert "proto=udp" in alert["explanation_summary"]
    assert "service=dns" in alert["explanation_summary"]


# ---------------------------------------------------------------------------
# _extract_window_payload: text extraction for UNSW records
# ---------------------------------------------------------------------------

def test_extract_window_payload_handles_unsw_window_object():
    window = _make_unsw_window()
    record = {"id": "unsw-abc", "window": window}

    extracted_window, text, window_id = detector._extract_window_payload(record)

    assert extracted_window["feature_schema"] == NETWORK_FLOW_FEATURE_SCHEMA
    assert window_id == "unsw-abc"
    # text must contain UNSW feature markers from ml_utils._network_flow_window_to_text
    assert "feature_schema=network_flow_v1" in text


def test_extract_window_payload_prefers_record_text_over_generated():
    """If the record already has a 'text' field, use it (avoids double generation)."""
    window = _make_unsw_window()
    record = {"id": "unsw-xyz", "text": "prebuilt text", "window": window}

    _, text, _ = detector._extract_window_payload(record)

    assert text == "prebuilt text"


def test_extract_window_payload_falls_back_to_window_to_text_when_no_text():
    window = _make_unsw_window()
    record = {"id": "unsw-fallback", "window": window}

    _, text, _ = detector._extract_window_payload(record)

    assert len(text) > 0
    assert "network_flow_v1" in text
