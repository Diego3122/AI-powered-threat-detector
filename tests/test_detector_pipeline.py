from services.detector import detector


def test_extract_window_payload_uses_precomputed_text_when_present():
    """Pre-computed text in the record should be used as-is (avoids double generation)."""
    precomputed = "login_total=3 signal_failed_no_mfa_present"
    record = {
        "id": "1700000000000-1700000010000",
        "text": precomputed,
        "window": {
            "window_start_ms": 1700000000000,
            "window_end_ms": 1700000010000,
            "counts_by_user": {"alice": 3},
            "failed_no_mfa_by_user": {"alice": 2},
        },
    }

    window, text, window_id = detector._extract_window_payload(record)

    assert window["counts_by_user"]["alice"] == 3
    assert text == precomputed
    assert window_id == record["id"]


def test_extract_window_payload_generates_text_from_window_when_no_text():
    """Without a pre-computed text field, text is generated from the window."""
    record = {
        "id": "1700000000000-1700000010000",
        "window": {
            "feature_schema": "network_flow_v1",
            "window_start_ms": 1700000000000,
            "window_end_ms": 1700000010000,
            "proto": "tcp",
            "service": "http",
            "state": "FIN",
            "sbytes": 1200,
            "dbytes": 400,
        },
    }

    window, text, window_id = detector._extract_window_payload(record)

    assert window["proto"] == "tcp"
    assert "feature_schema=network_flow_v1" in text
    assert "proto=tcp" in text
    assert window_id == record["id"]


def test_build_alert_uses_model_output_and_window_context():
    window = {
        "feature_schema": "network_flow_v1",
        "window_start_ms": 1700000000000,
        "window_end_ms": 1700000010000,
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "sbytes": 1200,
        "dbytes": 400,
    }
    result = {"label": 1, "score": 0.91, "model": "distilbert"}

    alert = detector._build_alert(window, "1700000000000-1700000010000", result, 0.5)

    assert alert["window_id"] == "1700000000000-1700000010000"
    assert alert["model_type"] == "distilbert"
    assert alert["model_score"] == 0.91
    assert "network flow" in alert["explanation_summary"]
    assert "score=0.91" in alert["explanation_summary"]


def test_build_alert_supports_structured_baseline_outputs():
    window = {
        "feature_schema": "network_flow_v1",
        "window_start_ms": 1700000000000,
        "window_end_ms": 1700000010000,
        "proto": "udp",
        "service": "dns",
        "state": "INT",
        "sbytes": 80,
        "dbytes": 120,
    }
    result = {
        "label": 1,
        "score": 0.88,
        "model": "structured_baseline",
        "threshold": 0.61,
    }

    alert = detector._build_alert(window, "1700000000000-1700000010000", result, 0.5)

    assert alert["model_type"] == "structured_baseline"
    assert "experts" not in alert
    assert "consensus" not in alert
    assert "network flow" in alert["explanation_summary"]
    assert "proto=udp" in alert["explanation_summary"]


def test_build_alert_uses_effective_threshold_for_model_label():
    window = {
        "feature_schema": "network_flow_v1",
        "window_start_ms": 1700000000000,
        "window_end_ms": 1700000010000,
        "proto": "tcp",
        "service": "-",
        "state": "con",
        "sbytes": 100,
        "dbytes": 50,
    }
    result = {
        "label": 1,
        "score": 0.61,
        "model": "structured_baseline",
        "threshold": 0.5,
    }

    alert = detector._build_alert(window, "1700000000000-1700000010000", result, 0.75)

    assert alert["model_label"] == 0
    assert alert["threshold"] == 0.75
