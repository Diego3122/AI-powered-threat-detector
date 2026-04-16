import json
from pathlib import Path

import joblib
import pytest

from services.ml.ml_utils import record_to_window_text, window_to_feature_dict

MODEL_PATH = Path("models/baseline_unsw.pkl")
DATA_PATH = Path("data/unsw_nb15_test.jsonl")
SAMPLE_SIZE = 200


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="baseline_unsw.pkl not trained yet")
@pytest.mark.skipif(not DATA_PATH.exists(), reason="unsw_nb15_test.jsonl not built yet")
def test_baseline_model_scores_sample_dataset():
    model_dict = joblib.load(MODEL_PATH)
    model = model_dict["model"]
    vectorizer = model_dict["vectorizer"]

    all_rows = [
        json.loads(line)
        for line in DATA_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Take equal samples from each class so both threat_scores and normal_scores are non-empty
    half = SAMPLE_SIZE // 2
    positives = [r for r in all_rows if r["label"] == 1][:half]
    negatives = [r for r in all_rows if r["label"] == 0][:half]
    rows = positives + negatives

    texts = [record_to_window_text(row)[1] for row in rows]
    windows = [record_to_window_text(row)[0] for row in rows]
    labels = [row["label"] for row in rows]

    probabilities = model.predict_proba(vectorizer.transform(texts))
    threat_scores = [probability[1] for probability, label in zip(probabilities, labels) if label == 1]
    normal_scores = [probability[1] for probability, label in zip(probabilities, labels) if label == 0]
    structured_model = model_dict["structured_model"]
    structured_vectorizer = model_dict["structured_vectorizer"]
    structured_scores = structured_model.predict_proba(
        structured_vectorizer.transform([window_to_feature_dict(window) for window in windows])
    )[:, 1]

    assert len(rows) > 0
    assert model_dict.get("structured_model") is not None
    assert threat_scores
    assert normal_scores
    assert max(threat_scores) >= 0.5
    assert (sum(threat_scores) / len(threat_scores)) > (sum(normal_scores) / len(normal_scores))
    assert max(structured_scores) >= 0.5
