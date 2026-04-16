#!/usr/bin/env python
import os
import sys
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, field_validator

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _HAVE_TRANSFORMERS = True
except Exception:
    _HAVE_TRANSFORMERS = False

from services.api.security import MAX_BATCH_SIZE, enforce_rate_limit, normalize_text
from services.ml.ml_utils import resolve_window_threat_family, score_structured_inputs, window_to_feature_dict, window_to_text
from services.model_server.explainability import DistilBertExplainer, TFIDFExplainer, format_explanation

# Configuration
MODEL_TYPE = os.getenv("MODEL_TYPE", "structured").lower()
DEVICE = os.getenv("DEVICE", "cpu")
TFIDF_MODEL_PATH = os.getenv("TFIDF_MODEL_PATH", "models/baseline.pkl")
DISTILBERT_MODEL_PATH = os.getenv("DISTILBERT_MODEL_PATH", "models/distilbert_finetuned")
DISTILBERT_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

_text_model = None
_text_vectorizer = None
_structured_model = None
_structured_vectorizer = None
_structured_threshold = 0.5
_structured_family_models = {}
_structured_family_vectorizers = {}
_structured_family_thresholds = {}
_distilbert_model = None
_distilbert_tokenizer = None
_text_explainer = None
_distilbert_explainer = None
_active_model = MODEL_TYPE


def _load_baseline_bundle() -> None:
    global _text_model, _text_vectorizer, _structured_model, _structured_vectorizer, _structured_threshold, _text_explainer
    global _structured_family_models, _structured_family_vectorizers, _structured_family_thresholds
    if not os.path.exists(TFIDF_MODEL_PATH):
        raise FileNotFoundError(f"Baseline model not found at {TFIDF_MODEL_PATH}")

    model_dict = joblib.load(TFIDF_MODEL_PATH)
    _text_model = model_dict["model"]
    _text_vectorizer = model_dict["vectorizer"]
    _structured_model = model_dict.get("structured_model")
    _structured_vectorizer = model_dict.get("structured_vectorizer")
    _structured_threshold = float(model_dict.get("structured_threshold", 0.5))
    _structured_family_models = model_dict.get("structured_family_models") or {}
    _structured_family_vectorizers = model_dict.get("structured_family_vectorizers") or {}
    _structured_family_thresholds = model_dict.get("structured_family_thresholds") or {}
    _text_explainer = TFIDFExplainer(_text_model, _text_vectorizer)
    print(f"Loaded baseline bundle from {TFIDF_MODEL_PATH}")


def _structured_available() -> bool:
    return _structured_model is not None and _structured_vectorizer is not None


def _text_available() -> bool:
    return _text_model is not None and _text_vectorizer is not None


def _distilbert_weights_available() -> bool:
    return any(os.path.exists(os.path.join(DISTILBERT_MODEL_PATH, filename)) for filename in DISTILBERT_WEIGHT_FILES)


def _load_distilbert() -> bool:
    global _distilbert_model, _distilbert_tokenizer, _distilbert_explainer
    if not _HAVE_TRANSFORMERS:
        print("transformers not installed; DistilBERT unavailable", file=sys.stderr)
        return False

    if not os.path.exists(DISTILBERT_MODEL_PATH):
        print(f"DistilBERT model not found at {DISTILBERT_MODEL_PATH}", file=sys.stderr)
        return False
    if not _distilbert_weights_available():
        expected = ", ".join(DISTILBERT_WEIGHT_FILES)
        print(
            f"DistilBERT weights missing in {DISTILBERT_MODEL_PATH}. Expected one of: {expected}",
            file=sys.stderr,
        )
        return False

    try:
        _distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
        _distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
        _distilbert_model.to(DEVICE)
        _distilbert_model.eval()
        _distilbert_explainer = DistilBertExplainer(_distilbert_model, _distilbert_tokenizer, device=DEVICE)
        print(f"Loaded DistilBERT model from {DISTILBERT_MODEL_PATH} (device: {DEVICE})")
        return True
    except Exception as exc:
        print(f"Failed to load DistilBERT: {exc}", file=sys.stderr)
        return False


print(f"Initializing models (MODEL_TYPE={MODEL_TYPE}, DEVICE={DEVICE})...")
baseline_available = True
try:
    _load_baseline_bundle()
except Exception as exc:
    baseline_available = False
    print(f"WARNING: Failed to load baseline bundle: {exc}", file=sys.stderr)

distilbert_available = _load_distilbert()

requested_model = MODEL_TYPE

if requested_model in {"structured", "structured_baseline"}:
    if _structured_available():
        _active_model = "structured_baseline"
    elif _text_available():
        print("ERROR: Structured model unavailable. Falling back to TF-IDF.", file=sys.stderr)
        _active_model = "tfidf_fallback"
    elif distilbert_available:
        print("ERROR: Structured model unavailable. Falling back to DistilBERT.", file=sys.stderr)
        _active_model = "distilbert"
elif requested_model == "distilbert":
    if distilbert_available:
        _active_model = "distilbert"
    elif _structured_available():
        print("ERROR: DistilBERT unavailable. Falling back to structured baseline.", file=sys.stderr)
        _active_model = "structured_baseline"
    elif _text_available():
        print("ERROR: DistilBERT unavailable. Falling back to TF-IDF.", file=sys.stderr)
        _active_model = "tfidf_fallback"
elif requested_model == "tfidf_fallback":
    if _text_available():
        _active_model = "tfidf_fallback"
    elif _structured_available():
        print("ERROR: TF-IDF unavailable. Falling back to structured baseline.", file=sys.stderr)
        _active_model = "structured_baseline"
    elif distilbert_available:
        print("ERROR: TF-IDF unavailable. Falling back to DistilBERT.", file=sys.stderr)
        _active_model = "distilbert"

print(f"Active model: {_active_model}")

app = FastAPI(title="Threat Detector", version="3.1")

http_requests_total = Counter(
    "model_server_http_requests_total",
    "Total HTTP requests handled by the model server",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "model_server_http_request_duration_seconds",
    "HTTP request duration in seconds for the model server",
    ["method", "endpoint"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    import time

    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.time() - start_time
        try:
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status_code,
            ).inc()
        except Exception:
            pass

    return response


class WindowRequest(BaseModel):
    text: str | None = None
    window: dict[str, Any] | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_text(value, field_name="text", max_length=5000)


class ScoreResponse(BaseModel):
    text: str
    label: int
    score: float
    model: str
    threshold: float | None = None


class ModelInfoResponse(BaseModel):
    active_model: str
    available_models: list[str]
    device: str
    active_threshold: float | None = None
    active_threshold_mode: str | None = None
    active_family_thresholds: dict[str, float] | None = None


def _score_tfidf(text: str) -> dict[str, Any]:
    if _text_model is None:
        raise ValueError("TF-IDF model not loaded")

    x_vec = _text_vectorizer.transform([text])
    proba = _text_model.predict_proba(x_vec)[0]
    score_val = float(proba[1]) if len(proba) >= 2 else float(proba[0]) if len(proba) == 1 else 0.0
    return {"label": int(score_val >= 0.5), "score": score_val, "model": "tfidf_fallback", "threshold": 0.5}


def _score_structured(window: dict[str, Any]) -> dict[str, Any]:
    if not _structured_available():
        raise ValueError("Structured model not loaded")

    family = resolve_window_threat_family(window)
    family_model = _structured_family_models.get(family)
    family_vectorizer = _structured_family_vectorizers.get(family)
    if family_model is not None and family_vectorizer is not None:
        family_threshold = float(_structured_family_thresholds.get(family, _structured_threshold))
        features = window_to_feature_dict(window)
        matrix = family_vectorizer.transform([features])
        score_val = float(family_model.predict_proba(matrix)[0][1])
        return {
            "label": int(score_val >= family_threshold),
            "score": score_val,
            "model": "structured_baseline",
            "threshold": family_threshold,
        }

    bundle = {
        "structured_model": _structured_model,
        "structured_vectorizer": _structured_vectorizer,
        "structured_threshold": _structured_threshold,
    }
    return score_structured_inputs(bundle, [window])[0]


def _score_distilbert(text: str) -> dict[str, Any]:
    if _distilbert_model is None:
        raise ValueError("DistilBERT model not loaded")

    with torch.no_grad():
        inputs = _distilbert_tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        outputs = _distilbert_model(**inputs)
        proba = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        score_val = float(proba[1]) if len(proba) >= 2 else float(proba[0]) if len(proba) == 1 else 0.0
    return {"label": int(score_val >= 0.5), "score": score_val, "model": "distilbert", "threshold": 0.5}


def _structured_explanation(window: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    if not _structured_available():
        return {"model": "structured_baseline", "top_features": [], "error": "Structured model not loaded"}

    family = resolve_window_threat_family(window)
    vectorizer = _structured_family_vectorizers.get(family, _structured_vectorizer)
    model = _structured_family_models.get(family, _structured_model)
    features = window_to_feature_dict(window)
    matrix = vectorizer.transform([features])
    dense = matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0] if hasattr(model, "coef_") else None

    nonzero_indices = dense.nonzero()[0]
    ranked: list[dict[str, Any]] = []
    for idx in nonzero_indices:
        value = dense[idx]
        coef = coefficients[idx] if coefficients is not None and idx < len(coefficients) else 0.0
        ranked.append(
            {
                "feature": feature_names[idx],
                "importance_score": float(abs(value * coef) if coefficients is not None else abs(value)),
                "contribution": "increases_threat" if coef > 0 else "decreases_threat",
                "feature_value": float(value),
            }
        )

    ranked.sort(key=lambda item: abs(item["importance_score"]), reverse=True)
    return {
        "model": "structured_baseline",
        "explanation_type": "Linear Feature Importance",
        "top_features": ranked[:top_k],
    }


def _require_structured_window(window: dict[str, Any] | None) -> dict[str, Any]:
    if not window:
        raise ValueError("Active structured baseline requires a window payload")
    return window


def _request_text(req: WindowRequest) -> str:
    if req.text:
        return req.text
    if req.window:
        return window_to_text(req.window)
    raise ValueError("Scoring requires either text or a window payload")


def _parse_batch_requests(payload: Any) -> list[WindowRequest]:
    raw_items = payload
    if isinstance(payload, dict):
        if "items" in payload:
            raw_items = payload["items"]
        elif "texts" in payload:
            raw_items = payload["texts"]
        else:
            raise HTTPException(
                status_code=422,
                detail="score_batch expects a JSON array or an object with 'items' or 'texts'",
            )

    if not isinstance(raw_items, list):
        raise HTTPException(status_code=422, detail="score_batch payload must be a JSON array")
    if not raw_items:
        raise HTTPException(status_code=422, detail="texts cannot be empty")
    if len(raw_items) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds max of {MAX_BATCH_SIZE}")

    batch_requests: list[WindowRequest] = []
    for item in raw_items:
        if isinstance(item, str):
            batch_requests.append(WindowRequest(text=normalize_text(item, field_name="text", max_length=5000)))
            continue
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="Batch items must be strings or objects")

        window = item.get("window")
        if window is not None and not isinstance(window, dict):
            raise HTTPException(status_code=422, detail="window must be an object when provided")
        raw_text = item.get("text")
        if raw_text is None and window is None:
            raise HTTPException(status_code=422, detail="Batch items must include text or window")
        normalized_text = normalize_text(raw_text, field_name="text", max_length=5000) if raw_text is not None else None
        batch_requests.append(WindowRequest(text=normalized_text, window=window))

    return batch_requests


def _format_response_payload(req: WindowRequest, result: dict[str, Any]) -> ScoreResponse:
    payload = dict(result)
    payload.pop("text", None)
    return ScoreResponse(text=_request_text(req), **payload)


def _explain_for_model(model_name: str, req: WindowRequest) -> dict[str, Any]:
    if model_name == "distilbert":
        explanation = _distilbert_explainer.explain(_request_text(req), top_k=5) if _distilbert_explainer else {}
        return format_explanation(explanation, "distilbert")
    if model_name == "structured_baseline":
        return _structured_explanation(_require_structured_window(req.window), top_k=5)
    explanation = _text_explainer.explain(_request_text(req), top_k=5) if _text_explainer else {}
    return format_explanation(explanation, "tfidf_fallback")


def _score_request(req: WindowRequest) -> dict[str, Any]:
    if _active_model == "structured_baseline":
        return _score_structured(_require_structured_window(req.window))
    if _active_model == "distilbert":
        return _score_distilbert(_request_text(req))
    return _score_tfidf(_request_text(req))


@app.get("/health")
def health():
    return {"status": "ok", "active_model": _active_model}


@app.get("/model/info")
def model_info() -> ModelInfoResponse:
    available = []
    if _structured_available():
        available.append("structured_baseline")
    if _text_available():
        available.append("tfidf_fallback")
    if distilbert_available:
        available.append("distilbert")

    active_threshold = None
    active_threshold_mode = None
    active_family_thresholds = None
    if _active_model == "structured_baseline":
        if _structured_family_thresholds:
            active_threshold_mode = "per_family"
            active_family_thresholds = {
                family_name: float(threshold)
                for family_name, threshold in sorted(_structured_family_thresholds.items())
            }
        else:
            active_threshold = _structured_threshold
            active_threshold_mode = "global"
    elif _active_model in {"tfidf_fallback", "distilbert"}:
        active_threshold = 0.5
        active_threshold_mode = "global"

    return ModelInfoResponse(
        active_model=_active_model,
        available_models=available,
        device=DEVICE,
        active_threshold=active_threshold,
        active_threshold_mode=active_threshold_mode,
        active_family_thresholds=active_family_thresholds,
    )


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score")
def score(req: WindowRequest, request: Request) -> ScoreResponse:
    enforce_rate_limit(request, scope="model_score", limit=30, window_seconds=60)
    try:
        return _format_response_payload(req, _score_request(req))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}") from exc


@app.post("/score_batch")
def score_batch(payload: Any = Body(...), request: Request = None) -> list[ScoreResponse]:
    enforce_rate_limit(request, scope="model_score_batch", limit=10, window_seconds=60)
    try:
        batch_requests = _parse_batch_requests(payload)
        return [_format_response_payload(req, _score_request(req)) for req in batch_requests]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Batch scoring failed: {exc}") from exc


@app.post("/compare")
def compare(req: WindowRequest, request: Request) -> dict[str, Any]:
    enforce_rate_limit(request, scope="model_compare", limit=20, window_seconds=60)
    try:
        text_payload = _request_text(req)
        results: dict[str, Any] = {}
        if req.window and _structured_available():
            results["structured_baseline"] = _score_structured(req.window)
        if _text_available():
            results["tfidf_fallback"] = _score_tfidf(text_payload)
        if distilbert_available:
            results["distilbert"] = _score_distilbert(text_payload)
        if not results:
            raise ValueError("No models available")
        return {"text": text_payload, "models": results, "active_model": _active_model}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Comparison failed: {exc}") from exc


@app.post("/explain")
def explain(req: WindowRequest, request: Request) -> dict[str, Any]:
    enforce_rate_limit(request, scope="model_explain", limit=15, window_seconds=60)
    try:
        formatted = _explain_for_model(_active_model, req)
        return {"text": req.text, "model": _active_model, "explanation": formatted}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Explanation failed: {exc}") from exc


@app.post("/score_with_explanation")
def score_with_explanation(req: WindowRequest, request: Request) -> dict[str, Any]:
    enforce_rate_limit(request, scope="model_score_with_explanation", limit=15, window_seconds=60)
    try:
        result = _score_request(req)
        formatted = _explain_for_model(result["model"], req)
        return {
            "text": req.text,
            "prediction": {
                "label": result["label"],
                "score": result["score"],
                "model": result["model"],
                "threshold": result.get("threshold"),
            },
            "explanation": formatted,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Score+explain failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
