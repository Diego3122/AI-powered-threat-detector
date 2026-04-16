import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Annotated, Any, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy.orm import Session

from services.api.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    Token,
    TokenData,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_role,
)
from services.api.security import (
    enforce_rate_limit,
    normalize_identifier,
    normalize_optional_identifier,
    normalize_optional_text,
    normalize_text,
    require_internal_api_key,
)
from services.database.db_service import AlertService, AuditService, InvestigationService, ModelService
from services.database.models import Alert, AlertInvestigation, AuditLog, Model, get_db_url, get_session

# Prometheus metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

active_alerts = Gauge(
    "active_alerts_total",
    "Number of active (triggered) alerts",
)

total_alerts = Gauge(
    "total_alerts_total",
    "Total number of alerts",
)

# Configuration
DB_URL = os.getenv("DATABASE_URL", get_db_url())

app = FastAPI(
    title="Threat Detector API",
    version="1.0",
    description="AI-powered cybersecurity threat detection system",
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]
DEFAULT_ALLOWED_HOSTS = "localhost,127.0.0.1,api,testserver"
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS).split(",")
    if host.strip()
]

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def get_db():
    session = get_session(DB_URL)
    try:
        yield session
    finally:
        session.close()


def get_safe_model_type(
    model_type: Annotated[Optional[str], Query(max_length=50)] = None,
) -> Optional[str]:
    return normalize_optional_identifier(model_type, field_name="model_type", max_length=50)


def get_safe_audit_user_id(
    user_id: Annotated[Optional[str], Query(max_length=255)] = None,
) -> Optional[str]:
    return normalize_optional_identifier(user_id, field_name="user_id")


def get_safe_audit_action(
    action: Annotated[Optional[str], Query(max_length=100)] = None,
) -> Optional[str]:
    return normalize_optional_identifier(action, field_name="action", max_length=100)


class AlertCreate(BaseModel):
    timestamp: int
    window_id: str
    model_type: str
    model_score: float
    threshold: float
    explanation_summary: Optional[str] = None

    @field_validator("window_id")
    @classmethod
    def validate_window_id(cls, value: str) -> str:
        return normalize_identifier(value, field_name="window_id")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, value: str) -> str:
        return normalize_identifier(value, field_name="model_type", max_length=50)

    @field_validator("explanation_summary")
    @classmethod
    def validate_explanation_summary(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_text(value, field_name="explanation_summary", max_length=1000)


class AlertUpdate(BaseModel):
    triggered: Optional[bool] = None
    explanation_summary: Optional[str] = None

    @field_validator("explanation_summary")
    @classmethod
    def validate_explanation_summary(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_text(value, field_name="explanation_summary", max_length=1000)


class InternalAlertIngestRequest(BaseModel):
    timestamp: int
    window_id: str
    model_type: str
    model_score: float
    threshold: float
    model_label: Optional[int] = None
    feature_schema: Optional[str] = None
    explanation_summary: Optional[str] = None
    window: Optional[dict[str, Any]] = None

    @field_validator("window_id")
    @classmethod
    def validate_window_id(cls, value: str) -> str:
        return normalize_identifier(value, field_name="window_id")

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, value: str) -> str:
        return normalize_identifier(value, field_name="model_type", max_length=50)

    @field_validator("explanation_summary")
    @classmethod
    def validate_explanation_summary(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_text(value, field_name="explanation_summary", max_length=1000)


class AlertResponse(BaseModel):
    id: int
    timestamp: int
    window_id: str
    model_type: str
    feature_schema: Optional[str] = None
    model_score: float
    threshold: float
    triggered: bool
    explanation_summary: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuditLogResponse(BaseModel):
    id: int
    user_id: Optional[str]
    action: str
    target: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InvestigationCreate(BaseModel):
    user_id: Optional[str] = None
    status: Literal["open", "investigating", "resolved", "false_positive"] = "open"
    notes: Optional[str] = None

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_identifier(value, field_name="user_id")

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_text(value, field_name="notes", max_length=2000)


class InvestigationUpdate(BaseModel):
    status: Optional[Literal["open", "investigating", "resolved", "false_positive"]] = None
    notes: Optional[str] = None

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: Optional[str]) -> Optional[str]:
        return normalize_optional_text(value, field_name="notes", max_length=2000)


class InvestigationResponse(BaseModel):
    id: int
    alert_id: int
    user_id: Optional[str]
    status: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StatsResponse(BaseModel):
    total_alerts: int
    by_model: List[dict]


class LoginRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        return normalize_identifier(value, field_name="username", max_length=64)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        return normalize_text(value, field_name="password", max_length=256)


class ModelResponse(BaseModel):
    id: int
    model_type: str
    version: Optional[str]
    accuracy: Optional[float]
    f1_score: Optional[float]
    roc_auc: Optional[float]
    n_samples: Optional[int]
    active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/alerts", response_model=AlertResponse)
def create_alert(
    alert: AlertCreate,
    request: Request,
    _current_user: TokenData = Depends(require_role("analyst")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="create_alert", limit=30, window_seconds=60)
    service = AlertService(db)
    return service.create_alert(
        timestamp=alert.timestamp,
        window_id=alert.window_id,
        model_type=alert.model_type,
        model_score=alert.model_score,
        threshold=alert.threshold,
        explanation=alert.explanation_summary,
    )


@app.post("/api/internal/alerts/ingest", response_model=AlertResponse)
def ingest_internal_alert(
    alert: InternalAlertIngestRequest,
    request: Request,
    _authorized: None = Depends(require_internal_api_key),
    db: Session = Depends(get_db),
):
    service = AlertService(db)
    result = service.create_alert(
        timestamp=alert.timestamp,
        window_id=alert.window_id,
        model_type=alert.model_type,
        model_score=alert.model_score,
        threshold=alert.threshold,
        explanation=alert.explanation_summary,
        triggered=alert.model_score >= alert.threshold,
        feature_schema=alert.feature_schema,
    )

    AuditService(db).log_action(
        user_id="system:alert-router",
        action="model_alert_ingested",
        target=f"alert:{result.id}",
        details=json.dumps(
            {
                "window_id": alert.window_id,
                "model_type": alert.model_type,
                "score": alert.model_score,
                "label": alert.model_label,
            }
        ),
    )
    return result


@app.get("/api/alerts", response_model=List[AlertResponse])
def list_alerts(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    triggered: Optional[bool] = None,
    _current_user: TokenData = Depends(require_role("viewer")),
    safe_model_type: Optional[str] = Depends(get_safe_model_type),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="list_alerts", limit=120, window_seconds=60)
    service = AlertService(db)
    return service.get_alerts(limit=limit, offset=offset, model_type=safe_model_type, triggered=triggered)


@app.get("/api/alerts/stream")
async def stream_alerts(
    request: Request,
    since_id: int = Query(0, ge=0),
    _current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Server-Sent Events stream of new triggered alerts.

    Clients connect once; new alerts are pushed as ``data: <json>\\n\\n``
    events. A ``: keepalive`` comment is sent when there are no new alerts
    so the connection stays alive through proxies.

    Auth: Bearer token in ``Authorization`` header (use ``fetch``, not the
    browser's native ``EventSource``, which cannot send custom headers).
    """

    async def event_generator():
        last_id = since_id
        while True:
            if await request.is_disconnected():
                break

            alerts = AlertService(db).get_alerts(limit=200, triggered=True)
            new_alerts = sorted(
                (a for a in alerts if a.id > last_id),
                key=lambda a: a.id,
            )
            if new_alerts:
                for alert in new_alerts:
                    payload = AlertResponse.model_validate(alert).model_dump_json()
                    yield f"data: {payload}\n\n"
                    last_id = alert.id
            else:
                yield ": keepalive\n\n"

            await asyncio.sleep(3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/alerts/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: int,
    request: Request,
    _current_user: TokenData = Depends(require_role("viewer")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="get_alert", limit=120, window_seconds=60)
    service = AlertService(db)
    alert = service.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.put("/api/alerts/{alert_id}", response_model=AlertResponse)
def update_alert(
    alert_id: int,
    update: AlertUpdate,
    request: Request,
    _current_user: TokenData = Depends(require_role("analyst")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="update_alert", limit=30, window_seconds=60)
    service = AlertService(db)
    update_data = update.model_dump(exclude_unset=True) if hasattr(update, "model_dump") else update.dict(exclude_unset=True)
    alert = service.update_alert(alert_id, **update_data)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.get("/api/alerts/stats")
def get_stats(
    request: Request,
    hours: int = Query(24, ge=1, le=720),
    _current_user: TokenData = Depends(require_role("viewer")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="get_stats", limit=60, window_seconds=60)
    service = AlertService(db)
    return service.get_alerts_stats(hours=hours)


@app.post("/api/alerts/{alert_id}/investigations", response_model=InvestigationResponse)
def create_investigation(
    alert_id: int,
    investigation: InvestigationCreate,
    request: Request,
    current_user: TokenData = Depends(require_role("analyst")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="create_investigation", limit=20, window_seconds=60)
    service = AlertService(db)
    alert = service.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    actor_user = current_user.username or "unknown"
    inv_service = InvestigationService(db)
    result = inv_service.create_investigation(
        alert_id=alert_id,
        user_id=actor_user,
        status=investigation.status,
        notes=investigation.notes,
    )

    AuditService(db).log_action(
        user_id=actor_user,
        action="create_investigation",
        target=f"alert:{alert_id}",
    )
    return result


@app.get("/api/alerts/{alert_id}/investigations", response_model=List[InvestigationResponse])
def get_investigations(
    alert_id: int,
    request: Request,
    _current_user: TokenData = Depends(require_role("viewer")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="get_alert_investigations", limit=120, window_seconds=60)
    service = AlertService(db)
    alert = service.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return InvestigationService(db).get_investigations(alert_id=alert_id)


@app.put("/api/investigations/{investigation_id}", response_model=InvestigationResponse)
def update_investigation(
    investigation_id: int,
    update: InvestigationUpdate,
    request: Request,
    current_user: TokenData = Depends(require_role("analyst")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="update_investigation", limit=20, window_seconds=60)
    service = InvestigationService(db)
    result = service.update_investigation(
        investigation_id=investigation_id,
        status=update.status,
        notes=update.notes,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Investigation not found")

    AuditService(db).log_action(
        user_id=current_user.username or "unknown",
        action="update_investigation",
        target=f"investigation:{investigation_id}",
    )
    return result


@app.get("/api/audit/logs", response_model=List[AuditLogResponse])
def get_audit_logs(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _current_user: TokenData = Depends(require_role("viewer")),
    safe_user_id: Optional[str] = Depends(get_safe_audit_user_id),
    safe_action: Optional[str] = Depends(get_safe_audit_action),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="get_audit_logs", limit=60, window_seconds=60)
    service = AuditService(db)
    return service.get_audit_logs(user_id=safe_user_id, action=safe_action, limit=limit, offset=offset)


@app.post("/api/auth/login", response_model=Token)
def login(request: LoginRequest, http_request: Request):
    enforce_rate_limit(http_request, scope="auth_login", limit=5, window_seconds=60)
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "roles": user.roles},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me")
def get_current_user_info(
    request: Request,
    current_user: TokenData = Depends(get_current_user),
):
    enforce_rate_limit(request, scope="auth_me", limit=60, window_seconds=60)
    return {"username": current_user.username, "roles": current_user.roles}


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics(db: Session = Depends(get_db)):
    try:
        alert_service = AlertService(db)
        total_alerts.set(alert_service.count_alerts())
        active_alerts.set(alert_service.count_alerts(triggered=True))
    except Exception:
        pass

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/models", response_model=List[ModelResponse])
def get_models(
    request: Request,
    active: Optional[bool] = Query(None),
    _current_user: TokenData = Depends(require_role("viewer")),
    safe_model_type: Optional[str] = Depends(get_safe_model_type),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="get_models", limit=60, window_seconds=60)
    models = ModelService(db).get_models(model_type=safe_model_type)
    if active is not None:
        models = [model for model in models if model.active == active]
    return models


@app.get("/api/models/active", response_model=ModelResponse)
def get_active_model(
    request: Request,
    _current_user: TokenData = Depends(require_role("viewer")),
    safe_model_type: Optional[str] = Depends(get_safe_model_type),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="get_active_model", limit=60, window_seconds=60)
    service = ModelService(db)

    if safe_model_type:
        model = service.get_active_model(safe_model_type)
    else:
        model = (
            service.get_active_model("structured_baseline")
            or service.get_active_model("distilbert")
            or service.get_active_model("tfidf_fallback")
        )

    if not model:
        raise HTTPException(status_code=404, detail="No active model found")
    return model


@app.get("/api/investigations", response_model=List[InvestigationResponse])
def list_investigations(
    request: Request,
    status: Annotated[Optional[Literal["open", "investigating", "resolved", "false_positive"]], Query()] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _current_user: TokenData = Depends(require_role("viewer")),
    db: Session = Depends(get_db),
):
    enforce_rate_limit(request, scope="list_investigations", limit=60, window_seconds=60)
    investigations = InvestigationService(db).get_investigations()
    if status:
        investigations = [inv for inv in investigations if inv.status == status]
    return investigations[offset:offset + limit]


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
