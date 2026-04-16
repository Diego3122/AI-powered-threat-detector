# AI-Powered Network Threat Detector

A production-grade network intrusion detection system trained on the UNSW-NB15 benchmark dataset. The system streams network flow windows through a real-time ML inference pipeline, persists alerts to PostgreSQL, and surfaces them in a React dashboard with investigation and audit capabilities.

## Architecture

```
UNSW-NB15 Replay
(services/ingest_sim/replay_producer.py)
         |
         v
   Kafka Topic: unsw_nb15.windowed
         |
         v
   Detector Service                 Model Server (port 8001)
   (services/detector/detector.py)  <--- POST /score ---
         |
         v
   Kafka Topic: alerts
         |
         v
   Alert Router
   (services/alert_router.py)
         |
         v POST /api/internal/alerts/ingest
   FastAPI Backend (port 8000)
   (services/api/alerts_api.py)
         |
         v
   PostgreSQL
         |
         v
   React Dashboard (port 3000)
   + Prometheus (port 9090) + Grafana (port 3001)
```

## Quick Start

### Prerequisites

- Docker Desktop (or Docker + Docker Compose v2)
- Python 3.11+ (only needed for local training/eval outside Docker)

### 1. Copy and configure environment

```bash
cp .env.example .env
```

Edit `.env` and replace all `change-me` placeholders. For local development the defaults work as-is; the demo user passwords (`DEMO_ADMIN_PASSWORD`, etc.) must be set to non-empty values.

### 2. Start the full stack

```bash
docker compose up --build
```

This starts:

| Service | URL |
|---|---|
| Dashboard (React + Nginx) | http://localhost:3000 |
| API (FastAPI) | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Model server | http://localhost:8001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

Log in at http://localhost:3000 with the demo credentials set in `.env` (`ENABLE_DEMO_USERS=true` must be set; three roles are available: admin, analyst, viewer).

### 3. Replay UNSW-NB15 traffic

The default stack starts without live event replay. To stream UNSW-NB15 windows through the detector:

```bash
docker compose --profile unsw up unsw-replay
```

Alerts will appear in the dashboard within seconds of the replay starting.

### 4. Stop the stack

```bash
docker compose down
```

To remove persisted volumes:

```bash
docker compose down -v
```

## ML Models

### Structured baseline (active)

Trained on 257,673 UNSW-NB15 records (82K train / 175K holdout). Uses DictVectorizer + LogisticRegression on 58 network flow features.

| Metric | Holdout |
|---|---|
| F1 (attack) | 0.888 |
| ROC-AUC | 0.881 |
| Negative recall (benign pass-through) | 0.510 at trained threshold |
| Operational threshold (`DETECTION_THRESHOLD`) | 0.5 (docker-compose default) |

The trained threshold stored in the bundle (0.2788) is optimized for attack-class F1. The operational threshold is set separately via the `DETECTION_THRESHOLD` environment variable (default: 0.5), which substantially reduces false positives on benign traffic. On retraining, `train_baseline.py` now uses `macro_f1` threshold selection to produce a more balanced bundle threshold.

Model files:
- `models/baseline_unsw.pkl` — active bundle (DictVectorizer + LogisticRegression + TF-IDF text model)
- `models/baseline_unsw.manifest.json` — metrics, quality gate result, label provenance

### DistilBERT (optional)

A fine-tuned DistilBERT model is available in `models/distilbert_unsw/`. The model server falls back to the structured baseline automatically when DistilBERT weights are absent or `MODEL_TYPE` is not set to `distilbert`.

To enable DistilBERT inference, set in `.env`:

```
MODEL_TYPE=distilbert
DISTILBERT_MODEL_PATH=models/distilbert_unsw
```

The `models/distilbert_unsw/` directory contains the full weights (`model.safetensors`, 256 MB). GitHub's 100 MB file limit means this file is excluded from the public repository. Download it separately and place it at that path.

### Training

Retrain the structured baseline:

```bash
python scripts/train_baseline.py \
  --data data/unsw_nb15_train.jsonl \
  --holdout-source unsw_nb15_test \
  --out models/baseline_unsw.pkl
```

The training script enforces a promotion gate: F1 > 0.6, negative recall > 0.4, strong holdout labels required. It will not overwrite the bundle if the gate fails.

Fine-tune DistilBERT on UNSW windows:

```bash
python scripts/train_distilbert.py \
  --data data/unsw_nb15_train.jsonl \
  --out models/distilbert_unsw
```

## Project Structure

```
.
├── .github/
│   └── workflows/ci.yml            # GitHub Actions: pytest, safety check, dashboard build
│
├── alembic/                        # Database migrations
│   └── versions/
│       └── 001_initial_schema.py
│
├── dashboard/                      # React frontend (Vite + TypeScript + Tailwind)
│   └── src/
│       ├── api/client.ts           # Axios client, token management, all API methods
│       ├── components/
│       │   ├── AlertDetail.tsx     # Slide-over panel: flow fields, score, investigation workflow
│       │   ├── IncidentList.tsx
│       │   ├── Layout.tsx
│       │   ├── ModelConfidence.tsx
│       │   └── ThreatTimeline.tsx
│       └── pages/
│           ├── Incidents.tsx       # Alert feed with SSE live updates, click-to-investigate
│           ├── Login.tsx
│           ├── Logs.tsx            # Audit log viewer
│           ├── Models.tsx          # Model registry, metrics, active model
│           └── Overview.tsx        # Threat timeline, model confidence widget
│
├── models/
│   ├── baseline_unsw.pkl           # Active structured baseline bundle
│   ├── baseline_unsw.manifest.json # Metrics, quality gate, label provenance
│   └── distilbert_unsw/            # Fine-tuned DistilBERT weights (not in public repo)
│
├── monitoring/
│   ├── grafana/                    # Grafana provisioning (datasources + dashboards)
│   └── prometheus.yml
│
├── scripts/
│   ├── build_unsw_nb15_dataset.py  # Convert raw UNSW-NB15 CSV to JSONL windows
│   ├── evaluate.py                 # Evaluate a bundle against a dataset
│   ├── evaluate_comparison.py      # Compare two bundles
│   ├── evaluate_credibility.py     # Cross-source credibility audit
│   ├── init_db.py                  # Initialize schema (local Postgres)
│   ├── init_db_docker.py           # Initialize schema + seed model (Docker entrypoint)
│   ├── register_model.py           # Register a bundle in the model registry
│   ├── repo_safety_check.py        # CI: scan for secrets, placeholder credentials
│   ├── train_baseline.py           # Train structured + TF-IDF baseline
│   └── train_distilbert.py         # Fine-tune DistilBERT on UNSW windows
│
├── services/
│   ├── alert_router.py             # Kafka consumer → POST /api/internal/alerts/ingest
│   ├── api/
│   │   ├── alerts_api.py           # FastAPI: alerts, investigations, audit, auth, metrics, SSE
│   │   ├── auth.py                 # JWT, RBAC, demo user provisioning
│   │   └── security.py             # Rate limiting, input normalization, internal API key
│   ├── database/
│   │   ├── db_service.py           # AlertService, InvestigationService, AuditService, ModelService
│   │   └── models.py               # SQLAlchemy ORM: Alert, AlertInvestigation, AuditLog, Model
│   ├── detector/
│   │   └── detector.py             # Kafka/file consumer → model server → alert emit
│   ├── ingest_sim/
│   │   └── replay_producer.py      # Replays UNSW-NB15 JSONL into Kafka
│   ├── ml/
│   │   ├── export_dataset.py       # Kafka/file consumer → labeled JSONL
│   │   └── ml_utils.py             # Feature extraction, threshold selection, metrics
│   └── model_server/
│       ├── app.py                  # FastAPI: /score, /score_batch, /model-info, /metrics
│       └── explainability.py       # SHAP/attention-based feature attribution
│
└── tests/                          # 83 tests across 14 files
    ├── test_api_security.py
    ├── test_detector_pipeline.py
    ├── test_detector_smoke.py
    ├── test_detector_unsw.py
    ├── test_evaluate_comparison.py
    ├── test_evaluation_metric_honesty.py
    ├── test_export_dataset.py
    ├── test_ml_utils.py
    ├── test_model_server_api.py
    ├── test_pipeline_integration.py  # End-to-end: ingest → DB → investigations
    ├── test_repo_safety_check.py
    ├── test_train_baseline.py
    ├── test_unsw_nb15_adapter.py
    └── test_unsw_train_baseline.py
```

## API Reference

### Authentication

All endpoints (except `/health`, `/metrics`, `/api/auth/login`) require a Bearer token.

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'
```

Response:
```json
{"access_token": "<jwt>", "token_type": "bearer"}
```

Three roles exist: `viewer` (read-only), `analyst` (read + create/update investigations), `admin` (full access).

### Alerts

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/api/alerts` | viewer | List alerts. Query params: `triggered`, `model_type`, `limit`, `offset` |
| GET | `/api/alerts/{id}` | viewer | Get single alert |
| GET | `/api/alerts/stream` | viewer | SSE stream of new triggered alerts (use `fetch`, not `EventSource`) |
| PUT | `/api/alerts/{id}` | analyst | Update `triggered` or `explanation_summary` |
| POST | `/api/internal/alerts/ingest` | internal key | Ingest alert from alert router |

### Investigations

| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/api/alerts/{id}/investigations` | analyst | Create investigation |
| GET | `/api/alerts/{id}/investigations` | viewer | List investigations for an alert |
| PUT | `/api/investigations/{id}` | analyst | Update status or notes |
| GET | `/api/investigations` | viewer | List all investigations; filter by `status` |

Investigation statuses: `open`, `investigating`, `resolved`, `false_positive`.

### Models and audit

| Method | Path | Role | Description |
|---|---|---|---|
| GET | `/api/models` | viewer | List registered models |
| GET | `/api/models/active` | viewer | Get active model record |
| GET | `/api/audit/logs` | viewer | Audit log; filter by `user_id`, `action` |

### Live alert stream (SSE)

The dashboard connects to `/api/alerts/stream` using the `fetch` API (not the browser's native `EventSource`, which cannot send custom headers). New triggered alerts are pushed as `data: <json>\n\n` events; a `: keepalive` comment is sent when no new alerts exist. The dashboard falls back to 10-second polling if the connection fails.

```js
fetch('/api/alerts/stream', {
  headers: { Authorization: `Bearer ${token}` }
}).then(async (res) => {
  const reader = res.body.getReader()
  // ... read SSE lines
})
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | — | Database password (required) |
| `POSTGRES_DB` | `threat_detector` | Database name |
| `JWT_SECRET_KEY` | — | Signing key for access tokens (required) |
| `INTERNAL_API_KEY` | — | Key used by alert router to call the ingest endpoint |
| `APP_ENV` | `development` | Set to `production` to disable demo users and enforce secrets |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | CORS allowed origins |
| `MODEL_TYPE` | `structured` | `structured`, `distilbert`, or `tfidf_fallback` |
| `TFIDF_MODEL_PATH` | `models/baseline_unsw.pkl` | Path to the structured/TF-IDF bundle |
| `DISTILBERT_MODEL_PATH` | `models/distilbert_finetuned` | Path to DistilBERT weights directory |
| `DEVICE` | `cpu` | `cpu` or `cuda` for model server inference |
| `DETECTION_THRESHOLD` | `0.5` | Alert threshold passed to the detector; overrides the bundle's trained threshold |
| `KAFKA_BOOTSTRAP` | `127.0.0.1:9092` | Kafka bootstrap address |
| `WINDOW_MS` | `10000` | Window aggregation size in milliseconds |
| `ENABLE_DEMO_USERS` | `true` | Provision demo accounts on startup (development only) |
| `DEMO_ADMIN_PASSWORD` | — | Password for demo `admin` account |
| `DEMO_ANALYST_PASSWORD` | — | Password for demo `analyst` account |
| `DEMO_VIEWER_PASSWORD` | — | Password for demo `viewer` account |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | JWT expiry |
| `GRAFANA_ADMIN_USER` | `admin` | Grafana admin username |
| `GRAFANA_ADMIN_PASSWORD` | — | Grafana admin password |
| `VITE_API_URL` | `http://localhost:8000` | API base URL injected into the dashboard build |

## Database Schema

Schema is managed with Alembic. Migrations run automatically on API container startup via `scripts/init_db_docker.py`.

To apply migrations manually against a local Postgres instance:

```bash
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/threat_detector \
  python scripts/init_db.py
```

To create a new migration after changing `services/database/models.py`:

```bash
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

Tables: `alerts`, `alert_investigations`, `audit_log`, `models`, `performance_metrics`.

## Model Registry

Models are registered with `scripts/register_model.py` and stored in the `models` table. The API `init_db_docker.py` entrypoint registers `models/baseline_unsw.manifest.json` on first boot if no active structured baseline exists.

To manually register a bundle:

```bash
python scripts/register_model.py \
  models/baseline_unsw.manifest.json \
  --activate
```

The active model is reported by `GET /api/models/active` and shown on the Models page in the dashboard.

## Testing

```bash
python -m pytest -q
```

The suite runs 83 tests covering: API security and rate limiting, detector pipeline, model server inference, ML utilities, evaluation metric honesty, dataset export, UNSW-NB15 adapter, training script promotion gates, and end-to-end alert ingestion through the investigation lifecycle.

CI runs on every push and pull request (`.github/workflows/ci.yml`): Python syntax validation, full test suite, repo safety check (scans for committed secrets and placeholder credentials), TypeScript type-check, and dashboard build.

## Security

- JWT authentication with role-based access control (viewer / analyst / admin).
- All user-supplied inputs are normalized, stripped of control characters, and length-bounded before reaching the database.
- Login and write endpoints are rate-limited per IP.
- The alert router authenticates to the API using a shared `INTERNAL_API_KEY` over the internal Docker network; this key is never exposed to the frontend.
- Demo users are provisioned only when `ENABLE_DEMO_USERS=true` and suppressed entirely in production (`APP_ENV=production`). Passwords must be set to non-empty values for demo accounts to be created.
- All containers run as non-root with a read-only root filesystem, no-new-privileges, and dropped Linux capabilities.
- Only `VITE_*` environment variables are embedded in the frontend build. Do not place secrets in `VITE_*` variables.

## Troubleshooting

**No alerts appearing after replay starts**

Check that the model server loaded the correct bundle:

```bash
curl http://localhost:8001/model-info
```

`active_model` should be `structured_baseline`. If it shows `distilbert` and inference is failing, set `MODEL_TYPE=structured` in `.env` and restart the model-server container.

**Login returns 401 with correct credentials**

Demo users are skipped if their password environment variables are empty. Verify `DEMO_ADMIN_PASSWORD` (and analyst/viewer equivalents) are set in `.env` and restart the API container.

**Kafka connection refused**

The Kafka container uses KRaft mode and takes 10–15 seconds to elect a controller on first start. The detector and alert router will retry automatically. If it remains unreachable, check:

```bash
docker compose logs kafka | tail -20
```

**DistilBERT not loading**

The model server falls back to the structured baseline silently when DistilBERT weights are missing. Check the model-server logs for `FileNotFoundError` mentioning the path configured in `DISTILBERT_MODEL_PATH`.

**Database migrations fail on startup**

The API container waits for Postgres to be ready before running migrations (up to 30 retries, 2 seconds apart). If migrations consistently fail, check the database logs:

```bash
docker compose logs postgres
```

## License

MIT — see [LICENSE](LICENSE).
