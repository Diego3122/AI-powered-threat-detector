# Seed the database with sample records after migrations have been applied.
import json
import os
import random
import sys
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.database.models import Alert, AlertInvestigation, AuditLog, Model, PerformanceMetrics


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/threat_detector",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def seed_models(session):
    models = [
        Model(
            model_type="distilbert",
            version="1.0",
            accuracy=0.95,
            f1_score=0.93,
            roc_auc=0.97,
            active=True,
        ),
        Model(
            model_type="tfidf_fallback",
            version="1.0",
            accuracy=0.87,
            f1_score=0.85,
            roc_auc=0.90,
            active=False,
        ),
    ]
    session.add_all(models)
    session.commit()
    print(f"Created {len(models)} sample models")
    return models


def seed_performance_metrics(session, models):
    now = datetime.utcnow()
    metrics = []

    for model in models:
        for i in range(7):
            date = now - timedelta(days=i)
            hour_ts = int((date.timestamp() * 1000) // 3600000) * 3600000
            metrics.append(
                PerformanceMetrics(
                    hour_timestamp=hour_ts,
                    alert_count=random.randint(5, 50),
                    avg_score=model.accuracy + random.uniform(-0.02, 0.02),
                    model_type=model.model_type,
                    created_at=date,
                )
            )

    session.add_all(metrics)
    session.commit()
    print(f"Created {len(metrics)} performance metric records")


def seed_alerts(session, models):
    now = datetime.utcnow()
    alert_types = [
        "Brute force attack detected",
        "SQL injection attempt",
        "DDoS pattern recognized",
        "Malware signature match",
        "Privilege escalation attempt",
        "Port scanning activity",
        "Unusual data exfiltration",
        "Suspicious login from new location",
    ]

    alerts = []
    for _ in range(20):
        timestamp = now - timedelta(hours=random.randint(0, 72))
        model = random.choice(models)
        score = random.uniform(0.5, 1.0)

        alerts.append(
            Alert(
                timestamp=int(timestamp.timestamp() * 1000),
                model_type=model.model_type,
                model_score=score,
                threshold=0.7,
                triggered=score > 0.7,
                explanation_summary=random.choice(alert_types),
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    session.add_all(alerts)
    session.commit()
    print(f"Created {len(alerts)} sample alerts")
    return alerts


def seed_investigations(session, alerts):
    statuses = ["open", "investigating", "resolved", "false_positive"]
    users = ["analyst_1", "analyst_2", "analyst_3", "admin"]

    investigations = []
    for alert in alerts[:10]:
        investigations.append(
            AlertInvestigation(
                alert_id=alert.id,
                user_id=random.choice(users),
                status=random.choice(statuses),
                notes=f"Investigation for {alert.explanation_summary}. Pattern analysis complete.",
                created_at=alert.created_at,
                updated_at=alert.created_at + timedelta(hours=random.randint(1, 24)),
            )
        )

    session.add_all(investigations)
    session.commit()
    print(f"Created {len(investigations)} sample investigations")


def seed_audit_logs(session):
    now = datetime.utcnow()
    actions = [
        "Model inference completed",
        "Alert triggered",
        "Investigation created",
        "Alert acknowledged",
        "Investigation closed",
        "Model updated",
        "Configuration changed",
    ]

    logs = []
    for _ in range(15):
        timestamp = now - timedelta(hours=random.randint(0, 72))
        action = random.choice(actions)
        logs.append(
            AuditLog(
                user_id=None,
                action=action,
                target=f"alert:{random.randint(1, 20)}" if "Alert" in action else f"model:{random.randint(1, 5)}",
                details=json.dumps({"status": "success", "duration_ms": random.randint(10, 500)}),
                created_at=timestamp,
            )
        )

    session.add_all(logs)
    session.commit()
    print(f"Created {len(logs)} audit logs")


def main():
    print("Seeding database with sample threat detection data")
    session = SessionLocal()
    try:
        models = seed_models(session)
        seed_performance_metrics(session, models)
        alerts = seed_alerts(session, models)
        seed_investigations(session, alerts)
        seed_audit_logs(session)
        print(f"Database seeding complete. Models={len(models)} Alerts={len(alerts)}")
    except Exception as e:
        print(f"Error seeding database: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
