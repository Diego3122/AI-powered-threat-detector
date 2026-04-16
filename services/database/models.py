from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, BigInteger, ForeignKey, Index
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(BigInteger, nullable=False)  # ms since epoch
    window_id = Column(String(255))
    model_type = Column(String(50))  # e.g. "structured_baseline", "distilbert", "tfidf_fallback"
    feature_schema = Column(String(50))  # e.g. "network_flow_v1", "cloudtrail_v1"
    model_score = Column(Float)
    threshold = Column(Float)
    triggered = Column(Boolean, default=True)
    explanation_summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    investigations = relationship("AlertInvestigation", back_populates="alert", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_alerts_timestamp', 'timestamp'),
        Index('idx_alerts_model_type', 'model_type'),
        Index('idx_alerts_triggered', 'triggered'),
        Index('idx_alerts_feature_schema', 'feature_schema'),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "window_id": self.window_id,
            "model_type": self.model_type,
            "feature_schema": self.feature_schema,
            "model_score": self.model_score,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "created_at": self.created_at.isoformat(),
        }


class AuditLog(Base):
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255))
    action = Column(String(100))  # "view_alert", "investigate", "dismiss", etc.
    target = Column(String(255))  # Resource ID being acted upon
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_audit_log_user_id', 'user_id'),
        Index('idx_audit_log_action', 'action'),
        Index('idx_audit_log_created_at', 'created_at'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "target": self.target,
            "created_at": self.created_at.isoformat(),
        }


class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True)
    model_type = Column(String(50), nullable=False)  # "tfidf_fallback", "distilbert"
    version = Column(String(50))
    accuracy = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    n_samples = Column(Integer)
    active = Column(Boolean, default=False)
    model_metadata = Column(Text)  # JSON metadata (renamed from metadata to avoid SQLAlchemy reserved name)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_models_model_type', 'model_type'),
        Index('idx_models_active', 'active'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "model_type": self.model_type,
            "version": self.version,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "n_samples": self.n_samples,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
        }


class AlertInvestigation(Base):
    __tablename__ = "alert_investigations"
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=False)
    user_id = Column(String(255))
    status = Column(String(50))  # "open", "investigating", "resolved", "false_positive"
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    alert = relationship("Alert", back_populates="investigations")
    
    __table_args__ = (
        Index('idx_alert_investigations_alert_id', 'alert_id'),
        Index('idx_alert_investigations_status', 'status'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "status": self.status,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class PerformanceMetrics(Base):
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True)
    hour_timestamp = Column(BigInteger)  # Hour rounded timestamp
    alert_count = Column(Integer)
    avg_score = Column(Float)
    model_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_performance_metrics_hour_timestamp', 'hour_timestamp'),
        Index('idx_performance_metrics_model_type', 'model_type'),
    )
    
    def to_dict(self):
        return {
            "id": self.id,
            "hour_timestamp": self.hour_timestamp,
            "alert_count": self.alert_count,
            "avg_score": self.avg_score,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
        }


def get_db_url(db_host="localhost", db_port=5432, db_name="threat_detector", db_user="postgres", db_password="postgres"):
    return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


_engine = None


def _get_engine(db_url: str):
    global _engine
    if _engine is None:
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=False,
        )
    return _engine


def get_session(db_url: str):
    Session = sessionmaker(bind=_get_engine(db_url))
    return Session()
