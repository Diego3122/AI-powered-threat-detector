from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from services.database.models import Alert, AuditLog, Model, AlertInvestigation, PerformanceMetrics


class AlertService:
    def __init__(self, session: Session):
        self.session = session
    
    def create_alert(
        self,
        timestamp: int,
        window_id: str,
        model_type: str,
        model_score: float,
        threshold: float,
        explanation: str = None,
        triggered: Optional[bool] = None,
        feature_schema: Optional[str] = None,
    ) -> Alert:
        existing = None
        if window_id:
            existing = self.session.query(Alert).filter(
                Alert.window_id == window_id,
                Alert.model_type == model_type,
                Alert.timestamp == timestamp,
            ).first()

        if existing:
            if explanation and not existing.explanation_summary:
                existing.explanation_summary = explanation
                existing.updated_at = datetime.utcnow()
                self.session.commit()
            return existing

        alert = Alert(
            timestamp=timestamp,
            window_id=window_id,
            model_type=model_type,
            model_score=model_score,
            threshold=threshold,
            explanation_summary=explanation,
            triggered=(model_score >= threshold) if triggered is None else triggered,
            feature_schema=feature_schema,
        )
        self.session.add(alert)
        self.session.commit()
        return alert
    
    def get_alert(self, alert_id: int) -> Optional[Alert]:
        return self.session.query(Alert).filter(Alert.id == alert_id).first()
    
    def get_alerts(self, limit: int = 100, offset: int = 0, model_type: str = None, 
                  triggered: bool = None) -> List[Alert]:
        query = self.session.query(Alert)
        
        if model_type:
            query = query.filter(Alert.model_type == model_type)
        if triggered is not None:
            query = query.filter(Alert.triggered == triggered)
        
        return query.order_by(Alert.timestamp.desc()).limit(limit).offset(offset).all()
    
    def get_alerts_by_timerange(self, start_ms: int, end_ms: int) -> List[Alert]:
        return self.session.query(Alert).filter(
            Alert.timestamp >= start_ms,
            Alert.timestamp <= end_ms
        ).order_by(Alert.timestamp.desc()).all()
    
    def update_alert(self, alert_id: int, **kwargs) -> Optional[Alert]:
        alert = self.get_alert(alert_id)
        if alert:
            for key, value in kwargs.items():
                if hasattr(alert, key):
                    setattr(alert, key, value)
            alert.updated_at = datetime.utcnow()
            self.session.commit()
        return alert
    
    def count_alerts(self, model_type: str = None, triggered: bool = None) -> int:
        query = self.session.query(func.count(Alert.id))
        if model_type:
            query = query.filter(Alert.model_type == model_type)
        if triggered is not None:
            query = query.filter(Alert.triggered == triggered)
        return query.scalar() or 0
    
    def get_alerts_stats(self, hours: int = 24) -> dict:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        total = self.session.query(func.count(Alert.id)).filter(
            Alert.created_at >= cutoff
        ).scalar() or 0
        
        by_model = self.session.query(
            Alert.model_type,
            func.count(Alert.id).label('count'),
            func.avg(Alert.model_score).label('avg_score'),
        ).filter(Alert.created_at >= cutoff).group_by(Alert.model_type).all()
        
        return {
            "total_alerts": total,
            "by_model": [
                {
                    "model": m[0],
                    "count": m[1],
                    "avg_score": float(m[2]) if m[2] else 0,
                }
                for m in by_model
            ]
        }


class AuditService:
    def __init__(self, session: Session):
        self.session = session
    
    def log_action(self, user_id: str, action: str, target: str, details: str = None) -> AuditLog:
        audit = AuditLog(
            user_id=user_id,
            action=action,
            target=target,
            details=details,
        )
        self.session.add(audit)
        self.session.commit()
        return audit
    
    def get_audit_logs(self, user_id: str = None, action: str = None, 
                      limit: int = 100, offset: int = 0) -> List[AuditLog]:
        query = self.session.query(AuditLog)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if action:
            query = query.filter(AuditLog.action == action)
        
        return query.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset).all()


class ModelService:
    def __init__(self, session: Session):
        self.session = session
    
    def register_model(self, model_type: str, version: str, accuracy: float, 
                      f1_score: float, roc_auc: float, n_samples: int, 
                      metadata: str = None) -> Model:
        model = Model(
            model_type=model_type,
            version=version,
            accuracy=accuracy,
            f1_score=f1_score,
            roc_auc=roc_auc,
            n_samples=n_samples,
            model_metadata=metadata,
            active=False,
        )
        self.session.add(model)
        self.session.commit()
        return model
    
    def set_active_model(self, model_id: int):
        model = self.session.query(Model).filter(Model.id == model_id).first()
        if model:
            # Deactivate others of same type
            self.session.query(Model).filter(
                Model.model_type == model.model_type,
                Model.id != model_id
            ).update({"active": False})
            
            model.active = True
            self.session.commit()
    
    def get_active_model(self, model_type: str) -> Optional[Model]:
        return self.session.query(Model).filter(
            Model.model_type == model_type,
            Model.active == True
        ).order_by(Model.created_at.desc()).first()
    
    def get_models(self, model_type: str = None) -> List[Model]:
        query = self.session.query(Model)
        if model_type:
            query = query.filter(Model.model_type == model_type)
        return query.order_by(Model.created_at.desc()).all()


class InvestigationService:
    def __init__(self, session: Session):
        self.session = session
    
    def create_investigation(self, alert_id: int, user_id: str, 
                           status: str = "open", notes: str = None) -> AlertInvestigation:
        investigation = AlertInvestigation(
            alert_id=alert_id,
            user_id=user_id,
            status=status,
            notes=notes,
        )
        self.session.add(investigation)
        self.session.commit()
        return investigation
    
    def update_investigation(self, investigation_id: int, status: str = None, 
                           notes: str = None) -> Optional[AlertInvestigation]:
        inv = self.session.query(AlertInvestigation).filter(
            AlertInvestigation.id == investigation_id
        ).first()
        
        if inv:
            if status:
                inv.status = status
            if notes:
                inv.notes = notes
            inv.updated_at = datetime.utcnow()
            self.session.commit()
        
        return inv
    
    def get_investigations(self, alert_id: int = None) -> List[AlertInvestigation]:
        query = self.session.query(AlertInvestigation)
        if alert_id:
            query = query.filter(AlertInvestigation.alert_id == alert_id)
        return query.order_by(AlertInvestigation.created_at.desc()).all()
