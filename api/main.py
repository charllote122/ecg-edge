"""ECG Edge API."""

from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db, init_db
from api.models import Alert
from api.schemas import (
    AlertCreate, AlertRead, ClassProbabilities,
    HealthResponse, PredictRequest, PredictResponse,
)
from edge.edge_inference import EdgeInference

API_VERSION = "0.1.0"

app = FastAPI(
    title="ECG Edge API",
    version=API_VERSION,
    description="Multi-label ECG arrhythmia classification with ONNX Runtime.",
)

inference_engine: EdgeInference | None = None


@app.on_event("startup")
def startup() -> None:
    global inference_engine
    print("[api] initializing...")
    try:
        inference_engine = EdgeInference()
        print(f"[api] loaded model: {inference_engine.model_path}")
    except FileNotFoundError as e:
        print(f"[api] WARNING: model not loaded - {e}")

    try:
        init_db()
        print("[api] database initialized")
    except Exception as e:
        print(f"[api] WARNING: database not available - {e}")


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "ECG Edge API",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/alerts"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(Alert.__table__.select().limit(1))
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok" if inference_engine is not None and db_ok else "degraded",
        version=API_VERSION,
        model_loaded=inference_engine is not None,
        db_connected=db_ok,
        timestamp=datetime.now(timezone.utc),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    signal = np.array(req.signal, dtype=np.float32)
    if signal.shape == (12, 5000):
        signal = signal.T
    if signal.shape != (5000, 12):
        raise HTTPException(status_code=400, detail=f"Expected (5000, 12), got {signal.shape}")
    result = inference_engine.predict(signal)
    return PredictResponse(
        device_id=req.device_id or "api-client",
        timestamp=datetime.now(timezone.utc),
        predictions=ClassProbabilities(**result["predictions"]),
        detected=result["detected"],
        top_class=result["top_class"],
        top_prob=result["top_prob"],
        latency_ms=result["latency_ms"],
    )


@app.get("/alerts", response_model=list[AlertRead], tags=["alerts"])
def list_alerts(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(Alert).order_by(Alert.timestamp.desc()).limit(limit).all()


@app.post("/alerts", response_model=AlertRead, status_code=201, tags=["alerts"])
def create_alert(alert: AlertCreate, db: Session = Depends(get_db)):
    db_alert = Alert(
        device_id=alert.device_id,
        timestamp=alert.timestamp,
        top_class=alert.top_class,
        top_prob=alert.top_prob,
        detected=",".join(alert.detected),
        latency_ms=alert.latency_ms,
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert
