"""Pydantic schemas for the ECG Edge API."""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    signal: list[list[float]] = Field(..., description="ECG signal (5000, 12) or (12, 5000)")
    device_id: str | None = None


class ClassProbabilities(BaseModel):
    NORM: float
    MI: float
    STTC: float
    CD: float
    HYP: float


class PredictResponse(BaseModel):
    device_id: str
    timestamp: datetime
    predictions: ClassProbabilities
    detected: list[str]
    top_class: str
    top_prob: float
    latency_ms: float


class AlertCreate(BaseModel):
    device_id: str
    timestamp: datetime
    top_class: str
    top_prob: float
    detected: list[str]
    latency_ms: float


class AlertRead(BaseModel):
    id: int
    device_id: str
    timestamp: datetime
    top_class: str
    top_prob: float
    detected: str
    latency_ms: float

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    db_connected: bool
    timestamp: datetime
