"""Pydantic schemas for the GlucoTwin REST API matching the frontend TypeScript interfaces."""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

# ─── Enums & Literals ────────────────────────────────────────────────────────

Severity = Literal["low", "moderate", "high"]
TwinStatus = Literal["not_personalized", "collecting", "personalized", "stale"]
Trend = Literal["rising", "flat", "falling"]
GlucoseStatus = Literal["low", "normal", "high", "critical"]

# ─── Patient Vitals & Labs ───────────────────────────────────────────────────

class PatientVitals(BaseModel):
    weight_kg: float
    height_cm: float
    bmi: float
    sbp_mmhg: int
    dbp_mmhg: int
    heart_rate_bpm: int


class PatientLabs(BaseModel):
    hba1c_pct: float
    fasting_glucose_mgdl: float
    egfr_ml_min: float
    ldl_mgdl: float
    hdl_mgdl: float
    triglycerides_mgdl: float
    last_updated: str


class DataCompleteness(BaseModel):
    cgm_pct: float
    hr_pct: float
    sleep_pct: float
    activity_pct: float
    overall_pct: float


class PatientSummary(BaseModel):
    id: str
    name: str
    mrn: Optional[str] = None
    age: int
    sex: Literal["M", "F"]
    diabetes_type: str
    risk_severity: Severity
    twin_status: TwinStatus
    current_glucose_mgdl: float
    trend: Trend
    spike_probability: float
    estimated_minutes_to_event: Optional[int] = None
    last_reading_at: str
    confidence_score: float = 0.95


class TargetRange(BaseModel):
    low: float
    high: float


class PatientDetail(PatientSummary):
    vitals: PatientVitals
    labs: PatientLabs
    active_conditions: List[str]
    current_medications: List[str]
    data_completeness: DataCompleteness
    clinician: str
    diagnose_date: str
    target_range: TargetRange


# ─── Forecast & Prediction ───────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    offset_min: int
    p10: float
    p50: float
    p90: float


class EventRisk(BaseModel):
    probability: float
    severity: Severity
    estimated_minutes_to_event: Optional[int] = None
    peak_predicted_mgdl: float


class DataQuality(BaseModel):
    cgm_coverage_pct: float = 98.5
    hr_coverage_pct: float = 97.2
    sensor_dropout: bool = False
    imputed_points: int = 0


class ModelMetadata(BaseModel):
    model_version: str
    retrained_at: str
    auroc: float
    mae_mgdl: float
    mard_pct: float
    lead_time_avg_min: float
    accuracy_within_30: float


class PredictResponse(BaseModel):
    patient_id: str
    generated_at: str
    forecast: List[ForecastPoint]
    event_risk: EventRisk
    data_quality: DataQuality
    model_metadata: ModelMetadata


# ─── CGM History Chart ───────────────────────────────────────────────────────

class CGMDataPoint(BaseModel):
    time: str
    timestamp: int
    glucose: Optional[float] = None
    p50: Optional[float] = None
    p10: Optional[float] = None
    p90: Optional[float] = None
    whatIf: Optional[float] = None
    isMeal: Optional[bool] = False
    isInsulin: Optional[bool] = False


# ─── What-If Scenario ────────────────────────────────────────────────────────

class WhatIfRequest(BaseModel):
    patient_id: Optional[str] = None
    carbs_g: float = Field(default=0.0, ge=0.0, le=200.0)
    walk_duration_min: float = Field(default=0.0, ge=0.0, le=120.0)
    stress_level: Literal["none", "mild", "high"] = "none"


class ScenarioStats(BaseModel):
    peak_mgdl: float
    spike_probability: float
    minutes_above_180: int


class WhatIfResponse(BaseModel):
    baseline: ScenarioStats
    scenario: ScenarioStats
    delta: ScenarioStats
    forecast_scenario: List[ForecastPoint]


# ─── Clinical Explainability ─────────────────────────────────────────────────

class FactorAttribution(BaseModel):
    feature: str
    share_pct: float
    direction: Literal["raises", "lowers"]
    plain_text: str


class ExplainResponse(BaseModel):
    patient_id: str
    attributions: List[FactorAttribution]
    summary: str
    confidence: float


# ─── WebSocket Streaming ─────────────────────────────────────────────────────

class TickMessage(BaseModel):
    type: Literal["tick", "alert", "model_update", "error"]
    timestamp: str
    patient_id: str
    glucose_mgdl: Optional[float] = None
    trend: Optional[Trend] = None
    event_risk: Optional[EventRisk] = None
    forecast: Optional[List[ForecastPoint]] = None
    message: Optional[str] = None


# ─── Health Check ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    patients_count: int
    version: str
