// ─── Severity & Status Enums ───────────────────────────────────────────────

export type Severity = 'low' | 'moderate' | 'high';
export type TwinStatus = 'not_personalized' | 'collecting' | 'personalized' | 'stale';
export type Trend = 'rising' | 'flat' | 'falling';
export type GlucoseStatus = 'low' | 'normal' | 'high' | 'critical';
export type AlertSeverity = 'info' | 'warning' | 'danger' | 'success';

// ─── Patient Types ─────────────────────────────────────────────────────────

export interface PatientVitals {
  weight_kg: number;
  height_cm: number;
  bmi: number;
  sbp_mmhg: number;
  dbp_mmhg: number;
  heart_rate_bpm: number;
}

export interface PatientLabs {
  hba1c_pct: number;
  fasting_glucose_mgdl: number;
  egfr_ml_min: number;
  ldl_mgdl: number;
  hdl_mgdl: number;
  triglycerides_mgdl: number;
  last_updated: string;
}

export interface DataCompleteness {
  cgm_pct: number;
  hr_pct: number;
  sleep_pct: number;
  activity_pct: number;
  overall_pct: number;
}

export interface PatientSummary {
  id: string;
  name: string;
  mrn?: string;
  age: number;
  sex: 'M' | 'F';
  diabetes_type: 'T2D' | 'T1D' | 'Pre-DM';
  risk_severity: Severity;
  twin_status: TwinStatus;
  current_glucose_mgdl: number;
  trend: Trend;
  spike_probability: number;       // 0–1
  estimated_minutes_to_event: number | null;
  last_reading_at: string;
  confidence_score: number;        // 0–1 — low means sensor dropout / missing data
}

export interface PatientDetail extends PatientSummary {
  vitals: PatientVitals;
  labs: PatientLabs;
  active_conditions: string[];
  current_medications: string[];
  data_completeness: DataCompleteness;
  clinician: string;
  diagnose_date: string;
  target_range: { low: number; high: number };
}

// ─── Forecast Types ────────────────────────────────────────────────────────

export interface ForecastPoint {
  offset_min: number;   // minutes from now
  p10: number;          // mg/dL 10th percentile
  p50: number;          // mg/dL median (best estimate)
  p90: number;          // mg/dL 90th percentile
}

export interface EventRisk {
  probability: number;              // 0–1
  severity: Severity;
  estimated_minutes_to_event: number | null;
  peak_predicted_mgdl: number;
}

export interface DataQuality {
  cgm_coverage_pct: number;
  hr_coverage_pct: number;
  sensor_dropout: boolean;
  imputed_points: number;
}

export interface ModelMetadata {
  model_version: string;
  retrained_at: string;
  auroc: number;
  mae_mgdl: number;
  mard_pct: number;
  lead_time_avg_min: number;
  accuracy_within_30: number;
}

export interface PredictResponse {
  patient_id: string;
  generated_at: string;
  forecast: ForecastPoint[];
  event_risk: EventRisk;
  data_quality: DataQuality;
  model_metadata: ModelMetadata;
}

// ─── CGM Chart Data ────────────────────────────────────────────────────────

export interface CGMDataPoint {
  time: string;
  timestamp: number;
  glucose?: number;
  p50?: number;
  p10?: number;
  p90?: number;
  whatIf?: number;
  isMeal?: boolean;
  isInsulin?: boolean;
}

// ─── What-If Types ─────────────────────────────────────────────────────────

export interface WhatIfRequest {
  patient_id: string;
  carbs_g: number;
  walk_duration_min: number;
  stress_level: 'none' | 'mild' | 'high';
}

export interface WhatIfResponse {
  baseline: {
    peak_mgdl: number;
    spike_probability: number;
    minutes_above_180: number;
  };
  scenario: {
    peak_mgdl: number;
    spike_probability: number;
    minutes_above_180: number;
  };
  delta: {
    peak_mgdl: number;
    spike_probability: number;
    minutes_above_180: number;
  };
  forecast_scenario: ForecastPoint[];
}

// ─── Explanation Types ─────────────────────────────────────────────────────

export interface FactorAttribution {
  feature: string;
  share_pct: number;              // 0–100
  direction: 'raises' | 'lowers';
  plain_text: string;
}

export interface ExplainResponse {
  patient_id: string;
  attributions: FactorAttribution[];
  summary: string;
  confidence: number;
}

// ─── WebSocket Streaming ───────────────────────────────────────────────────

export interface TickMessage {
  type: 'tick' | 'alert' | 'model_update' | 'error';
  timestamp: string;
  patient_id: string;
  glucose_mgdl?: number;
  trend?: Trend;
  event_risk?: EventRisk;
  forecast?: ForecastPoint[];
  message?: string;
}

// ─── Stream State ─────────────────────────────────────────────────────────

export interface StreamState {
  isConnected: boolean;
  isPlaying: boolean;
  speed: number;
  simulatedTime: Date;
  currentGlucose: number;
  trend: Trend;
  recentReadings: { timestamp: number; glucose: number }[];
  eventRisk: EventRisk | null;
  forecast: ForecastPoint[];
  lastTick: TickMessage | null;
}
