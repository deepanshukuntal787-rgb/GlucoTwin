# GlucoTwin: Probabilistic Physiological Digital Twin

[![API Tests](https://img.shields.io/badge/API%20Tests-Passing-brightgreen.svg)]()
[![Model Performance](https://img.shields.io/badge/AUROC-0.9978-blue.svg)]()
[![Forecasting RMSE](https://img.shields.io/badge/RMSE%20%40%2060m-12.54%20mg%2FdL-teal.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)]()

GlucoTwin is an end-to-end clinical decision support platform and physiological digital twin for Type 2 Diabetes (T2D) management. It continuously models continuous glucose monitoring (CGM) telemetry and electronic health record (EHR) phenotypes to provide:
1. **Probabilistic 2-Hour Trajectory Forecasting**: Generates non-crossing $P_{10}$, $P_{50}$, and $P_{90}$ predictive intervals every 15 minutes.
2. **Early Hyperglycemia Spike Alerting**: Detects excursions $>180$ mg/dL up to 45 minutes in advance (AUROC 0.9978, AUPRC 0.8544, calibrated via isotonic regression).
3. **Counterfactual What-If Simulation**: Dynamically models physiological responses to planned carbohydrate intake, physical activity, and stress.
4. **Clinical Explainability**: Decomposes predicted trajectories into physiological driver attributions (meal bolus, circadian rhythm, baseline insulin resistance, physical exertion).
5. **Real-time Telemetry Streaming**: Replays 15-minute sensor ticks over persistent WebSocket connections.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │              React / Vite UI                  │
                      │  (Modern Clinical Twin Dashboard on :5173)    │
                      └───────────────────────┬───────────────────────┘
                                              │ HTTP / WebSocket
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                FastAPI Backend (:8000)                                 │
│                                                                                        │
│  ┌───────────────────────┐   ┌───────────────────────────┐   ┌──────────────────────┐  │
│  │     Routers & API     │   │       ModelRegistry       │   │  SQLite WAL Database │  │
│  │  - /patients          │   │  - FusionTwinNet (PyTorch)│   │  - 100 T2D Patients  │  │
│  │  - /patients/{id}     │◄─►│  - Calibrator (Isotonic)  │◄─►│  - 576,000 Readings  │  │
│  │  - /predict           │   │  - WhatIfSimulator        │   │  - High-concurrency  │  │
│  │  - /explain           │   │  - TwinExplainer          │   │    WAL mode          │  │
│  │  - /whatif            │   │  - Pre-warmed singleton   │   └──────────────────────┘  │
│  │  - /stream (WebSocket)│   └───────────────────────────┘                             │
│  └───────────────────────┘                                                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart & Local Execution

### Prerequisites
- Python 3.10+ (Recommended: 3.12)
- Node.js 18+ & npm

### 1. Backend Setup
```bash
# Clone and enter repo
git clone https://github.com/ishanbhardwaj17/GlucoTwin.git
cd GlucoTwin

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run automated API integration tests
make test-api

# Start development ASGI server (runs on http://localhost:8000)
make serve
```

### 2. Frontend Setup
```bash
cd web
npm install
npm run dev
# Dashboard is live at http://localhost:5173
```

---

## Production Deployment

### Option A: One-Command Docker Compose (Recommended)
GlucoTwin includes multi-stage container configurations for both backend (FastAPI) and frontend (Nginx reverse-proxy):

```bash
# Build and launch both containers in detached mode
docker compose up -d

# Verify services
docker compose ps
curl http://localhost:8000/health
```
- **Web Dashboard**: `http://localhost:3000` (proxies `/api/` to backend automatically)
- **Backend API & Swagger Docs**: `http://localhost:8000/docs`

To stop:
```bash
docker compose down
```

### Option B: Bare-Metal / Production ASGI
Run Uvicorn with multi-worker concurrency on Linux/macOS:
```bash
# Using Makefile
make serve-prod

# Or directly with Uvicorn
uvicorn gluco_twin.service.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

---

## Configuration & Environment Variables

All settings can be configured via environment variables or a `.env` file:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENV` | `production` | Environment mode (`development` or `production`) |
| `HOST` | `0.0.0.0` | Bind host address |
| `PORT` | `8000` | Bind port number |
| `WORKERS` | `1` | Uvicorn worker process count |
| `DATABASE_URL` | `sqlite:///data/glucotwin.db` | SQLAlchemy database URL (SQLite or PostgreSQL) |
| `MODEL_DIR` | `models/current` | Directory containing model weights & calibrator |
| `FORCE_CPU` | `false` | Force CPU execution (set `1` in standard containers) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated for specific domains) |
| `LOG_LEVEL` | `INFO` | Application log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## REST & WebSocket API Reference

Interactive OpenAPI documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

### Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck reporting model state, device, and patient counts |
| `GET` | `/api/v1/patients` | Cohort roster with current glycemia, trend, and spike risk |
| `GET` | `/api/v1/patients/{id}` | Detailed clinical profile, vitals, labs, and history |
| `GET` | `/api/v1/patients/{id}/predict` | 2-hour $P_{10}/P_{50}/P_{90}$ trajectory forecast & alert probability |
| `GET` | `/api/v1/patients/{id}/history` | Recent CGM observations formatted for Recharts |
| `GET` | `/api/v1/patients/{id}/explain` | Physiological driver attributions (carbs, circadian, insulin sensitivity) |
| `POST` | `/api/v1/patients/{id}/whatif` | Counterfactual simulation for planned meals, steps, and stress |
| `GET` | `/api/v1/model/info` | Model card specifications, benchmarks, and validation metrics |
| `WS` | `/api/v1/patients/{id}/stream` | WebSocket telemetry stream replaying 15-minute sensor ticks |

### Example What-If Simulation Request
```bash
curl -X POST http://localhost:8000/api/v1/patients/P001/whatif \
  -H "Content-Type: application/json" \
  -d '{
    "carbs_g": 60.0,
    "walk_duration_min": 20.0,
    "stress_level": "none"
  }'
```

---

## Machine Learning Model Specifications

- **Architecture**: `FusionTwinNet` (291,140 parameters)
  - Static EHR MLP Encoder (21 clinical features)
  - Conv1D Temporal Stem + 2-layer Bidirectional GRU (16 dynamic channels)
  - Multimodal Late Fusion layer
  - Future Covariate Decoder GRU (8 future steps with cyclic TOD)
  - Monotonic Non-Crossing Quantile Head ($P_{10}$, $P_{50}$, $P_{90}$)
  - Event Spike Risk Classifier ($>180$ mg/dL)
- **Held-Out Test Cohort Validation**:
  - **AUROC**: `0.9978`
  - **AUPRC**: `0.8544`
  - **RMSE @ 60 min**: `12.54 mg/dL`
  - **MAE @ 60 min**: `9.81 mg/dL`
  - **Expected Calibration Error (ECE)**: `0.0009`
  - **Intervention Monotonicity**: 96% carb compliance, 95% step compliance
- **Zero Leakage**: Strict patient-level holdout partition (70 train, 15 validation, 15 test).

---

## License
Apache-2.0
