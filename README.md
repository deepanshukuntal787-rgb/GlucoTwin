# GlucoTwin: Probabilistic Physiological Digital Twin

[![API Tests](https://img.shields.io/badge/API%20Tests-Passing-brightgreen.svg)](#testing) 
[![AUROC](https://img.shields.io/badge/AUROC-0.9978-blue.svg)](#machine-learning-model) 
[![Forecasting RMSE](https://img.shields.io/badge/RMSE%20%40%2060m-12.54%20mg%2FdL-teal.svg)](#machine-learning-model) 
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/) 
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/) 
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**GlucoTwin** is an end-to-end clinical decision support platform and probabilistic physiological digital twin designed for Type 2 Diabetes (T2D) management. It combines continuous glucose monitoring (CGM) telemetry with electronic health record (EHR) phenotypes to model patient-specific glucose dynamics, forecast future trajectories, identify potential hyperglycemic events, and simulate hypothetical lifestyle interventions.

The platform provides an interactive clinical dashboard, a FastAPI backend, a multimodal deep learning model, explainability components, and real-time telemetry streaming.

> **Project status:** Digital Twin Challenge 2026 submission. GlucoTwin is a research-oriented proof of concept intended for demonstration and experimentation, not a clinically validated medical device.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Digital Twin Challenge 2026](#digital-twin-challenge-2026)
- [Healthcare Use Case](#healthcare-use-case)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Machine Learning Model](#machine-learning-model)
- [Dataset and Privacy](#dataset-and-privacy)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Configuration](#configuration)
- [REST and WebSocket API](#rest-and-websocket-api)
- [What-If Simulation Example](#what-if-simulation-example)
- [Testing](#testing)
- [Model Evaluation](#model-evaluation)
- [Clinical Safety and Limitations](#clinical-safety-and-limitations)
- [Team](#team)
- [Submission Assets](#submission-assets)
- [License](#license)

---

## Overview

Traditional diabetes management often depends on reviewing historical glucose readings and responding after undesirable glucose excursions have occurred. GlucoTwin explores a more proactive approach by creating a computational representation of a patient's glucose dynamics.

The system integrates static clinical characteristics with temporal glucose observations to generate probabilistic forecasts and hypothetical intervention scenarios.

Its core capabilities include:

1. **Probabilistic glucose forecasting:** Predicts glucose trajectories over a two-hour horizon using the 10th, 50th, and 90th percentiles.
2. **Early hyperglycemia risk estimation:** Estimates the probability of glucose exceeding 180 mg/dL and identifies potentially high-risk trajectories.
3. **Counterfactual intervention simulation:** Explores hypothetical changes in carbohydrate intake, walking duration, and stress level.
4. **Physiological explainability:** Provides driver attributions to help interpret model predictions.
5. **Real-time telemetry:** Streams simulated CGM readings over persistent WebSocket connections.
6. **Interactive clinical dashboard:** Presents patient profiles, glucose histories, forecasts, risk indicators, and simulation results.

## Key Features

### 1. Probabilistic Two-Hour Forecasting

GlucoTwin generates three forecast quantiles:

- **P10:** Lower predictive quantile.
- **P50:** Median predicted trajectory.
- **P90:** Upper predictive quantile.

Forecasts are generated at 15-minute intervals over a two-hour horizon.

The quantile head is designed to maintain non-crossing predictive intervals, helping preserve the ordering:

`P10 <= P50 <= P90`

These forecasts represent model uncertainty and should not be interpreted as guaranteed bounds on future glucose measurements.

### 2. Early Hyperglycemia Spike Alerting

The risk classifier estimates the likelihood of a future glucose excursion above 180 mg/dL.

The reported benchmark metrics include:

- AUROC: 0.9978
- AUPRC: 0.8544
- Expected Calibration Error: 0.0009

The intended use is to support earlier review of potentially concerning glucose trajectories. The reported lead time of up to 45 minutes is a model target and should be independently validated under realistic deployment conditions.

### 3. Counterfactual What-If Simulation

Clinicians can explore hypothetical scenarios involving:

- Planned carbohydrate consumption.
- Walking duration.
- Stress level.

The simulator estimates how the predicted glucose trajectory may differ under the selected scenario.

These outputs are model-based counterfactual estimates, not proof of how a real patient will respond to an intervention.

### 4. Explainable Physiological Drivers

GlucoTwin provides driver attributions associated with factors such as:

- Carbohydrate intake and meal-related effects.
- Circadian patterns.
- Baseline insulin resistance and insulin sensitivity.
- Physical activity and exertion.

These explanations are intended to help users inspect model behavior. They do not establish causal relationships or replace clinical judgment.

### 5. Real-Time Telemetry Streaming

The backend supports WebSocket-based replay of simulated 15-minute sensor ticks.

The dashboard can use the stream to display evolving glucose measurements and update the patient's modeled state.

### 6. Interactive Clinical Dashboard

The frontend provides a unified interface for exploring:

- Patient cohorts and clinical profiles.
- Current glucose measurements and trends.
- Historical CGM observations.
- Probabilistic forecasts and risk estimates.
- Physiological driver explanations.
- Counterfactual simulation results.

---

## Digital Twin Challenge 2026

### Problem Statement

Predicting adverse clinical events requires continuous, multimodal modeling of patient physiology. However, healthcare data privacy requirements and restricted access to representative clinical datasets make the development of such systems challenging.

Clinicians also need tools that can visualize potential future physiological trajectories and explore hypothetical interventions before making treatment decisions.

GlucoTwin addresses this challenge through a research-oriented physiological digital twin that combines anonymized or synthetic patient characteristics with simulated glucose telemetry to demonstrate predictive risk assessment and interactive scenario analysis.

### Proposed Solution

The platform creates a computational representation of a virtual patient using:

- Static clinical and demographic features.
- Longitudinal glucose observations.
- Temporal context and future covariates.
- Probabilistic forecasting.
- Event-risk estimation.
- Counterfactual scenario simulation.

The result is an interactive environment for investigating how glucose trajectories might evolve and how hypothetical lifestyle changes could affect model predictions.

### Challenge Objectives

- Demonstrate a localized, patient-level physiological prediction use case.
- Support proactive identification of potentially adverse glucose trajectories.
- Explore multimodal integration of clinical features and time-series telemetry.
- Provide interpretable outputs through probabilistic forecasts and driver attributions.
- Demonstrate privacy-conscious development using synthetic and appropriately anonymized data.

---

## Healthcare Use Case

**Target condition:** Type 2 Diabetes (T2D)

**Primary outcome:** Forecasting glucose trajectories and estimating the risk of hyperglycemic excursions.

| Clinical application | GlucoTwin capability |
|---|---|
| Proactive glycemic monitoring | Two-hour probabilistic glucose forecasts |
| Hyperglycemia risk assessment | Risk estimates for glucose exceeding 180 mg/dL |
| Lifestyle scenario exploration | Counterfactual meal, activity, and stress simulations |
| Prediction transparency | Physiological driver attributions |
| Continuous monitoring demonstration | WebSocket-based telemetry replay |
| Clinical review support | Patient profiles, history, forecasts, and risk visualization |

GlucoTwin is designed to demonstrate how predictive modeling could support clinician review. It does not autonomously diagnose diabetes, prescribe medication, or determine insulin dosing.

---

## System Architecture

The platform follows a modular architecture separating the user interface, API layer, model inference components, and persistent storage.

```text
                    ┌────────────────────────────────┐
                    │       React / Vite UI          │
                    │   Clinical Twin Dashboard      │
                    │           :5173                │
                    └───────────────┬────────────────┘
                                    │
                              HTTP / WebSocket
                                    │
                    ┌───────────────▼────────────────┐
                    │       FastAPI Backend          │
                    │             :8000              │
                    ├────────────────────────────────┤
                    │ Routers and API Services       │
                    │                                │
                    │ /patients                      │
                    │ /predict                       │
                    │ /explain                       │
                    │ /whatif                        │
                    │ /stream                        │
                    └────────┬─────────────┬─────────┘
                             │             │
                ┌────────────▼─────┐  ┌────▼──────────────┐
                │ Model Registry   │  │ SQLite Database   │
                │                  │  │                  │
                │ FusionTwinNet    │  │ Patient Profiles │
                │ Isotonic         │  │ CGM Observations │
                │ Calibrator       │  │ Clinical History │
                │ WhatIfSimulator  │  │                  │
                │ TwinExplainer    │  │ WAL Mode         │
                └──────────────────┘  └───────────────────┘
```

### Architecture Components

**Frontend**
- React and Vite.
- Clinical dashboard and visualization components.
- HTTP requests for API operations.
- WebSocket connection for telemetry streaming.

**Backend**
- FastAPI application.
- REST endpoints for patient data, forecasts, explanations, and simulations.
- WebSocket endpoint for streaming.
- Model registry for inference and model access.

**Machine Learning**
- Static EHR feature encoder.
- Temporal convolution and recurrent sequence encoder.
- Multimodal fusion layer.
- Probabilistic quantile forecasting head.
- Hyperglycemia event-risk classifier.
- Isotonic calibration component.
- Explainability and counterfactual simulation components.

**Storage**
- SQLite database.
- Write-Ahead Logging (WAL) mode.
- Patient profiles and glucose observations.

The architecture diagram is illustrative. Refer to the actual repository configuration for the authoritative service definitions, ports, and deployment settings.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Backend API | FastAPI |
| ASGI server | Uvicorn |
| Machine learning | PyTorch |
| Data processing | Python |
| Model calibration | Isotonic regression |
| Database | SQLite |
| Database concurrency | SQLite WAL mode |
| API communication | HTTP / REST |
| Streaming | WebSockets |
| Containerization | Docker, Docker Compose |
| Web serving / reverse proxy | Nginx |
| Testing | Automated API integration tests |

---

## Machine Learning Model

### FusionTwinNet

GlucoTwin uses a multimodal architecture named **FusionTwinNet**, designed to combine static clinical features with longitudinal glucose telemetry.

Reported architecture specifications:

- Approximately 291,140 parameters.
- Static EHR MLP encoder with 21 clinical features.
- Temporal convolutional stem.
- Two-layer bidirectional GRU for 16 dynamic channels.
- Multimodal late-fusion layer.
- Future-covariate decoder GRU with eight future steps and cyclic time-of-day features.
- Monotonic, non-crossing quantile prediction head.
- Hyperglycemia event-risk classifier.

### Model Pipeline

1. Load patient clinical features and recent glucose observations.
2. Encode static clinical characteristics.
3. Extract temporal patterns from the glucose history.
4. Fuse static and dynamic representations.
5. Generate future glucose quantiles.
6. Estimate the probability of a future hyperglycemic excursion.
7. Apply the configured calibration component.
8. Return predictions and associated explanatory outputs through the API.

### Forecasting Outputs

The model produces:

- P10 glucose forecast.
- P50 glucose forecast.
- P90 glucose forecast.
- Hyperglycemia event-risk estimate.
- Physiological driver attributions.
- Counterfactual scenario predictions.

The two-hour forecast is represented at eight 15-minute future steps.

### Calibration

An isotonic regression component is used to calibrate event-risk probabilities. Calibration quality depends on the representativeness and independence of the calibration dataset.

### Validation Split

The reported patient-level partition is:

| Partition | Patients |
|---|---:|
| Training | 70 |
| Validation | 15 |
| Testing | 15 |
| **Total** | **100** |

Patient-level separation is intended to reduce leakage between the training, validation, and test cohorts.

---

## Dataset and Privacy

GlucoTwin is designed to operate within the Digital Twin Challenge sandbox using synthetic or appropriately anonymized data.

### Clinical Features

The EHR-inspired component represents static patient characteristics and clinical variables similar in purpose to those found in structured healthcare records.

The development approach references:

- Synthetic patient generation inspired by Synthea.
- Anonymized clinical data structures inspired by MIMIC-IV.
- Simulated clinical profiles for demonstration and testing.

### CGM Telemetry

The time-series component uses synthetic glucose readings designed to reproduce selected characteristics of continuous glucose monitoring data.

The system can replay readings at 15-minute intervals and combine historical observations with static clinical features for prediction.

### Privacy Principles

- Avoid exposing directly identifiable patient information.
- Use synthetic or appropriately anonymized data in demonstrations.
- Restrict access to any permitted sensitive data.
- Document dataset provenance and preprocessing.
- Apply appropriate access controls and retention policies before real-world deployment.

**Important:** Using synthetic data does not automatically establish legal compliance. Actual DPDP Act, HIPAA, institutional, and competition-sandbox compliance depends on the data sources, processing, deployment, and applicable requirements.

---

## Project Structure

The following is a conceptual overview of the main components. Actual paths may differ depending on the repository version.

```text
GlucoTwin/
├── gluco_twin/
│   └── service/
│       └── main.py
├── web/
│   ├── package.json
│   └── src/
├── models/
│   └── current/
├── data/
├── docs/
│   ├── GlucoTwin Architecture Diagram.pdf
│   └── GlucoTwin Project Presentation.pdf
├── tests/
├── requirements.txt
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

Check the actual repository before assuming that every example file or directory above is present.

---

## Prerequisites

Install the following before running GlucoTwin locally:

- Python 3.10 or later; Python 3.12 is recommended.
- Node.js 18 or later.
- npm.
- Git.
- Docker and Docker Compose for containerized deployment.

Depending on the configured model and installed dependencies, additional system libraries or model files may be required.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ishanbhardwaj17/GlucoTwin.git
cd GlucoTwin
```

### 2. Set Up the Backend

Create and activate a virtual environment.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If the project uses a `.env` file, create it from the supplied example and configure the required values.

```bash
cp .env.example .env
```

Run API integration tests:

```bash
make test-api
```

Start the backend development server:

```bash
make serve
```

The backend is expected to be available at:

- API base URL: http://localhost:8000
- Swagger documentation: http://localhost:8000/docs
- ReDoc documentation: http://localhost:8000/redoc

Verify the health endpoint:

```bash
curl http://localhost:8000/health
```

### 3. Set Up the Frontend

Open a second terminal and navigate to the frontend directory:

```bash
cd web
npm install
npm run dev
```

The Vite development dashboard is expected at:

http://localhost:5173

Ensure that the frontend API configuration points to the correct backend URL and that CORS settings permit the development origin.

---

## Docker Deployment

Docker Compose is the recommended deployment option when the repository's container configuration is available.

### Start the Application

From the repository root:

```bash
docker compose up -d --build
```

Check container status:

```bash
docker compose ps
```

Inspect logs:

```bash
docker compose logs -f
```

Verify the backend:

```bash
curl http://localhost:8000/health
```

### Access the Application

The documented deployment configuration is intended to expose:

- Web dashboard: http://localhost:3000
- Backend API: http://localhost:8000
- Swagger documentation: http://localhost:8000/docs

The frontend development server uses port 5173, while the containerized dashboard is expected to use port 3000 through the Nginx configuration.

Actual port mappings depend on the repository's Docker Compose configuration.

### Stop the Application

```bash
docker compose down
```

To remove the Compose-managed volumes as well, inspect the configuration before running:

```bash
docker compose down -v
```

**Warning:** Removing volumes can delete persisted database data.

---

## Production Deployment

For deployments that support the provided Makefile target:

```bash
make serve-prod
```

Alternatively, start Uvicorn directly:

```bash
uvicorn gluco_twin.service.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

Before using multiple workers, verify that the model initialization, database connections, and application state are safe for multiple processes.

For production environments:

- Use a production-grade database configuration appropriate to the workload.
- Configure explicit CORS origins.
- Protect administrative and patient-data endpoints.
- Use HTTPS at the reverse proxy.
- Configure persistent storage and backups.
- Store secrets outside version control.
- Monitor API latency, failures, and model behavior.
- Validate the model on representative external data before clinical use.

---

## Configuration

GlucoTwin supports configuration through environment variables or a `.env` file, subject to the application's settings implementation.

| Variable | Documented default | Description |
|---|---|---|
| `ENV` | `production` | Application environment |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Backend port |
| `WORKERS` | `1` | Uvicorn worker count |
| `DATABASE_URL` | `sqlite:///data/glucotwin.db` | Database connection URL |
| `MODEL_DIR` | `models/current` | Directory containing model weights and calibration artifacts |
| `FORCE_CPU` | `false` | Force CPU inference when supported |
| `CORS_ORIGINS` | `*` | Allowed frontend origins |
| `LOG_LEVEL` | `INFO` | Application logging level |

Example `.env` configuration:

```dotenv
ENV=development
HOST=0.0.0.0
PORT=8000
WORKERS=1
DATABASE_URL=sqlite:///data/glucotwin.db
MODEL_DIR=models/current
FORCE_CPU=true
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
LOG_LEVEL=INFO
```

These values are examples based on the documented configuration. Confirm the supported environment variables in the actual application before deploying. Avoid wildcard CORS origins in production when specific origins can be configured.

---

## REST and WebSocket API

Interactive API documentation is available through Swagger UI at `/docs` and ReDoc at `/redoc`.

The following endpoints describe the intended API surface. Exact prefixes, response schemas, and supported parameters should be verified against the running application's OpenAPI specification.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check and service/model status |
| `GET` | `/api/v1/patients` | Patient cohort and current glucose trends |
| `GET` | `/api/v1/patients/{id}` | Patient profile, clinical features, and history |
| `GET` | `/api/v1/patients/{id}/predict` | Two-hour probabilistic forecast and risk estimate |
| `GET` | `/api/v1/patients/{id}/history` | Recent CGM observations |
| `GET` | `/api/v1/patients/{id}/explain` | Physiological driver attributions |
| `POST` | `/api/v1/patients/{id}/whatif` | Counterfactual intervention simulation |
| `GET` | `/api/v1/model/info` | Model metadata and reported evaluation metrics |
| `WS` | `/api/v1/patients/{id}/stream` | WebSocket telemetry replay |

### Example: Retrieve Patient Data

```bash
curl http://localhost:8000/api/v1/patients
```

### Example: Retrieve a Glucose Forecast

```bash
curl http://localhost:8000/api/v1/patients/P001/predict
```

### Example: Retrieve Model Information

```bash
curl http://localhost:8000/api/v1/model/info
```

Replace `P001` with a patient identifier returned by the running application.

---

## What-If Simulation Example

GlucoTwin supports hypothetical scenario analysis through its what-if endpoint.

### Request

```bash
curl -X POST \
  http://localhost:8000/api/v1/patients/P001/whatif \
  -H "Content-Type: application/json" \
  -d '{
    "carbs_g": 60.0,
    "walk_duration_min": 20.0,
    "stress_level": "none"
  }'
```

### Parameters

| Parameter | Example | Description |
|---|---|---|
| `carbs_g` | `60.0` | Hypothetical carbohydrate intake in grams |
| `walk_duration_min` | `20.0` | Hypothetical walking duration in minutes |
| `stress_level` | `"none"` | Scenario stress category |

The parameter names and accepted values must match the backend request schema.

### Expected Behavior

The application processes the scenario and returns the supported simulation outputs, which may include a predicted glucose trajectory and associated risk estimates.

The simulation should be interpreted as a computational experiment, not a reliable guarantee of a real patient's response.

---

## Testing

Run the automated API integration suite using the repository's Makefile target:

```bash
make test-api
```

If the target is unavailable in the checked-out version, inspect the Makefile and the test configuration for the supported command.

Additional verification can include:

```bash
curl http://localhost:8000/health
```

```bash
docker compose ps
```

```bash
docker compose logs --tail=100
```

Recommended testing areas include:

- API request validation.
- Patient and observation retrieval.
- Forecast response schemas.
- Quantile ordering.
- Risk probability bounds.
- Calibration behavior.
- Counterfactual input validation.
- WebSocket connection and disconnection.
- Model-loading failures.
- Database persistence and concurrency.

A passing test suite confirms only the behavior covered by those tests; it does not establish clinical safety or model validity.

---

## Model Evaluation

The following figures are the **reported project benchmark results** and should be interpreted in the context of the dataset, evaluation protocol, and implementation used to produce them.

| Metric | Reported result | Interpretation |
|---|---:|---|
| AUROC | 0.9978 | Discrimination for the evaluated event-risk classification task |
| AUPRC | 0.8544 | Precision-recall performance for the evaluated event-risk task |
| RMSE at 60 minutes | 12.54 mg/dL | Root mean squared glucose prediction error |
| MAE at 60 minutes | 9.81 mg/dL | Mean absolute glucose prediction error |
| Expected Calibration Error | 0.0009 | Reported probability calibration error |
| Carbohydrate intervention monotonicity | 96% | Reported compliance with the project's intervention consistency checks |
| Step intervention monotonicity | 95% | Reported compliance with the project's activity consistency checks |

### Interpreting the Results

- **AUROC** measures how well the classifier ranks positive events above negative events.
- **AUPRC** summarizes the precision-recall trade-off and is particularly relevant when event prevalence is low.
- **RMSE** penalizes larger prediction errors more heavily.
- **MAE** reports the average absolute prediction error.
- **Expected Calibration Error** summarizes the discrepancy between predicted probabilities and observed event frequencies under the evaluation's binning procedure.
- **Intervention monotonicity** describes compliance with predefined consistency checks; it does not prove that the simulated physiological response is medically correct.

The reported values should be reproduced from the actual evaluation pipeline before being treated as independently verified results. In particular, very high classification performance should prompt checks for temporal leakage, patient overlap, label construction artifacts, and differences between synthetic and real-world distributions.

---

## Clinical Safety and Limitations

GlucoTwin is a research prototype and proof of concept.

- It is not a substitute for professional medical advice, diagnosis, or treatment.
- Predictions are not guaranteed to match future glucose measurements.
- Counterfactual scenarios do not establish the causal effect of an intervention.
- Synthetic data may not represent the full variability of real-world patients.
- Performance metrics from internal test data may not generalize to external populations.
- The platform must not be used to autonomously prescribe medication or adjust insulin doses.
- Real-world clinical use would require rigorous external validation, appropriate regulatory assessment, privacy and security review, and clinical oversight.

**Intended use:** Research, demonstration, and exploration of probabilistic physiological digital twins for diabetes management.

---

## Team

**Digital Twin Challenge 2026**

| Role | Name |
|---|---|
| Team Leader | Deepanshu Kuntal |
| Team Member | Lakshya Pant |
| Team Member | Ishan Bhardwaj |

**Institution:** Manipal Institute of Technology

**Team GitHub identity:** `deepanshukuntal787`

**Project repository:** [GlucoTwin](https://github.com/ishanbhardwaj17/GlucoTwin)

---

## Submission Assets

### Demo Video

15–20 minute project demonstration:
- **Master Video Artifact**: [`glucotwin_demo_presentation.mp4`](glucotwin_demo_presentation.mp4) (1080p Full HD, 17:21 runtime, 32 MB)
- **Synchronized Subtitles (.srt)**: [`docs/glucotwin_demo_subtitles.srt`](docs/glucotwin_demo_subtitles.srt)
- **Word-for-Word Script & Timestamps**: [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md)
- **YouTube Stream**: `[Insert unlisted YouTube demo link]`

### Architecture Diagram

[View the GlucoTwin Architecture Diagram](docs/GlucoTwin%20Architecture%20Diagram.pdf)

### Project Presentation

[View the GlucoTwin Project Presentation](docs/GlucoTwin%20Project%20Presentation.pdf)

Ensure that these files exist at the specified repository paths and that the demo URL is replaced with the actual video link before submitting the project.

---

## License

This project is intended to be distributed under the **Apache License 2.0**.

See the [`LICENSE`](LICENSE) file for the complete license terms. Ensure that the repository contains the appropriate license file before claiming that the project is formally licensed under Apache-2.0.

---

## Acknowledgments

GlucoTwin brings together ideas from physiological digital twins, probabilistic time-series forecasting, clinical decision support, multimodal machine learning, and privacy-conscious synthetic data generation.

The project demonstrates how these techniques can be integrated into an interactive research platform for exploring predictive and proactive approaches to Type 2 Diabetes management.

**GlucoTwin — From reactive glucose monitoring to predictive physiological modeling.**
