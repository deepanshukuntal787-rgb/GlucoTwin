# Architecture

**Project:** GlucoTwin (working title)
**Related documents:** `PRD.md` (requirements), `TRD.md` (technical specification), `APP-FLOW.md` (screens and flows), `BACKEND-SCHEMA.md` (data contracts), `IMPLEMENTATION.md` (build guide)
**Version:** 1.0
**Date:** 2026-10-02

All diagrams use Mermaid, which GitHub renders directly in Markdown. Section 14 explains how to export them to PDF or PPT for the submission.

---

## 1. Purpose and scope

This document describes the structure of the system: the parts, how they connect, how data moves, and why the main choices were made. It does not repeat requirement detail from the PRD or tuning detail from the TRD.

**One sentence summary:** synthetic EHR data and wearable time-series are merged into one dataset, a two-branch neural network fuses them to forecast glucose and flag a hyperglycemic event up to 2 hours ahead, and a FastAPI service plus a React dashboard let a doctor inspect, question, and test the virtual patient.

## 2. System context

```mermaid
flowchart LR
    subgraph SRC["Data sources"]
        S1["Synthea synthetic EHR"]
        S2["Open CGM datasets"]
        S3["Own simulator"]
    end
    subgraph OFF["Offline pipeline"]
        P1["Ingest and harmonize"]
        P2["Features and labels"]
        P3["Train, calibrate, evaluate"]
        P4["Model files"]
    end
    subgraph ON["Online service"]
        A1["FastAPI REST"]
        A2["WebSocket replay engine"]
        A3["Inference and explanation"]
        A4[("SQLite demo database")]
    end
    W["React doctor dashboard"]
    U["Doctor or jury member"]

    S1 --> P1
    S2 --> P1
    S3 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> A3
    A4 --> A1
    A4 --> A2
    A1 --> A3
    A2 --> A3
    A1 --> W
    A2 --> W
    W --> U
```

Two worlds exist side by side:

| World | When it runs | What it produces |
|---|---|---|
| Offline | Before the demo, on a laptop | A trained, calibrated model, metrics, and a demo cohort |
| Online | During the demo | Predictions, alerts, explanations, and what-if results served to the dashboard |

The online service never trains. It only loads files that the offline pipeline wrote.

## 3. Layered view

```mermaid
flowchart TB
    subgraph L1["Presentation layer"]
        UI["Pages and components"]
        HK["REST client and stream hook"]
    end
    subgraph L2["API layer"]
        RT["REST routers"]
        VS["Pydantic validation"]
        WS["WebSocket handler"]
    end
    subgraph L3["Domain services"]
        PR["Prediction service"]
        AL["Alert policy"]
        WI["What-if service"]
        EX["Explanation service"]
        PE["Personalization service"]
        RP["Replay engine"]
    end
    subgraph L4["Model runtime"]
        RG["Model registry"]
        NN["FusionTwinNet"]
        LG["LightGBM baseline"]
        CA["Calibrator"]
    end
    subgraph L5["Data access"]
        DB[("SQL database")]
        PQ[("Parquet files")]
        MF[("Model files")]
    end

    UI --> HK
    HK --> RT
    HK --> WS
    RT --> VS
    VS --> PR
    VS --> WI
    VS --> EX
    VS --> PE
    WS --> RP
    RP --> PR
    PR --> AL
    PR --> RG
    WI --> RG
    EX --> RG
    PE --> RG
    RG --> NN
    RG --> LG
    RG --> CA
    RG --> MF
    PR --> DB
    RP --> DB
    PE --> DB
    DB --- PQ
```

Rules that keep the layers clean:

1. The presentation layer talks only to the API layer, never to the database or model files.
2. Routers contain no business logic. They validate input, call one service, and shape the output.
3. Only the model registry touches model files. Services receive tensors and return plain Python objects.
4. Only the data access layer touches the database. The model code never imports SQLAlchemy.

## 4. Components

| Component | Responsibility | Key inputs | Key outputs | Spec |
|---|---|---|---|---|
| Ingestors | Read each open dataset and convert it to the canonical schema | Raw files | Canonical Parquet | TRD 4.1, 4.5 |
| Synthea post-processor | Turn Synthea CSV into static features with India-oriented adjustments | Synthea CSV | `static_features.parquet` | TRD 4.3 |
| Simulator | Generate glucose and wearable traces tied to the static profile | Static features | Time-series Parquet | TRD 4.4 |
| Feature and label builder | Build windows, scale features, create labels | Canonical Parquet | Training windows | TRD 4.6, 4.7 |
| Trainer | Train the baseline and `FusionTwinNet`, calibrate, export | Windows | Model files and metrics | TRD 5.2 to 5.4 |
| Evaluator | Compute metrics, ablations, lead time, plots | Test windows, model | `reports/results.md` | TRD 5.8 |
| Model registry | Load model, scaler, calibrator, metadata once at startup | `models/current/` | Ready-to-call predictor | TRD 6.1 |
| Prediction service | Build one window for a patient and time, run the model, apply calibration | Patient, time | Forecast and event probability | TRD 6.2 |
| Alert policy | Turn probability into severity, time to event, cool-down logic | Predictions over time | Alert state | TRD 5.4 |
| What-if service | Edit future covariates, run baseline and scenario passes | Edits | Two forecasts and deltas | TRD 5.6 |
| Explanation service | Integrated Gradients, grouped into readable factors | Window | Factor shares and text | TRD 5.7 |
| Personalization service | Fit and store the per-patient adapter | Patient history | `twin_state` row | TRD 5.5 |
| Replay engine | Walk a stored timeline at an accelerated speed and emit ticks | Patient, speed | WebSocket messages | TRD 6.3 |
| REST and WebSocket API | Expose everything to the dashboard | HTTP and WS requests | JSON | BACKEND-SCHEMA |
| Dashboard | Show patient list, live chart, risk, reasons, what-if | API | Visual interface | APP-FLOW |

## 5. Offline pipeline (training flow)

```mermaid
flowchart TD
    A["Download open datasets"] --> B["Ingest to canonical schema"]
    C["Run Synthea"] --> D["Post-process static features"]
    D --> E["Run simulator per patient"]
    B --> F["Merge real and synthetic cohorts"]
    E --> F
    F --> G["Resample to 15 minute grid, flag gaps"]
    G --> H["Patient-wise split: train, validation, test"]
    H --> I["Fit scalers on train only"]
    I --> J["Build windows, labels, covariates"]
    J --> K["Train LightGBM baseline"]
    J --> L["Train FusionTwinNet"]
    K --> M["Evaluate on validation"]
    L --> M
    M --> N["Calibrate and choose alert threshold"]
    N --> O["Final test on held-out real patients"]
    O --> P["Export model files and results report"]
    P --> Q["Build demo cohort and database"]
```

Key properties:

- The test split is created before any scaling or tuning, and it contains held-out real patients.
- Every step is a command in the Makefile, so the full chain can be repeated from a clean machine.
- Every artifact carries a data hash so a model can be traced back to the data it saw.

## 6. Online inference flow

A doctor opens a patient and the dashboard requests a prediction for the current simulated time.

```mermaid
sequenceDiagram
    participant D as Doctor
    participant W as React app
    participant A as FastAPI
    participant C as Prediction cache
    participant S as Prediction service
    participant M as Model runtime
    participant DB as Database

    D->>W: Open patient page
    W->>A: Request patient profile and twin state
    A->>DB: Read profile and twin state
    DB-->>A: Rows
    A-->>W: Profile JSON
    W->>A: Request prediction at time t
    A->>C: Look up patient, time, scenario
    alt Cache hit
        C-->>A: Stored prediction
    else Cache miss
        A->>S: Predict at t
        S->>DB: Read 24 hours of history and latest static values
        DB-->>S: Time-series and static features
        S->>M: Tensors for dynamic, static, future covariates
        M-->>S: Quantile forecast and event logit
        S->>S: Calibrate probability and apply alert policy
        S-->>A: Prediction object
        A->>C: Store result
    end
    A-->>W: Forecast, event risk, alert, data quality
    W->>A: Request explanation
    A->>S: Explain at t
    S->>M: Integrated Gradients on event logit
    M-->>S: Attributions
    S-->>A: Grouped factors and text
    A-->>W: Explanation JSON
    W-->>D: Chart, risk panel, reasons
```

## 7. Live replay flow

The replay engine makes stored data behave like a live wearable feed.

```mermaid
sequenceDiagram
    participant W as React app
    participant WS as WebSocket handler
    participant R as Replay engine
    participant S as Prediction service
    participant DB as Database

    W->>WS: Open stream for patient
    WS->>R: Create session at start time
    loop Every grid step at the chosen speed
        R->>DB: Read next 15 minute sample
        DB-->>R: Sample
        R->>S: Predict at new time
        S-->>R: Prediction and alert
        R-->>WS: Tick message
        WS-->>W: Tick message
        W->>W: Update chart, risk badge, alert banner
    end
    W->>WS: Control message such as pause, speed, or inject meal
    WS->>R: Apply control
    R-->>WS: State change confirmed by next tick
    R-->>WS: End message when timeline finishes
```

Design notes:

- The replay engine owns time. The browser never decides which sample comes next.
- If inference is slower than the tick interval, the engine skips ticks instead of building a queue, so the dashboard never falls behind.
- `inject_meal` writes an extra meal into the replayed session (not into the stored patient data) so the demo can show the twin reacting to something new.
- On a dropped connection the client reconnects with backoff and sends a `seek` to the last simulated time it received.

## 8. What-if flow

```mermaid
sequenceDiagram
    participant D as Doctor
    participant W as React app
    participant A as FastAPI
    participant X as What-if service
    participant M as Model runtime

    D->>W: Add a 20 minute walk after dinner
    W->>A: Send what-if request with edits
    A->>A: Validate edit types and bounds
    A->>X: Run what-if
    X->>M: Pass 1, no extra plan, cov_mask 1
    M-->>X: Baseline forecast and probability
    X->>M: Pass 2, edited plan, cov_mask 1
    M-->>X: Scenario forecast and probability
    X->>X: Compute peak, probability, and minutes above 180
    X-->>A: Both forecasts and deltas
    A-->>W: What-if response with disclaimer
    W-->>D: Before and after chart and summary
```

## 9. Model architecture

```mermaid
flowchart TD
    SV["Static vector, 21 values"] --> SE["Static encoder MLP"]
    SE --> ZS["Static embedding, 64"]
    ZS --> H0["Initial hidden state"]
    DH["Dynamic history, 96 steps by 16 channels"] --> ST["Conv1d stem"]
    ST --> GR["GRU encoder, 2 layers"]
    H0 --> GR
    GR --> SM["Summary of last steps"]
    SM --> FU["Fusion layer"]
    ZS --> FU
    FU --> F["Fused patient state, 128"]
    F --> AD["Per-patient adapter, added after fusion"]
    AD --> FS["Adapted state"]
    FC["Future covariates, 8 steps by 5 channels"] --> DG["Decoder GRU"]
    FS --> DG
    DG --> QH["Quantile head"]
    QH --> OUT1["Forecast P10, P50, P90 for 8 steps"]
    FS --> EH["Event head"]
    DG --> EH
    EH --> OUT2["Event probability, then calibrated"]
```

What makes this a fusion model and not two models glued together:

1. The static profile sets the starting state of the temporal encoder, so identical sensor traces are read differently for different patients.
2. The static embedding is also joined after the encoder, so it can adjust the final decision.
3. The fused state conditions both the forecast decoder and the event head.

What makes it behave like a twin:

1. A persistent fused state per patient, with an optional learned adapter.
2. Known-future covariates, so planned meals and activity can be changed and the future re-simulated.
3. Uncertainty bands, so the twin shows how sure it is.

## 10. Data model overview

The full definition is in `BACKEND-SCHEMA.md`. This diagram shows the relationships.

```mermaid
erDiagram
    PATIENTS ||--o{ CONDITIONS : has
    PATIENTS ||--o{ MEDICATIONS : takes
    PATIENTS ||--o{ LAB_RESULTS : has
    PATIENTS ||--o{ TIMESERIES : produces
    PATIENTS ||--o{ PREDICTIONS : receives
    PATIENTS ||--o{ ALERTS : triggers
    PATIENTS ||--o| TWIN_STATE : owns
    MODEL_VERSIONS ||--o{ PREDICTIONS : created_by

    PATIENTS {
        string patient_id PK
        string source
        int age
        string sex
        float bmi
        float hba1c
        float prs_z
    }
    TIMESERIES {
        string patient_id FK
        datetime ts
        float glucose_mgdl
        float hr_bpm
        int steps
        string sleep_stage
        float carbs_g
    }
    PREDICTIONS {
        int id PK
        string patient_id FK
        datetime at
        float event_probability
        string severity
        string model_version FK
    }
    ALERTS {
        int id PK
        string patient_id FK
        datetime started_at
        string severity
        int event_occurred
    }
    TWIN_STATE {
        string patient_id PK
        int personalized
        float days_of_data
        blob adapter_blob
    }
```

## 11. Deployment view

```mermaid
flowchart LR
    subgraph HOST["Laptop or single server with Docker"]
        subgraph WEBC["web container"]
            NG["nginx serving the React build"]
        end
        subgraph APIC["api container"]
            UV["Uvicorn running FastAPI"]
            MD["Model runtime, CPU"]
        end
        V1[("Volume: models/current, read only")]
        V2[("Volume: data/demo with SQLite, read only")]
    end
    BR["Browser"] --> NG
    NG -->|"proxy /api and WebSocket"| UV
    UV --> MD
    MD --> V1
    UV --> V2
```

Notes:

- No outbound internet access is needed at demo time.
- The `trainer` container is optional and runs only when reproducing training.
- For a hosted demo, put the same two containers on any small server and add HTTPS at the proxy.

## 12. Key design decisions

The full decision log is in `TRD.md` section 14. The ones that shape the architecture:

| Decision | Reason | Trade-off |
|---|---|---|
| Separate offline and online worlds | Training is slow and the demo must be fast and safe | Model changes need a re-export |
| Replay engine instead of real device ingestion | Reproducible, safe, works offline | Not a true live feed |
| One shared model runtime inside the API process | Simple, low latency, no network hop | Scaling needs one process per worker |
| Calibration outside the network | Probabilities can be refit without retraining | One more artifact to version |
| Explanation by grouped Integrated Gradients | Doctors read factors, not 1,500 raw features | Grouping choices must be documented |
| SQLite for the demo cohort | No setup, easy to ship in the repository | Not for concurrent heavy writes |
| Typed contracts shared with the frontend through OpenAPI | Fewer integration bugs | Need to regenerate types after API changes |

## 13. Cross-cutting concerns

### 13.1 Reliability and degradation

| Failure | Behavior |
|---|---|
| Model files missing or invalid | API starts but `/health` reports degraded. Prediction endpoints return 503. The dashboard shows a clear message. |
| Glucose data missing for more than 40 percent of the window | Prediction is returned with `low_confidence = true` and the alert is suppressed. |
| Static features outside the training range | Same low confidence behavior and a note in the explanation. |
| Inference slower than the tick interval | Replay skips ticks. |
| WebSocket disconnect | Client reconnects with backoff and resumes from the last simulated time. |
| Database file locked or unreadable | API returns 503 with a retry hint. |
| Explanation too slow | Fall back to group occlusion. |

### 13.2 Observability

- Structured JSON logs with request ID, patient ID (synthetic), endpoint, latency, model version.
- A `/health` endpoint that reports model version, database status, and cache size.
- Latency numbers for predict, explain, and what-if are logged so the performance budget can be checked.
- No logging of anything that could be personal data (the system has none, but the habit matters).

### 13.3 Configuration

All tunable values live in `configs/default.yaml` and environment variables prefixed `GT_`. No magic numbers in code. The model metadata file records the configuration used at training time.

### 13.4 Security and privacy

- Synthetic or open anonymized data only.
- Input validation on every request with explicit bounds.
- CORS limited to the dashboard origin.
- No secrets in the repository.

### 13.5 Clinical safety

- A non-dismissable prototype banner on every page.
- No treatment, dose, or diet advice anywhere.
- What-if outputs labeled as model estimates.
- Low confidence handling as described above.

## 14. Extension path

| Direction | What changes |
|---|---|
| Another condition, such as hypertension | Replace the event definition, the dynamic channels (blood pressure, heart rate variability), and the static feature list. The layers, API shape, and dashboard pattern stay the same. |
| Real wearable feed | Replace the replay engine input with an ingestion service reading from a device API or Health Connect, and add authentication and consent. |
| Real EHR | Replace the synthetic loader with a FHIR reader and add privacy controls, audit logs, and encryption. |
| More patients | Move SQLite to PostgreSQL, run several API workers, and move the cache to Redis. |
| Better physiology | Blend a mechanistic glucose model with the learned model (hybrid twin). |

## 15. Exporting diagrams for the submission

The challenge requires an architecture diagram as PDF or PPT. Two simple routes:

**Route A: Mermaid CLI**

```bash
npm install -g @mermaid-js/mermaid-cli

# save one diagram block, for example the system context, as docs/diagrams/system.mmd
mmdc -i docs/diagrams/system.mmd -o docs/diagrams/system.pdf
mmdc -i docs/diagrams/layers.mmd -o docs/diagrams/layers.pdf

# or render every diagram in this file to images
mmdc -i docs/ARCHITECTURE.md -o docs/ARCHITECTURE.rendered.md
```

Merge the PDFs into one file (`architecture.pdf`) with any PDF tool and add a title page and a legend.

**Route B: draw.io**

In draw.io, use Arrange, Insert, Advanced, Mermaid to paste a diagram, then restyle it and export as PDF. Use this route if you want a more polished slide-ready look.

Recommended contents of the submitted diagram file:

1. Page 1: system context (section 2) and layered view (section 3).
2. Page 2: offline pipeline (section 5).
3. Page 3: model architecture (section 9).
4. Page 4: online inference and replay flows (sections 6 and 7).

Keep the same names and colors in the slides and the video so judges can follow along.
