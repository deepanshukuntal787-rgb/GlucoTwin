# Technical Requirements Document (TRD)

**Project:** GlucoTwin (working title)
**Companion documents:** `PRD.md` (what and why), `ARCHITECTURE.md` (system design and diagrams)
**Document version:** 1.0
**Date:** 2026-10-02
**Status:** Draft for team review

---

## 1. Purpose and scope

This document specifies how GlucoTwin is built: data sources and processing, the machine learning model, the backend API, the database, the dashboard, deployment, testing, and safety. Every requirement ID (FR-xx) refers to the PRD.

In scope: one condition (Type 2 Diabetes), one event (hyperglycemia within 120 minutes), synthetic and open data only, a local or single-server deployment.

Out of scope: real patient data, clinical validation, real wearable hardware, treatment advice.

## 2. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language (backend and ML) | Python 3.11 | Type hints throughout |
| Deep learning | PyTorch 2.x | CPU is enough, GPU optional |
| Classical ML | scikit-learn, LightGBM | Baseline, calibration, metrics |
| Explainability | Captum (Integrated Gradients), SHAP (for LightGBM) | Grouped into readable factors |
| Data handling | pandas, NumPy, PyArrow (Parquet) | Parquet for training data |
| Synthetic EHR | Synthea (Java 11 or newer) | Output CSV, post-processed |
| Experiment tracking | Plain JSON and CSV logs (MLflow optional) | Keep it light |
| API | FastAPI, Uvicorn, Pydantic v2 | REST plus WebSocket |
| Database | SQLite for development and demo, PostgreSQL optional | SQLAlchemy 2.x |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS | |
| Charts | Recharts | Composed chart for band and lines |
| Client data | TanStack Query (REST), custom WebSocket hook | |
| Packaging | Docker and Docker Compose | One command start |
| Quality | pytest, ruff, mypy (optional), Vitest | |
| CI | GitHub Actions | Lint and tests on each push |
| Diagrams | Mermaid in Markdown, draw.io for the PDF or PPT | |
| License | MIT (code), dataset licenses listed separately | |

## 3. Repository layout

The repository folder must be named `Team Name_College Name`.

```
Team Name_College Name/
├── README.md
├── LICENSE
├── docs/
│   ├── PRD.md
│   ├── TRD.md
│   ├── ARCHITECTURE.md
│   ├── architecture.pdf            (submission diagram)
│   ├── presentation.pdf            (submission slides)
│   ├── DATA_PROVENANCE.md
│   └── MODEL_CARD.md
├── data/
│   ├── raw/                        (not committed, scripts download it)
│   ├── synthea/                    (config and generated CSV, small sample committed)
│   ├── processed/                  (Parquet, not committed)
│   └── demo/                       (small committed demo cohort, 5 patients)
├── gluco_twin/                     (Python package)
│   ├── config.py
│   ├── data/
│   │   ├── ingest_cgmacros.py
│   │   ├── ingest_bigideas.py
│   │   ├── ingest_shanghai.py
│   │   ├── synthea_postprocess.py
│   │   ├── simulator.py            (synthetic glucose and wearable generator)
│   │   ├── harmonize.py            (canonical schema and grid)
│   │   ├── features.py
│   │   ├── labels.py
│   │   └── splits.py
│   ├── models/
│   │   ├── baseline_lgbm.py
│   │   ├── fusion_net.py
│   │   ├── train.py
│   │   ├── calibrate.py
│   │   ├── personalize.py
│   │   ├── whatif.py
│   │   └── explain.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── lead_time.py
│   │   └── report.py
│   ├── service/
│   │   ├── main.py                 (FastAPI app)
│   │   ├── routers/
│   │   ├── schemas.py
│   │   ├── registry.py             (model loading)
│   │   ├── replay.py               (WebSocket replay engine)
│   │   ├── alerts.py
│   │   └── db.py
│   └── cli.py                      (data, train, evaluate commands)
├── web/                            (React app)
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── api/
│   └── package.json
├── models/current/                 (model.pt, scaler.json, calibrator.pkl, meta.json)
├── tests/
├── configs/default.yaml
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## 4. Data engineering

### 4.1 Data sources

Verify availability, access steps, and license for every dataset before committing to it. Record the result in `docs/DATA_PROVENANCE.md`.

| Source | Type | Use | Notes to verify |
|---|---|---|---|
| Synthea | Synthetic EHR | Static features for the synthetic cohort | Default demographics are US-based, so post-processing is needed. Genetic markers are not produced by Synthea. |
| CGMacros (PhysioNet) | Open, real, anonymized | Real CGM, wearable, meals, and clinical info for a small group including people with prediabetes and T2D | Check access terms, participant count, sensor types, and which channels exist |
| BIG IDEAs Lab glycemic variability and wearable data (PhysioNet) | Open, real, anonymized | CGM, wrist sensor heart rate, food logs | Small cohort with elevated HbA1c. Check license. |
| Shanghai T2DM CGM dataset | Open, real, anonymized | CGM plus clinical tables for people with T2D | Likely no heart rate or sleep. Check license. |
| Own simulator | Synthetic | Scale, physiology tied to static profile, Indian meal patterns | See section 4.4 |
| MIMIC-IV | Credentialed | Not used | Needs credentialed access and training, too slow for this timeline |

### 4.2 Training cohort design

The core challenge is that real open CGM datasets rarely have a full EHR attached. The plan uses two cohorts and states the limitation openly.

| Cohort | Static features from | Dynamic features from | Role |
|---|---|---|---|
| A: Real open | The dataset's own demographic and lab tables (age, sex, BMI, HbA1c, fasting glucose, medicines when present) | Real CGM and wearables (missing channels flagged) | Validation and final test, plus part of training |
| B: Synthetic | Synthea patients, post-processed, with a synthetic polygenic risk score | Simulator, with physiology parameters derived from the static profile | Scale and controlled fusion |

Rules:

1. The **final test set contains only real open subjects** (held out by patient), plus a separate synthetic test set from a different random seed. Report both.
2. Never mix subjects across splits.
3. Report results trained on B only, A only, and A plus B.
4. State in the README that Cohort B results show internal consistency, and Cohort A results show external realism.

### 4.3 Synthea configuration and post-processing

1. Generate 300 to 500 adult patients with a diabetes module enabled. Keep only patients with Type 2 Diabetes or prediabetes diagnoses.
2. Extract from the CSV outputs: `patients`, `conditions`, `observations` (HbA1c, BMI, blood pressure, eGFR when available), `medications`.
3. Post-process to improve Indian relevance (document each step):
   - Shift age and BMI distributions toward ranges typical for Indian adults with T2D (for example, lower BMI thresholds for risk, since risk appears at lower BMI in South Asians).
   - Map medicines to those common in India (metformin, glimepiride, gliclazide, sitagliptin or vildagliptin, and insulin in a smaller share).
   - Add a **synthetic polygenic risk score** `prs_z` drawn from a normal distribution with a small positive correlation to HbA1c. Label it synthetic everywhere.
   - Add `family_history` (binary) correlated with `prs_z`.
4. Output one row per patient into `data/processed/static_features.parquet`.

### 4.4 Synthetic glucose and wearable simulator

The simulator produces 30 or more days per patient on a 5-minute internal grid, then resamples to the canonical grid. The structure is a simple, documented model, not a clinical simulator.

**Glucose dynamics (per 5-minute step):**

```
dG/dt = -k_c * S * (G - G_b)  +  Ra(t) / V_g  -  k_a * A(t) * max(G - G_floor, 0)  +  dawn(t)  +  eta(t)
```

| Term | Meaning | Source of the value |
|---|---|---|
| `G_b` | Basal glucose | Derived from HbA1c using the estimated average glucose relation eAG = 28.7 x HbA1c - 46.7 (mg/dL), then calibrated so simulated mean glucose is close to eAG |
| `S` | Insulin sensitivity (dimensionless, around 1) | Decreases with BMI, HbA1c, and age, and with poor sleep the night before (assumption: 10 to 25 percent reduction after under 6 hours or low deep sleep). Increases slightly with metformin. |
| `k_c` | Clearance rate constant | Fixed, tuned for plausible meal recovery times |
| `Ra(t)` | Rate of glucose appearance from meals | Sum over meals of `carbs_g * GI_factor * gamma_kernel(t - t_meal)` with a peak around 45 to 90 minutes |
| `V_g` | Distribution volume | Fixed |
| `A(t)` | Activity intensity | Derived from the step series, acts with a short delay |
| `k_a`, `G_floor` | Activity glucose uptake and floor | Fixed |
| `dawn(t)` | Early morning rise | 10 to 20 mg/dL bump between 04:00 and 08:00 |
| `eta(t)` | Process and sensor noise | AR(1) noise plus white sensor noise (standard deviation about 3 mg/dL), occasional sensor dropouts of 15 to 60 minutes |

**Behavior generators:**

| Signal | Generation rule |
|---|---|
| Meals | Breakfast 07:30 to 09:30, lunch 13:00 to 14:30, evening snack 17:00 to 18:00, dinner 20:00 to 22:30. Carbohydrates drawn per patient habit (main meals roughly 50 to 130 g, snack 15 to 40 g). Rice, roti, and sweet-heavy meals get a higher `GI_factor`. Some patients are "late dinner" types. |
| Steps | Daily routine with a morning or evening walk probability, random bursts, and low activity overnight |
| Heart rate | Resting HR from age and BMI, plus activity effect, drop during sleep, random variability |
| Sleep | Nightly schedule with random bedtime, duration (5 to 9 hours), and a simple hypnogram of awake, light, deep, REM |
| Medicine adherence | Optional random missed dose days that lower `S` slightly |

**Calibration checks (must pass before use):**

- Mean simulated glucose per patient is within 15 mg/dL of the eAG implied by HbA1c for at least 90 percent of patients.
- Time in range (70 to 180) and time above range distributions are plausible for T2D (a spread across patients, not all the same).
- Post-meal peak shape and timing resemble real traces from Cohort A (visual check plus summary statistics such as peak delay and excursion size).

**Honesty note:** because static traits drive the simulator, a model can learn the simulator rules. This is why Cohort A is the final test and why results are reported separately.

### 4.5 Canonical schema and time grid

Everything is converted to one schema before modeling.

| Column | Type | Description |
|---|---|---|
| `patient_id` | string | Unique, prefixed by source (`cgm_`, `big_`, `sha_`, `syn_`) |
| `ts` | timestamp | Start of the grid interval, local time |
| `glucose_mgdl` | float, nullable | Mean of readings in the interval |
| `hr_bpm` | float, nullable | Mean heart rate |
| `steps` | int, nullable | Steps in the interval |
| `sleep_stage` | category, nullable | `awake`, `light`, `deep`, `rem` |
| `carbs_g` | float, default 0 | Carbohydrates logged at a meal in the interval |
| `source` | category | `real_open` or `synthetic` |

Rules:

- Canonical grid: **15 minutes** (`GRID_MINUTES=15` in config). It works for 5-minute and 15-minute sensors. Finer grids are a config change.
- Gaps in glucose up to 30 minutes are filled by linear interpolation, and the missing-data flag stays on. Longer gaps stay missing.
- Physiologically impossible values are removed (glucose below 40 or above 500 mg/dL, heart rate below 30 or above 220 bpm).
- Units are converted to mg/dL for glucose.
- Static features are stored separately, one row per patient.

### 4.6 Model inputs

**Dynamic input per step (16 channels), history window T = 96 steps (24 hours):**

| # | Channel | Notes |
|---|---|---|
| 1 | `glucose_z` | Z-scored glucose, missing filled with the last value |
| 2 | `glucose_diff` | One-step change |
| 3 | `hr_z` | Z-scored heart rate |
| 4 | `steps_z` | Log-scaled, then z-scored |
| 5 to 8 | `sleep_awake`, `sleep_light`, `sleep_deep`, `sleep_rem` | One-hot, all zero when unknown |
| 9 | `carbs_z` | Meal size at this step |
| 10 | `since_meal_z` | Minutes since last meal, clipped at 360 |
| 11 to 12 | `tod_sin`, `tod_cos` | Time of day |
| 13 to 16 | `mask_glucose`, `mask_hr`, `mask_steps`, `mask_sleep` | 1 if the channel was observed |

**Future covariates per horizon step (5 channels), horizon H = 8 steps (120 minutes):**

| # | Channel | Notes |
|---|---|---|
| 1 | `plan_carbs_z` | Planned meal size |
| 2 | `plan_steps_z` | Planned activity |
| 3 to 4 | `tod_sin`, `tod_cos` | Time of day, always known |
| 5 | `cov_mask` | 1 if the planned values are provided, 0 if unknown |

**Static vector (about 21 values):**

| Group | Features |
|---|---|
| Demographics and vitals | age, sex, BMI, systolic blood pressure |
| Labs | HbA1c, fasting glucose, eGFR (each with a missing flag, median imputation) |
| Disease | years since diagnosis |
| Medicines (binary) | metformin, sulfonylurea, DPP-4 inhibitor, SGLT2 inhibitor, insulin |
| Comorbidities (binary) | hypertension, dyslipidemia, chronic kidney disease |
| Risk | family history, synthetic polygenic risk score `prs_z` |

Scalers are fitted on the training split only and saved to `models/current/scaler.json`.

### 4.7 Labels

For each prediction time `t` (a grid step) with a full 24-hour history:

```python
def make_labels(glucose, t, horizon=8, thresh=180, run=3):
    """glucose: array on the 15-minute grid. Uses only values after t."""
    future = glucose[t + 1 : t + 1 + horizon]          # next 120 minutes
    targets = {                                          # regression targets
        "g_30":  future[1],                              # +30 min
        "g_60":  future[3],                              # +60 min
        "g_120": future[7],                              # +120 min
        "path":  future,                                 # all 8 steps
    }
    above = future > thresh
    # event: `run` consecutive readings above threshold starting inside the horizon
    event = any(above[i:i + run].all() for i in range(len(above) - run + 1))
    already_high = glucose[t] > thresh
    return targets, int(event), bool(already_high)       # already_high is excluded from the classifier metric
```

Notes:

- A sample is dropped if more than 25 percent of its future glucose values are missing.
- Event onset time is stored for lead-time calculation.
- The threshold and run length are config values (`event.threshold_mgdl`, `event.min_run_steps`).

### 4.8 Splits and leakage controls

| Split | Content |
|---|---|
| Train | About 70 percent of patients (real and synthetic) |
| Validation | About 15 percent, used for early stopping, calibration, and threshold choice |
| Test (real) | Held-out real open patients, never used in training or tuning |
| Test (synthetic) | Synthetic patients generated with a different seed |

Additional protocols:

- **Grouped K-fold** (K = 5) by `patient_id` for confidence intervals on the train and validation pool.
- **Leave-one-dataset-out:** train on synthetic plus two real datasets, test on the third.
- **Patient bootstrap** (1000 resamples) for confidence intervals on the test metrics.

Leakage checklist (each item has an automated test):

1. No patient appears in two splits.
2. Scalers and imputers are fit on training data only.
3. Inputs use only data at or before `t`. Targets use only data after `t`.
4. No feature is derived from the future (for example, daily mean glucose that includes future hours).
5. The `already_high` samples are excluded from the classification metrics.
6. Meal inputs in the history are logged meals. Future meals appear only as future covariates.

## 5. Modeling

### 5.1 Baselines

| Baseline | Purpose |
|---|---|
| Persistence (last value) | Floor for regression |
| Linear trend extrapolation | Simple dynamics check |
| LightGBM with engineered features | Strong classical comparison |

LightGBM features: last glucose, deltas over 15, 30, 60 minutes, rolling mean, standard deviation, min and max over 1, 3, 6, 24 hours, time since last meal, last meal carbohydrates, average heart rate and steps over several windows, total sleep and deep sleep fraction last night, time of day, and all static features. One classifier for the event, and three regressors for the +30, +60, +120 values.

### 5.2 Fusion network

Name: `FusionTwinNet`. About 300 thousand parameters, so it trains on a laptop.

```
Static vector (B, 21)
   -> StaticEncoder: Linear(21,64) -> GELU -> LayerNorm -> Dropout(0.1) -> Linear(64,64)   => z_s (B, 64)
   -> h0 = Linear(64 -> 2x128) reshaped as the initial hidden state of the temporal encoder

Dynamic history (B, 96, 16)
   -> Conv1d stem (kernel 3, 16 -> 64) -> GELU
   -> GRU (2 layers, hidden 128), initial hidden state = h0          => outputs (B, 96, 128)
   -> summary = concat(last step output, mean of the last 8 outputs)  => (B, 256) -> Linear -> (B, 128)

Fusion
   -> f = GELU(Linear(concat(summary 128, z_s 64) -> 128)) -> Dropout(0.1)   => f (B, 128)

Decoder (known-future covariates)
   -> Future covariates (B, 8, 5)
   -> GRU (hidden 64), initial hidden state = Linear(f -> 64)        => (B, 8, 64)
   -> Quantile head: Linear(64 -> 3) per step => P50, plus softplus offsets for P10 and P90 (non-crossing)

Event head
   -> concat(f 128, mean of decoder outputs 64) -> Linear(192 -> 64) -> GELU -> Linear(64 -> 1)   => event logit
```

Why this design:

- The static profile does two jobs: it **initializes the dynamic state** (so the same sensor history means different things for different patients), and it is **concatenated late** (so it can also adjust the final decision). This is true fusion, not two separate models.
- Known-future covariates are what make what-if simulation possible.
- Quantile outputs give the uncertainty band shown on the dashboard.

Variants for ablation: TCN encoder instead of GRU, FiLM conditioning instead of hidden-state initialization, small transformer encoder.

### 5.3 Training

| Item | Value |
|---|---|
| Loss | Pinball loss for quantiles 0.1, 0.5, 0.9 (targets scaled as (G - 140) / 50) plus binary cross-entropy for the event, weighted 1.0 each to start |
| Class imbalance | `pos_weight` set from training prevalence, and balanced batch sampling as an option |
| Optimizer | AdamW, learning rate 1e-3, weight decay 1e-2, cosine schedule with warmup |
| Batch size | 256 |
| Epochs | Up to 40, early stopping with patience 6 on validation AUPRC plus negative RMSE at 60 minutes |
| Window stride | 2 steps for training, 1 step for evaluation |
| Augmentation | Gaussian noise on glucose (standard deviation 2 mg/dL), random channel dropout for heart rate, steps, and sleep (probability 0.15 each, mask set to 0), future covariate masking (probability 0.5, `cov_mask = 0`), small time shifts |
| Data mix | Real training subjects oversampled so they are at least 30 percent of each batch |
| Seeds | Fixed. Report mean and standard deviation over 3 seeds. |

### 5.4 Calibration and alert policy

1. Fit isotonic regression (or Platt scaling if data is small) on validation event probabilities. Save as `calibrator.pkl`.
2. Choose the alert threshold on validation: the smallest threshold with precision at least 0.6 and fewer than 6 alerts per patient per day. Report the precision-recall curve.
3. Severity: Low below 0.30, Moderate 0.30 to 0.60, High above 0.60 or a P50 forecast above 250 mg/dL within 60 minutes (starting values, adjust on validation).
4. Time to event: the first horizon step where the P50 forecast crosses 180 mg/dL. If it does not cross but the probability is high, report "within 120 minutes".
5. Cool-down: after an alert, suppress the same or lower severity for 60 minutes unless probability rises by 0.15 or more. Clear the alert when probability stays below threshold minus 0.10 for 2 steps.

### 5.5 Personalization (the "twin" behavior)

- After a patient has at least 3 days of data, learn a small **per-patient adapter**: a bias vector `b_p` (dimension 128) added to the fused state `f`, plus a scale and shift on the glucose output.
- The base network is frozen. Train for at most 200 gradient steps with L2 regularization on `b_p`.
- Store the adapter in `twin_state` (see section 7).
- Evaluation: compare RMSE and AUPRC on days after the adaptation period with and without the adapter. Report the result, even if the gain is small.

### 5.6 What-if simulation

Input: a list of edits to the future covariates.

| Edit type | Parameters | Effect on covariates |
|---|---|---|
| `meal` | `start_offset_min`, `carbs_g` | Sets `plan_carbs_z` at the matching step |
| `skip_meal` | `start_offset_min` | Sets planned carbohydrates to 0 at that step |
| `activity` | `start_offset_min`, `duration_min`, `steps_per_15min` | Sets `plan_steps_z` over the matching steps |

Procedure:

1. Run the baseline pass with `cov_mask = 1` and all-zero plans ("nothing extra").
2. Run the scenario pass with `cov_mask = 1` and the edited plans.
3. Return both forecasts and the change in peak, event probability, and time above 180 mg/dL.
4. Validate inputs: carbohydrates from 0 to 200 g, start offset within the horizon, duration up to 120 minutes, steps within the observed range.

Sanity tests (reported in the model card):

- **Monotonic checks** on test windows: adding carbohydrates should not lower the predicted peak, and adding activity should not raise it. Target at least 90 percent of tested windows satisfying each rule.
- Scenario outputs always show the label "model estimate for exploration, not medical advice".

### 5.7 Explanations

- **Method:** Integrated Gradients (Captum) on the event logit, 32 interpolation steps, baseline equal to the per-channel training median. For LightGBM, SHAP TreeExplainer.
- **Grouping:** attributions are summed by absolute value into readable factors and converted to percentages:

| Factor shown to the doctor | Inputs grouped |
|---|---|
| Recent glucose trend | Glucose and glucose change over the last 2 hours |
| Glucose history | Glucose over the earlier window |
| Last meal | Carbohydrates and time since meal channels |
| Sleep | Sleep channels over the previous night |
| Heart rate | Heart rate channel |
| Activity | Steps channel |
| Time of day | Time encoding |
| Background risk | All static features |

- Direction (raises or lowers risk) comes from the signed attribution sum for each group.
- Text template example: "Risk is mainly driven by a late high-carbohydrate meal (42 percent) and poor sleep last night (23 percent)."
- Fallback if Integrated Gradients is too slow: group occlusion (zero out one group at a time and measure the change in probability).

### 5.8 Evaluation protocol and reporting

| Metric | Definition |
|---|---|
| AUROC, AUPRC | Event within 120 minutes, `already_high` samples excluded, with prevalence shown beside AUPRC |
| Brier score, expected calibration error | Probability quality, with a reliability diagram |
| RMSE and MAE | At +30, +60, +120 minutes, in mg/dL |
| Interval coverage | Share of true values inside the P10 to P90 band (target near 80 percent) |
| Median warning lead time | For true events that triggered an alert, time between first alert and event onset |
| Sensitivity at chosen threshold | Share of events that were alerted before onset |
| Alerts per patient per day | Alert burden |
| Clarke or Parkes error grid | Optional, for the +30 and +60 minute forecasts |

Required ablations:

1. Static only, dynamic only, and fused.
2. With and without heart rate, steps, and sleep.
3. With and without personalization.
4. GRU, TCN, and FiLM variants.
5. Trained on synthetic only, real only, and both.

Every result table includes mean, standard deviation across seeds, and a patient-bootstrap confidence interval. `gluco_twin/eval/report.py` writes `reports/results.md` and figures, and the README links to it.

### 5.9 Initial performance targets

These are starting goals to revise after the baseline is measured.

| Metric | Target |
|---|---|
| AUROC | 0.85 or higher |
| RMSE at +30, +60, +120 minutes | 15, 25, 35 mg/dL or lower |
| Median lead time | 60 minutes or more |
| Expected calibration error | Below 0.08 |
| Alerts per patient per day | Fewer than 6 |

## 6. Backend service

### 6.1 Structure

- `service/main.py` creates the FastAPI app, loads the model once at startup through `registry.py`, and mounts routers.
- Model files live in `models/current/`: `model.pt` (state dict or TorchScript), `scaler.json`, `calibrator.pkl`, `meta.json` (version, training data hash, metrics, thresholds).
- Inference uses `torch.inference_mode()` and `torch.set_num_threads` set from config.
- Predictions are cached in an LRU cache keyed by `(patient_id, sim_time, scenario_hash)`.
- CORS is limited to the dashboard origin.

### 6.2 REST API (prefix `/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and model version |
| GET | `/model/info` | Model metadata, metrics, thresholds, data provenance summary |
| GET | `/patients` | List patients with current glucose, risk level, last update |
| GET | `/patients/{id}` | Static profile and twin state |
| GET | `/patients/{id}/timeseries` | Query parameters `from`, `to`, `channels` |
| POST | `/patients/{id}/predict` | Prediction at a given simulated time |
| POST | `/patients/{id}/whatif` | Baseline and scenario forecasts |
| GET | `/patients/{id}/explain` | Grouped attributions at a given time |
| POST | `/patients/{id}/personalize` | Fit or refresh the per-patient adapter |
| GET | `/patients/{id}/alerts` | Alert history |
| WS | `/patients/{id}/stream` | Replay stream (see 6.3) |

**Predict request:**

```json
{ "at": "2026-03-14T19:45:00", "use_personalization": true }
```

**Predict response:**

```json
{
  "patient_id": "syn_0042",
  "at": "2026-03-14T19:45:00",
  "current_glucose_mgdl": 156,
  "forecast": [
    { "offset_min": 15,  "p10": 150, "p50": 164, "p90": 181 },
    { "offset_min": 30,  "p10": 152, "p50": 175, "p90": 201 },
    { "offset_min": 60,  "p10": 160, "p50": 203, "p90": 244 },
    { "offset_min": 120, "p10": 148, "p50": 214, "p90": 266 }
  ],
  "event": {
    "probability": 0.78,
    "severity": "high",
    "estimated_minutes_to_event": 75,
    "threshold_mgdl": 180
  },
  "alert": { "active": true, "since": "2026-03-14T19:30:00" },
  "data_quality": { "glucose": 0.99, "hr": 0.94, "sleep": 1.0, "last_update_s": 12 },
  "model_version": "0.1.0"
}
```

**What-if request:**

```json
{
  "at": "2026-03-14T19:45:00",
  "edits": [
    { "type": "activity", "start_offset_min": 15, "duration_min": 20, "steps_per_15min": 1200 }
  ]
}
```

**What-if response:**

```json
{
  "baseline": { "peak_mgdl": 238, "event_probability": 0.78, "minutes_above_180": 70 },
  "scenario": { "peak_mgdl": 201, "event_probability": 0.41, "minutes_above_180": 30 },
  "baseline_forecast": [ { "offset_min": 15, "p10": 150, "p50": 164, "p90": 181 } ],
  "scenario_forecast": [ { "offset_min": 15, "p10": 148, "p50": 160, "p90": 176 } ],
  "disclaimer": "Model estimate for exploration, not medical advice."
}
```

**Explain response:**

```json
{
  "factors": [
    { "name": "Last meal", "share": 0.42, "direction": "raises" },
    { "name": "Sleep", "share": 0.23, "direction": "raises" },
    { "name": "Recent glucose trend", "share": 0.18, "direction": "raises" },
    { "name": "Background risk", "share": 0.17, "direction": "raises" }
  ],
  "text": "Risk is mainly driven by a late high-carbohydrate meal (42%) and poor sleep last night (23%)."
}
```

Errors use a standard body: `{ "error": { "code": "invalid_edit", "message": "carbs_g must be between 0 and 200" } }` with suitable HTTP status codes (400, 404, 422, 500).

### 6.3 WebSocket replay protocol

The server replays a patient's stored timeline as if it were live.

- Speed presets (simulated time per real time): 60x, 300x, 900x. Default 900x means one 15-minute grid step per real second.
- Server to client message on each step:

```json
{
  "type": "tick",
  "sim_time": "2026-03-14T19:45:00",
  "glucose": 156, "hr": 82, "steps": 340, "sleep_stage": null,
  "prediction": { "...same shape as the predict response..." },
  "alert": { "active": true, "severity": "high" }
}
```

- Client to server control messages:

```json
{ "type": "control", "action": "play" }
{ "type": "control", "action": "pause" }
{ "type": "control", "action": "seek", "value": "2026-03-14T08:00:00" }
{ "type": "control", "action": "speed", "value": 900 }
{ "type": "control", "action": "inject_meal", "carbs_g": 80 }
```

- `inject_meal` modifies the replayed history from that step onward, so the live demo can show the twin reacting to a new event.
- The server sends a `{ "type": "end" }` message when the timeline finishes, and a heartbeat every 10 seconds.
- If the model is slow, the server drops intermediate ticks rather than queueing them.

## 7. Database schema

SQLite for the demo. The same schema works on PostgreSQL. Large training data stays in Parquet, and only the demo cohort lives in the database.

```sql
CREATE TABLE patients (
  patient_id TEXT PRIMARY KEY,
  display_name TEXT,                 -- fictional name for the demo
  source TEXT NOT NULL,              -- real_open or synthetic
  age INTEGER, sex TEXT, bmi REAL,
  hba1c REAL, fasting_glucose REAL, egfr REAL, systolic_bp REAL,
  years_since_dx REAL,
  family_history INTEGER, prs_z REAL,       -- prs_z is synthetic
  created_at TIMESTAMP
);

CREATE TABLE conditions (
  id INTEGER PRIMARY KEY, patient_id TEXT REFERENCES patients(patient_id),
  name TEXT, onset_date DATE
);

CREATE TABLE medications (
  id INTEGER PRIMARY KEY, patient_id TEXT REFERENCES patients(patient_id),
  drug_class TEXT, drug_name TEXT, start_date DATE
);

CREATE TABLE timeseries (
  patient_id TEXT REFERENCES patients(patient_id),
  ts TIMESTAMP,
  glucose_mgdl REAL, hr_bpm REAL, steps INTEGER, sleep_stage TEXT, carbs_g REAL,
  PRIMARY KEY (patient_id, ts)
);

CREATE TABLE predictions (
  id INTEGER PRIMARY KEY, patient_id TEXT, at TIMESTAMP,
  event_probability REAL, severity TEXT, forecast_json TEXT,
  model_version TEXT, personalized INTEGER
);

CREATE TABLE alerts (
  id INTEGER PRIMARY KEY, patient_id TEXT, started_at TIMESTAMP,
  ended_at TIMESTAMP, severity TEXT, peak_probability REAL,
  event_occurred INTEGER            -- filled when the outcome is known
);

CREATE TABLE twin_state (
  patient_id TEXT PRIMARY KEY, personalized INTEGER, days_of_data REAL,
  adapter_blob BLOB, updated_at TIMESTAMP,
  last_rmse_before REAL, last_rmse_after REAL
);
```

## 8. Frontend

### 8.1 Pages and components

| Page | Components |
|---|---|
| Patient list (`/`) | `PatientTable`, `RiskBadge`, `PrototypeBanner` |
| Patient detail (`/patients/:id`) | `PatientCard`, `GlucoseChart`, `RiskPanel`, `ExplanationPanel`, `WhatIfPanel`, `TwinStatePanel`, `DataQualityBar`, `ReplayControls` |
| About the model (`/about`) | `MetricsTable`, `CalibrationPlot`, `DataProvenance`, `Limitations` |

### 8.2 Implementation notes

- `usePatientStream(patientId)` opens the WebSocket, keeps a rolling buffer (last 24 hours plus the latest prediction), and exposes `play`, `pause`, `seek`, `setSpeed`, `injectMeal`.
- REST calls go through a typed client in `src/api/` with types matching `schemas.py` (generate TypeScript types from the OpenAPI document).
- `GlucoseChart` uses a Recharts `ComposedChart`: a line for history, a line for the P50 forecast, a shaded band between P10 and P90 (drawn with two stacked areas, the lower one transparent), and reference lines at 70 and 180 mg/dL.
- Risk colors: low (green), moderate (amber), high (red), always paired with a text label and an icon.
- Accessibility: keyboard focus order, ARIA labels, a text summary under the chart ("Glucose is 156 and rising, predicted to pass 180 in about 75 minutes"), contrast ratio at least 4.5 to 1.
- The prototype banner is part of the root layout and cannot be dismissed.
- Responsive down to tablet width. Phone layout is a stretch goal.

### 8.3 Frontend tests

Vitest and React Testing Library for `RiskBadge`, `ExplanationPanel`, and the stream hook (mock WebSocket). One end-to-end smoke test (optional Playwright): open a patient, press play, see a risk badge update.

## 9. DevOps

### 9.1 Docker Compose

| Service | Image | Ports | Notes |
|---|---|---|---|
| `api` | Python 3.11 slim, Uvicorn | 8000 | Mounts `models/current` and `data/demo` read-only |
| `web` | Node build stage, nginx serve stage | 8080 | Proxies `/api` to `api` |
| `trainer` | Same base as `api` | none | Optional profile for reproducing training |

### 9.2 Make targets

```
make setup       # create environment, install dependencies
make data        # download datasets, run Synthea, simulator, harmonize, build Parquet
make baseline    # train and evaluate LightGBM
make train       # train FusionTwinNet, calibrate, export to models/current
make evaluate    # run full evaluation, write reports/results.md
make demo-data   # build the small demo cohort for the database
make up          # docker compose up --build
make test        # run Python and frontend tests
make lint        # ruff and type checks
```

### 9.3 Configuration (`configs/default.yaml`)

```yaml
grid_minutes: 15
history_steps: 96
horizon_steps: 8
event:
  threshold_mgdl: 180
  min_run_steps: 3
exclude_already_high: true
model:
  gru_hidden: 128
  static_dim: 64
  decoder_hidden: 64
  dropout: 0.1
train:
  lr: 0.001
  weight_decay: 0.01
  batch_size: 256
  max_epochs: 40
  patience: 6
  seeds: [1, 2, 3]
  real_fraction_min: 0.3
alerts:
  precision_target: 0.6
  max_per_day: 6
  cooldown_minutes: 60
service:
  replay_default_speed: 900
  cache_size: 2048
```

### 9.4 Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GT_DB_URL` | `sqlite:///data/demo/demo.db` | Database |
| `GT_MODEL_DIR` | `models/current` | Model files |
| `GT_CORS_ORIGINS` | `http://localhost:8080` | Allowed origins |
| `GT_LOG_LEVEL` | `INFO` | Logging |
| `GT_TORCH_THREADS` | `4` | Inference threads |

### 9.5 CI

GitHub Actions workflow on push and pull request: install dependencies, run `ruff`, run `pytest` (with the small demo cohort), build the frontend, run Vitest. Cache pip and npm.

## 10. Testing plan

| Area | Tests |
|---|---|
| Labels | Event definition on hand-made glucose arrays, boundary cases (exactly 3 readings, already high), no use of values at or before `t` in targets |
| Leakage | Automated assertions for the checklist in section 4.8 |
| Simulator | Mean glucose near eAG, plausible time in range spread, post-meal peak timing |
| Harmonization | Unit conversion, gap filling limits, outlier removal |
| Model | Output shapes, quantiles never cross, deterministic output for a fixed seed, loading from `models/current` |
| What-if | Monotonic sanity rules, input validation errors |
| API | Contract tests against the OpenAPI schema, 404 and 422 handling, WebSocket control messages |
| Performance | Single prediction under 500 ms and what-if under 1 second on the target laptop (measured in a test with a generous margin) |
| Frontend | Component tests and one smoke test |
| Release | Fresh clone, `make up`, open the dashboard, run a replay, all within 10 minutes |

## 11. Performance budgets

| Operation | Budget (laptop CPU) |
|---|---|
| Single prediction | Under 500 ms |
| Prediction with explanation | Under 1.5 seconds |
| What-if (two passes) | Under 1 second |
| Replay tick processing at 900x | Under 800 ms per tick |
| Dashboard first load | Under 3 seconds |
| Training run (full) | Under 60 minutes on CPU, much less on GPU |

## 12. Security, privacy, and safety

### 12.1 Privacy and compliance

- Only synthetic or open anonymized data. No real identifiers are collected, stored, or logged.
- `DATA_PROVENANCE.md` lists every dataset, its source, license, and how it was used. The README links to it.
- The design is compatible with the principles of India's DPDP Act and HIPAA because it never handles personal data, but the project makes no compliance claim beyond that.
- If this were extended to real data, the architecture would need consent management, encryption at rest and in transit, access control, audit logs, and a data protection impact assessment. List this as future work.

### 12.2 Application security

- Pydantic validation on every input, with explicit bounds.
- CORS restricted to the dashboard origin.
- No secrets in the repository. `.env.example` only.
- Pinned dependencies and a dependency audit (`pip-audit`, `npm audit`) in CI.
- The PoC has no login. An optional static token for the API is a stretch item.

### 12.3 Clinical safety

- The prototype banner appears on every page.
- No medicine, dose, or diet recommendations anywhere in the output.
- What-if results are labeled as model estimates.
- `MODEL_CARD.md` documents intended use, training data, metrics, known failure modes (for example, unusual meals, sensor compression lows, patients unlike the training data), and out-of-scope uses.
- Out-of-distribution guard: if static features are outside the training range or more than 40 percent of the history window is missing, the dashboard shows "low confidence" and suppresses the alert.

## 13. Implementation plan

| Week | Tasks |
|---|---|
| 1 | Repository skeleton, CI, Docker base. Download and ingest datasets. Canonical schema, labeling, splits, leakage tests. LightGBM baseline and first metrics. |
| 2 | Run Synthea and post-process. Build the simulator and its calibration checks. Merge cohorts. Exploratory notebook. |
| 3 | `FusionTwinNet`, training loop, calibration, threshold and alert policy. Ablations. Integrated Gradients grouping. |
| 4 | FastAPI service, registry, predict, explain, WebSocket replay. React dashboard, chart, risk and explanation panels. What-if and personalization. |
| 5 | Full evaluation report, model card, README, architecture PDF, slides, 20-minute video recording, clean-clone test, license check, final submission. |

Cut order if time runs short: hybrid physiological model, hypoglycemia, alert history page, TCN and FiLM ablations, personalization, then what-if. Do not cut the baseline, the fused model, the dashboard, or the documentation.

## 14. Technical decisions

| ID | Decision | Reason | Alternative |
|---|---|---|---|
| D1 | 15-minute canonical grid | Works across sensor types, short sequences train fast | 5-minute grid for finer horizons |
| D2 | GRU encoder as default | Small data, stable training, fast CPU inference | TCN or transformer (kept as ablations) |
| D3 | Quantile outputs | Gives an honest uncertainty band | Gaussian output or ensembles |
| D4 | Static features initialize the temporal state | Makes fusion deep, not only at the end | Late fusion only |
| D5 | Known-future covariates with masking | Enables what-if without a second model | Separate simulator model |
| D6 | Integrated Gradients with grouping | Readable explanations for a clinician | Raw SHAP values per feature |
| D7 | Replay stream instead of real wearable hardware | Reproducible demo, no hardware risk | Phone app with Health Connect (future) |
| D8 | SQLite for the demo | Zero setup | PostgreSQL |
| D9 | Real open subjects as the final test | Guards against learning only the simulator | Random split of mixed data (rejected) |
| D10 | Hybrid physiological model as a stretch | High effort, uncertain gain | Include early (rejected for time) |

## 15. Appendix: reproducibility checklist

- [ ] Fixed seeds in data generation, splits, and training
- [ ] Versions pinned in `pyproject.toml` and `package-lock.json`
- [ ] Data hash stored in `meta.json`
- [ ] One command each for data, train, evaluate, serve
- [ ] Results tables regenerated by `make evaluate`, never typed by hand
- [ ] Dataset licenses recorded and checked
- [ ] Demo works without internet access
