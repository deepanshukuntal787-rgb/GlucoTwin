# Product Requirements Document (PRD)

**Project:** GlucoTwin (working title, rename freely)
**Challenge:** Happiest Health, Reimagining and Reforming Healthcare in India Summit 2026, Digital Twin Challenge
**Document version:** 1.0
**Date:** 2026-10-02
**Status:** Draft for team review
**Team:** `<SuperNova>`, `<Manipal Intitute of Technology>`
**Team leader:** `<Deepanshu Kuntal>`

---

## 1. Summary

GlucoTwin is a proof-of-concept Digital Twin for adults with Type 2 Diabetes (T2D). It keeps a virtual copy of a patient that combines two kinds of data:

1. **Static data:** a simulated Electronic Health Record (demographics, diagnoses, lab results, medicines, a synthetic genetic risk marker).
2. **Dynamic data:** time-series from a continuous glucose monitor (CGM) and a consumer wearable (heart rate, steps, sleep).

The twin predicts a **hyperglycemic event up to 2 hours before it happens**, forecasts the glucose curve for the next 30, 60, and 120 minutes, explains why the alert fired, and lets a doctor run **what-if scenarios** (for example, "what if the patient walks for 20 minutes after dinner?") on a web dashboard.

All data is synthetic, anonymized, or open-source. No real patient data is used.

## 2. Background and context

### 2.1 The challenge

The challenge asks student teams to build a proof-of-concept Digital Twin focused on one specific, localized health outcome. The brief requires:

- One chronic or lifestyle condition prevalent in India (Type 2 Diabetes, Hypertension, or Cardiovascular risk).
- Fusion of **static/historical** data (simulated EHR) and **dynamic/real-time** data (wearable or IoT time-series).
- A working algorithmic model that ingests both streams and predicts an adverse event.
- A conceptual UI dashboard showing how a doctor would interact with the virtual patient.
- Only anonymized, open-source, or synthetic data (the DPDP Act and HIPAA are cited as the reason).

### 2.2 Why Type 2 Diabetes and glucose prediction

| Reason | Detail |
|---|---|
| Relevance to India | India has one of the largest populations of people with diabetes in the world, so the use case is easy to justify to a healthcare audience. |
| Data availability | Several open CGM datasets exist, and glucose is a dense, continuous signal that is easy to evaluate. |
| Direct fit with the brief | The brief itself gives "predicting a sudden glucose spike 2 hours in advance" as an example. |
| Clear clinical action | A doctor or patient can act on a 2-hour warning (diet, activity, medication timing review). |
| Demo friendliness | A live glucose curve with a predicted band is visual and easy to explain in a 20-minute video. |

## 3. Problem statement

People with T2D usually learn about a glucose spike only after it has happened, through a fingerstick, a CGM alarm that fires at the threshold, or a lab result weeks later. Doctors see a few data points per visit and cannot see how a patient's sleep, activity, meals, and background risk interact day to day.

**Problem:** There is no simple, explainable way for a clinician to see a patient's likely glucose future, understand the reasons behind it, and test changes before they happen.

**Opportunity:** A Digital Twin that fuses the patient's record with live sensor data can move care from reactive ("the glucose is high") to proactive ("the glucose is likely to be high in 90 minutes, mainly because of this late high-carbohydrate meal and a poor night of sleep").

## 4. Goals and non-goals

### 4.1 Goals

| ID | Goal |
|---|---|
| G1 | Demonstrate real fusion of static EHR features and dynamic sensor time-series inside one model. |
| G2 | Predict a hyperglycemic event up to 120 minutes ahead with clearly reported accuracy and warning lead time. |
| G3 | Make the twin feel like a living patient: per-patient state, personalization, and what-if simulation. |
| G4 | Give a doctor an explainable, clear dashboard. |
| G5 | Meet every submission requirement of the challenge with a clean, reproducible public repository. |
| G6 | Be honest about synthetic data, limits, and safety (this is a research prototype, not a medical device). |

### 4.2 Non-goals

- Whole-body or multi-organ twin.
- Real patient data, clinical validation, or regulatory approval.
- Medication dosing advice or any automated treatment decision.
- A patient-facing mobile app (a doctor-facing web dashboard only).
- Real hardware integration with a physical wearable (the stream is simulated by replaying data).
- Type 1 Diabetes insulin-dosing control loops.

## 5. Users and personas

### 5.1 Primary persona: Dr. Meera Iyer, diabetologist (fictional)

- Sees 25 to 40 patients a day in a city clinic.
- Has a few minutes per patient.
- Wants to know quickly: who is at risk now, why, and what could change it.
- Trusts a tool only if it explains itself and shows uncertainty.

### 5.2 Secondary persona: the hackathon jury

- Technical and medical experts scoring in a limited time.
- Wants to see a working prototype, honest evaluation, clear documentation, and real-world impact.
- Will read the README first and watch the demo video second.

### 5.3 Out of scope persona: the patient

Patient-facing features (notifications, coaching) are listed only as future work.

## 6. User stories

Priority key: **P0** must have for submission, **P1** should have, **P2** nice to have.

| ID | As a... | I want to... | So that... | Priority |
|---|---|---|---|---|
| US-1 | Doctor | see a list of my virtual patients sorted by current risk | I know who needs attention first | P0 |
| US-2 | Doctor | open a patient and see a summary card (age, BMI, HbA1c, medicines, comorbidities) | I understand the background in seconds | P0 |
| US-3 | Doctor | see a live glucose trace with the predicted next 2 hours and an uncertainty band | I can see where the glucose is heading | P0 |
| US-4 | Doctor | get a clear risk alert with severity and expected time to event | I can act before the spike | P0 |
| US-5 | Doctor | see the top reasons behind an alert in plain language | I can trust and verify the prediction | P0 |
| US-6 | Doctor | change a future input (add a meal, add a 20-minute walk, skip dinner) and see the forecast change | I can test advice before giving it | P1 |
| US-7 | Doctor | see how the twin adapted to this specific patient over time | I know it is personalized and not generic | P1 |
| US-8 | Doctor | see data quality indicators (missing sensor data, last update time) | I know when to doubt the prediction | P1 |
| US-9 | Doctor | see a log of past alerts and whether they were correct | I can judge the model on this patient | P2 |
| US-10 | Jury member | start a replay of a patient day at accelerated speed | I can watch the twin work live in a short demo | P0 |
| US-11 | Jury member | read model metrics and the data provenance statement inside the app | I can judge honesty and rigor quickly | P1 |

## 7. Functional requirements

### 7.1 Data and ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-D1 | The system shall load a cohort of synthetic EHR records (demographics, diagnoses, labs, medications, synthetic genetic risk marker). | P0 |
| FR-D2 | The system shall load glucose and wearable time-series (glucose, heart rate, steps, sleep stage, meals) for each patient. | P0 |
| FR-D3 | The system shall resample all time-series to one canonical time grid (default 15 minutes, configurable) with explicit missing-data flags. | P0 |
| FR-D4 | The system shall support a synthetic generator that produces glucose and wearable traces whose physiology depends on the patient's static profile. | P0 |
| FR-D5 | The system shall store data provenance (source, license, synthetic or real-open) for every dataset. | P1 |

### 7.2 Prediction

| ID | Requirement | Priority |
|---|---|---|
| FR-P1 | The model shall ingest a history window (default 24 hours) of dynamic data and a static feature vector in one forward pass. | P0 |
| FR-P2 | The model shall output glucose forecasts at +30, +60, and +120 minutes, with a lower and upper bound (P10 and P90). | P0 |
| FR-P3 | The model shall output the probability of a hyperglycemic event beginning within the next 120 minutes. | P0 |
| FR-P4 | The system shall convert probability to an alert with severity (Low, Moderate, High) and an estimated time to event. | P0 |
| FR-P5 | The system shall include a gradient boosting baseline for comparison. | P0 |
| FR-P6 | The system shall provide feature attributions grouped into readable factors (recent glucose trend, last meal, sleep, heart rate, activity, background risk). | P0 |
| FR-P7 | The system shall support per-patient personalization from that patient's early data. | P1 |
| FR-P8 | The system shall support what-if scenarios that edit future inputs and re-run the forecast. | P1 |
| FR-P9 | The system shall apply alert cool-down logic to prevent repeated alerts for the same event. | P1 |

### 7.3 Dashboard

| ID | Requirement | Priority |
|---|---|---|
| FR-U1 | A patient list view with name or ID, current glucose, current risk level, and last update time. | P0 |
| FR-U2 | A patient detail view with summary card, live chart, risk panel, and explanation panel. | P0 |
| FR-U3 | A glucose chart showing history, forecast, uncertainty band, and the 70 and 180 mg/dL reference lines. | P0 |
| FR-U4 | A replay control (play, pause, speed) that simulates the live stream. | P0 |
| FR-U5 | A what-if panel with controls for meals and activity and a visible before/after forecast. | P1 |
| FR-U6 | A twin state panel showing personalization status and data completeness. | P1 |
| FR-U7 | A persistent banner stating that the app is a research prototype using synthetic data and is not for clinical use. | P0 |
| FR-U8 | An "About the model" page with metrics, data sources, and limitations. | P1 |

### 7.4 Platform

| ID | Requirement | Priority |
|---|---|---|
| FR-S1 | The backend shall expose a documented REST API and a WebSocket stream. | P0 |
| FR-S2 | The full system shall start with one command (Docker Compose). | P0 |
| FR-S3 | The repository shall contain tests for data labeling, the API, and model inference. | P1 |

## 8. Definition of the predicted event

| Item | Definition |
|---|---|
| Hyperglycemic event | At least 3 consecutive CGM readings (at the 15-minute grid, 45 minutes) above 180 mg/dL. Configurable. |
| Prediction target | The event **starts** within the next 120 minutes. |
| Exclusion | Samples where the patient is already above 180 mg/dL at prediction time are excluded from the classification metric, so the model cannot score well by only reading the current value. |
| Reference range | 70 to 180 mg/dL is the commonly used time-in-range band. |
| Secondary target (P2) | Hypoglycemia below 70 mg/dL. |

## 9. Model requirements (summary)

Full detail is in `TRD.md`.

| Metric | Initial target | Notes |
|---|---|---|
| AUROC (event within 120 min) | 0.85 or higher | Patient-wise held-out test. Revise after the baseline result. |
| AUPRC | Clearly above event prevalence | Report prevalence next to it. |
| RMSE at +30, +60, +120 min | 15, 25, 35 mg/dL or lower | Targets are starting points, not promises. |
| Median warning lead time | 60 minutes or more | Measured only on true events that were alerted. |
| Alerts per patient per day | Fewer than 6 at the chosen threshold | Prevents alert fatigue. |
| Calibration | Expected calibration error below 0.08 | Risk percentages must be believable. |
| Neural model vs baseline | Must be reported honestly | If the baseline wins, say so and explain. |

## 10. UX requirements

### 10.1 Design principles

1. **Glanceable:** the most important fact (risk now, time to event) is visible without scrolling.
2. **Explainable:** every alert is next to its reasons.
3. **Honest about uncertainty:** forecasts always show a band, never a single line.
4. **Calm:** colour is reserved for risk. Avoid alarm fatigue visuals.
5. **Accessible:** colour is never the only signal, text contrast meets WCAG AA, and charts have text summaries.

### 10.2 Patient detail wireframe

```
+--------------------------------------------------------------------------+
| PROTOTYPE BANNER: synthetic data, research use only, not medical advice  |
+--------------------------------------------------------------------------+
| Patient card                    | Risk panel                             |
| P-0042, 54 y, F                 |  HIGH risk in ~95 min   (prob 0.78)    |
| BMI 29.4  HbA1c 8.1%            |  Expected peak: 238 mg/dL              |
| Metformin, Glimepiride          |  Why: late high-carb dinner (42%),     |
| Hypertension                    |  poor sleep last night (23%),          |
| Twin: personalized (6 days)     |  rising trend (18%), background (17%)  |
+--------------------------------------------------------------------------+
| Glucose chart (mg/dL)                                                    |
|   history ----------\            forecast band  ..........              |
|                      \_______/\ .'''''''''''''''.                        |
|   180 line ------------------------------------------ ... ---           |
|   70 line  ------------------------------------------------------       |
|   [Play] [Pause] [Speed 60x]   now: 19:40                                |
+--------------------------------------------------------------------------+
| What-if panel                                                            |
| Add meal: [0 g carbs]   Walk: [20 min at 19:55]   [Run scenario]         |
| Result: peak 238 -> 201 mg/dL, event probability 0.78 -> 0.41            |
+--------------------------------------------------------------------------+
| Data quality: CGM 99% | HR 94% | Sleep 100%    Last update: 12 s ago     |
+--------------------------------------------------------------------------+
```

### 10.3 Risk levels

| Level | Rule (starting values, tuned on validation data) |
|---|---|
| Low | Probability below 0.30 |
| Moderate | 0.30 to 0.60 |
| High | Above 0.60, or a forecast above 250 mg/dL within 60 minutes |

## 11. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | Single prediction under 500 ms on a laptop CPU. What-if run under 1 second. Dashboard chart updates at least once per second during replay. |
| Reliability | The demo must run offline with no external API calls. Pre-computed fallbacks for the demo patients. |
| Reproducibility | Fixed random seeds, pinned dependencies, one script to rebuild data, train, and evaluate. |
| Privacy | Only synthetic or open anonymized data. No personal identifiers anywhere in the repository. |
| Safety | Banner on every screen, no treatment recommendations, limits listed in the README and the About page. |
| Portability | Runs on Windows, macOS, and Linux through Docker. |
| Maintainability | Typed Python (type hints), linted code, modular folders, short docstrings on public functions. |
| Licensing | Code under the MIT license. Every dataset license recorded and respected. |

## 12. Success metrics

### 12.1 Technical

- Patient-wise test AUROC and AUPRC reported with confidence intervals.
- RMSE at each horizon, reported against a naive "last value" baseline and the LightGBM baseline.
- Calibration plot and median lead time.
- Ablation: static only, dynamic only, and fused models, to prove that fusion adds value.

### 12.2 Product

- A first-time viewer understands the risk and the reason within 30 seconds of opening a patient.
- A full demo can be run from a clean clone in under 10 minutes.

### 12.3 Hackathon

- Pass Phase 1 screening (top 10), then reach the top 5 for the summit on 20 November 2026.

## 13. Scope

### 13.1 MVP (must be done for submission)

All P0 items: synthetic cohort, fused model, baseline, alerts, explanations, dashboard with replay, README, video, diagram, slides, license.

### 13.2 Stretch

- What-if simulator and per-patient personalization (P1, strongly recommended because they make it a twin).
- Hybrid physiological model (Bergman minimal model) blended with the learned model.
- Hypoglycemia as a second event.
- Alert history with outcome tracking.
- Hindi or Kannada labels in the dashboard.

## 14. Assumptions, constraints, dependencies

### 14.1 Assumptions

- Open CGM datasets can be downloaded before development starts (verify access and license for each).
- Team of 1 to 4 students, each with part-time availability.
- A laptop with at least 16 GB RAM is available. A GPU is helpful but not required (the model is small).

### 14.2 Constraints

- Synthetic and open data only.
- Public GitHub repository, folder named `Team Name_College Name`.
- Video of at least 20 minutes.
- The Phase 1 submission deadline is **not stated in the brief provided**. Confirm it on the registration page and adjust the plan in section 16.

### 14.3 Dependencies

| Dependency | Risk if unavailable | Fallback |
|---|---|---|
| Synthea (Java 11 or newer) | Cannot generate EHRs | Use the dataset clinical tables plus a Python EHR generator |
| PhysioNet datasets (some need a free account and agreement) | Delay in access | Use the synthetic generator as the main training source |
| Docker | Setup issues on a team machine | Provide a plain `make` and virtual environment path |

## 15. Risks and mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Real open datasets have no linked EHR, so "fusion" looks artificial | High | High | Use dataset demographics and lab tables as static features where they exist, and build a synthetic cohort where static traits drive the simulated physiology. State this clearly. |
| R2 | Circular logic: the model learns the simulator, not real physiology | Medium | High | Always test on held-out real open CGM subjects. Report simulator-only and real-only results separately. |
| R3 | Small real datasets cause overfitting | High | Medium | Patient-wise cross-validation, small model, early stopping, data augmentation, confidence intervals. |
| R4 | Neural model does not beat LightGBM | Medium | Medium | Report honestly. Differentiate with fusion, uncertainty, personalization, and what-if. |
| R5 | Too much scope for the time | High | High | Build in vertical slices. Baseline and dashboard first, extras later. |
| R6 | Alert fatigue or poor calibration | Medium | Medium | Threshold tuning, cool-down, calibration step. |
| R7 | Live demo fails | Medium | High | Replay mode from local files, recorded backup video, pre-computed predictions. |
| R8 | Claims sound like medical advice | Low | High | Banner, wording review, no dosing, listed limitations. |
| R9 | Synthea defaults are US-based, so records are not Indian | High | Low | Post-process distributions (age, BMI, medicines common in India), document the limitation. |

## 16. Timeline (assumed 5 weeks, compress if the deadline is earlier)

| Week | Focus | Exit criteria |
|---|---|---|
| 1 | Setup, data access, ingestion, canonical schema, labels, LightGBM baseline | Baseline AUROC on a patient-wise split, repository skeleton, Docker running |
| 2 | Synthetic generator, Synthea cohort, merged dataset, EDA | Cohort of 300 or more virtual patients with 30 or more days each |
| 3 | Neural fusion model, calibration, ablations, explanations | Model beats or matches baseline, or has an honest write-up |
| 4 | FastAPI service, WebSocket replay, React dashboard, what-if, personalization | End-to-end demo working locally |
| 5 | Evaluation report, polish, README, architecture PDF, slides, 20-minute video, license check, final test from a clean clone | Submission package complete |

### Suggested team roles (adjust to team size)

| Role | Responsibilities |
|---|---|
| ML lead | Data labeling, models, evaluation, explanations |
| Data engineer | Ingestion, Synthea, synthetic generator, schema |
| Full-stack developer | FastAPI, WebSocket, React dashboard |
| Documentation and demo lead | README, diagrams, slides, video script, clinical wording review |

A one or two person team should merge roles and drop P2 items first, then the hybrid physiological model.

## 17. Submission deliverables checklist

| # | Deliverable | Owner | Done |
|---|---|---|---|
| 1 | Public GitHub repository, folder named `Team Name_College Name` | Leader | [ ] |
| 2 | README with team and college details | Docs | [ ] |
| 3 | README: project title | Docs | [ ] |
| 4 | README: problem statement and healthcare use case | Docs | [ ] |
| 5 | README: technical stack, AI/ML model and framework details | ML lead | [ ] |
| 6 | Video of at least 20 minutes (slides plus live demo, PowerPoint recording or unlisted YouTube) | Docs | [ ] |
| 7 | Open-source license details (MIT for code, dataset licenses listed) | Leader | [ ] |
| 8 | Architecture diagram in PDF or PPT | Docs | [ ] |
| 9 | Presentation in PDF or PPT with project details and outcome | Docs | [ ] |
| 10 | All links in the README open without login | Leader | [ ] |
| 11 | Submission platform entry: leader name, phone, email, and public repository link only | Leader | [ ] |
| 12 | Team details confirmed by every member (team cannot be changed after registration) | All | [ ] |

## 18. Mapping to likely evaluation criteria

The brief does not publish scoring weights. The table below shows how each likely criterion is covered. Treat it as an assumption.

| Likely criterion | How this project addresses it |
|---|---|
| Fusion of static and dynamic data | Two-branch model, plus an ablation that proves the benefit |
| Predictive quality and lead time | Patient-wise evaluation, AUROC, AUPRC, RMSE, median lead time |
| Digital Twin quality | Per-patient state, personalization, what-if simulation |
| Clinical relevance and impact | India-relevant T2D use case, doctor workflow, explanations |
| UI and usability | Dashboard designed for a 30-second read |
| Data ethics and compliance | Synthetic and open data only, provenance table, safety banner |
| Documentation and reproducibility | README, Docker, tests, pinned versions, scripts |
| Presentation | 20-minute video with live demo, clear diagram and slides |

## 19. Open questions

1. What is the exact Phase 1 submission deadline?
2. Is the video required to contain a live demo, or can it be fully slide based? (Plan for a live demo to be safe.)
3. Are pre-trained models or external APIs allowed? (This plan avoids external APIs.)
4. Are there rules on using public datasets that require a signed data use agreement?
5. How many team members will be available, and for how many hours a week?

## 20. Glossary

| Term | Meaning |
|---|---|
| Digital Twin | A virtual replica of a patient, updated with real-time data, used to simulate and predict |
| EHR | Electronic Health Record |
| CGM | Continuous Glucose Monitor |
| T2D | Type 2 Diabetes |
| HbA1c | A blood test reflecting average glucose over about three months |
| TIR | Time in range, the share of time glucose is between 70 and 180 mg/dL |
| Synthea | Open-source synthetic patient generator |
| LSTM, TCN | Neural network types for time-series (Long Short-Term Memory, Temporal Convolutional Network) |
| SHAP | A method that explains how much each feature contributed to a prediction |
| DPDP Act | India's Digital Personal Data Protection Act |
| PoC | Proof of concept |
