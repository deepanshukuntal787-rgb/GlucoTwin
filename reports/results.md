# GlucoTwin Model Evaluation & Benchmark Report

**Model:** `FusionTwinNet` (Multimodal Probabilistic Digital Twin)  
**Evaluation Date:** 2026-10-23  
**Cohort:** 100 Patients (70 Train, 15 Validation, 15 Held-Out Test, strictly zero patient leakage)  
**Total Evaluation Windows:** 21,225 (Held-Out Test Set)  

---

## 1. Executive Summary

| Metric | Target | FusionTwinNet | LightGBM Baseline | Status |
|---|---|---|---|---|
| **Spike Event AUROC (2h)** | >= 0.85 | **0.9978** | 0.9648 | Target Exceeded |
| **Spike Event AUPRC** | > 0.15 | **0.8544** | 0.1863 | Target Exceeded |
| **Horizon RMSE @ +30 min** | < 15.0 mg/dL | **11.98 mg/dL** | 10.68 mg/dL | Target Exceeded |
| **Horizon RMSE @ +60 min** | < 18.0 mg/dL | **12.54 mg/dL** | 13.50 mg/dL | Target Exceeded |
| **Horizon RMSE @ +120 min** | < 25.0 mg/dL | **12.59 mg/dL** | 14.12 mg/dL | Target Exceeded |
| **Expected Calibration Error (ECE)** | < 0.05 | **0.0009** | - | Calibrated |
| **Brier Score** | < 0.05 | **0.0039** | - | High Reliability |
| **Average Early Warning Lead Time** | > 45 min | **45.3 min** | - | Clinically Actionable |

---

## 2. Event Prediction Performance (Hyperglycemia Spikes > 180 mg/dL within 120 min)

*Note: In accordance with clinical trial and TRD guidelines, already-hyperglycemic samples (G_t > 180 mg/dL) are excluded from classification metrics to evaluate true early-warning detection capability.*

* **AUROC:** `0.9978`
* **AUPRC:** `0.8544` (vs. test prevalence `1.24%`)
* **Alert Decision Threshold:** `0.096`
* **Sensitivity (Recall):** `93.49%`
* **Specificity:** `99.08%`
* **Positive Predictive Value (Precision):** `56.09%`
* **Median Lead Time:** `45 minutes`

---

## 3. Continuous Trajectory Forecasting (8 Steps / 120 Minutes)

| Horizon Offset | Time (+Min) | RMSE (mg/dL) | MAE (mg/dL) | [P10, P90] Empirical Coverage |
|---|---|---|---|---|
| Step 1 | +15 min | 12.56 | 9.71 | 79.6% |
| Step 2 | +30 min | 11.98 | 9.45 | 77.1% |
| Step 3 | +45 min | 12.29 | 9.65 | 75.0% |
| Step 4 | +60 min | 12.54 | 9.81 | 74.2% |
| Step 5 | +75 min | 12.49 | 9.72 | 75.3% |
| Step 6 | +90 min | 12.33 | 9.66 | 76.5% |
| Step 7 | +105 min | 12.34 | 9.69 | 77.7% |
| Step 8 | +120 min | 12.59 | 9.82 | 79.1% |

---

## 4. What-If Counterfactual Monotonicity Checks

As specified in TRD Section 5.6:
* **Carbohydrate Ingestion Test:** Adding carbs does not decrease predicted glucose peak in **96.0%** of test windows (Target >= 90%).
* **Physical Activity Test:** Adding steps does not increase predicted glucose peak in **95.0%** of test windows (Target >= 90%).
* **Overall Monotonicity Compliance:** **PASS**

---

## 5. Personalization (Patient Digital Twin Adapter)

* **Baseline RMSE (Global Model):** `15.28 mg/dL`
* **Personalized RMSE (With Adapter b_p):** `14.72 mg/dL`
* **Error Reduction (Delta RMSE):** **`0.56 mg/dL`**

---

## 6. Artifact Inventory

* Checkpointed Weights: `models/current/model.pt`
* Isotonic Calibrator: `models/current/calibrator.pkl`
* Normalization Scaler: `models/current/scaler.json`
* Model Metadata Card: `models/current/meta.json`
* Tabular Baseline Models: `models/current/lgbm_classifier.joblib`, `models/current/lgbm_regressors.joblib`
