"""Data preprocessing and feature engineering pipeline for GlucoTwin."""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from gluco_twin.config import cfg, RAW_DATA_DIR, CURRENT_MODEL_DIR, PROCESSED_DATA_DIR

def parse_static_features(static_path=None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    """Parse static EHR features from dataset2.csv into a 21-dimensional vector per patient."""
    if static_path is None:
        static_path = RAW_DATA_DIR / "dataset2.csv"
    
    df = pd.read_csv(static_path)
    
    # Fill NAs
    df["medications"] = df["medications"].fillna("None")
    df["comorbidities"] = df["comorbidities"].fillna("None")
    
    # 1. Demographics & Continuous
    features = pd.DataFrame()
    features["patient_id"] = df["patient_id"]
    features["age"] = df["age"].astype(float)
    features["sex_female"] = (df["sex"].str.upper() == "F").astype(float)
    features["bmi"] = df["bmi"].astype(float)
    features["hba1c"] = df["hba1c_pct"].astype(float)
    features["diabetes_type_2"] = (df["diabetes_type"].str.contains("2")).astype(float)
    features["duration_years"] = df["diabetes_duration_years"].astype(float)
    
    # 2. Medications (binary)
    meds_lower = df["medications"].str.lower()
    features["med_metformin"] = meds_lower.str.contains("metformin").astype(float)
    features["med_sulfonylurea"] = meds_lower.str.contains("sulfonylurea").astype(float)
    features["med_dpp4"] = meds_lower.str.contains("dpp4").astype(float)
    features["med_sglt2"] = meds_lower.str.contains("sglt2").astype(float)
    features["med_insulin"] = meds_lower.str.contains("insulin").astype(float)
    features["med_statin"] = meds_lower.str.contains("statin").astype(float)
    features["med_ace_inhibitor"] = meds_lower.str.contains("ace").astype(float)
    
    # 3. Comorbidities (binary)
    comorb_lower = df["comorbidities"].str.lower()
    features["comorb_htn"] = comorb_lower.str.contains("htn").astype(float)
    features["comorb_dyslipidemia"] = comorb_lower.str.contains("dyslipidemia").astype(float)
    features["comorb_ckd"] = comorb_lower.str.contains("ckd").astype(float)
    features["comorb_neuropathy"] = comorb_lower.str.contains("neuropathy").astype(float)
    features["comorb_obesity"] = comorb_lower.str.contains("obesity").astype(float)
    features["comorb_cad"] = comorb_lower.str.contains("cad").astype(float)
    
    # 4. Estimated clinical risk markers
    # eAG (estimated average glucose) formula: 28.7 * A1c - 46.7
    features["est_avg_glucose"] = 28.7 * features["hba1c"] - 46.7
    # PRS risk score z-score approximation
    features["prs_z"] = ((features["duration_years"] / 10.0) + (features["hba1c"] - 7.0) / 1.5).clip(-3.0, 3.0)
    
    feature_cols = [c for c in features.columns if c != "patient_id"]
    assert len(feature_cols) == cfg.static_dim, f"Expected {cfg.static_dim} static features, got {len(feature_cols)}"
    
    patient_map = {}
    for _, row in features.iterrows():
        patient_map[row["patient_id"]] = row[feature_cols].values.astype(np.float32)
        
    return features, patient_map, feature_cols


def build_patient_splits(patients: List[str], seed=42) -> Tuple[List[str], List[str], List[str]]:
    """Create strict patient-grouped splits (70 train, 15 val, 15 test)."""
    rng = np.random.RandomState(seed)
    shuffled = sorted(patients)
    rng.shuffle(shuffled)
    
    train_pts = shuffled[:cfg.train_patients]
    val_pts = shuffled[cfg.train_patients : cfg.train_patients + cfg.val_patients]
    test_pts = shuffled[cfg.train_patients + cfg.val_patients :]
    
    assert len(set(train_pts).intersection(set(val_pts))) == 0, "Patient leakage between train and val!"
    assert len(set(train_pts).intersection(set(test_pts))) == 0, "Patient leakage between train and test!"
    assert len(set(val_pts).intersection(set(test_pts))) == 0, "Patient leakage between val and test!"
    
    return train_pts, val_pts, test_pts


def process_dynamic_patient_series(patient_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a single patient's continuous 15-min time-series into arrays:
    - dynamic_channels (N, 16)
    - future_covariates (N, 5)
    - raw_glucose (N,)
    """
    patient_df = patient_df.sort_values("timestamp").reset_index(drop=True)
    n = len(patient_df)
    
    glucose = patient_df["glucose"].values.astype(np.float32)
    hr = patient_df["heart_rate"].values.astype(np.float32)
    steps = patient_df["steps"].values.astype(np.float32)
    sleep = patient_df["sleep_stage"].values.astype(np.int32)
    carbs = patient_df["carbs_g"].values.astype(np.float32)
    
    # Timestamps to minute of day
    ts = pd.to_datetime(patient_df["timestamp"])
    minute_of_day = (ts.dt.hour * 60 + ts.dt.minute).values.astype(np.float32)
    tod_sin = np.sin(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)
    tod_cos = np.cos(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)
    
    # Glucose diff
    g_diff = np.zeros(n, dtype=np.float32)
    g_diff[1:] = (glucose[1:] - glucose[:-1]) / 15.0
    
    # Time since last meal (>0 carbs)
    since_meal = np.zeros(n, dtype=np.float32)
    last_meal_idx = -9999
    for i in range(n):
        if carbs[i] > 0:
            last_meal_idx = i
        since_meal[i] = min(360.0, (i - last_meal_idx) * 15.0)
    since_meal_z = (since_meal - 180.0) / 100.0
    
    # Normalized channels
    glucose_z = (glucose - cfg.glucose_mean) / cfg.glucose_std
    hr_z = (hr - 72.0) / 11.0
    steps_z = (np.log1p(steps) - 1.5) / 2.0
    carbs_z = carbs / 20.0
    
    # Sleep one-hot (4 channels: awake=0, light=[1,2], deep=3, rem=4)
    sleep_awake = (sleep == 0).astype(np.float32)
    sleep_light = ((sleep == 1) | (sleep == 2)).astype(np.float32)
    sleep_deep = (sleep == 3).astype(np.float32)
    sleep_rem = (sleep == 4).astype(np.float32)
    
    # Observation masks
    mask_ones = np.ones(n, dtype=np.float32)
    
    # Stack dynamic (N, 16)
    dynamic_channels = np.stack([
        glucose_z,        # 0
        g_diff,           # 1
        hr_z,             # 2
        steps_z,          # 3
        sleep_awake,      # 4
        sleep_light,      # 5
        sleep_deep,       # 6
        sleep_rem,        # 7
        carbs_z,          # 8
        since_meal_z,     # 9
        tod_sin,          # 10
        tod_cos,          # 11
        mask_ones,        # 12: mask_glucose
        mask_ones,        # 13: mask_hr
        mask_ones,        # 14: mask_steps
        mask_ones,        # 15: mask_sleep
    ], axis=1).astype(np.float32)
    
    # Future covariates (N, 5): plan_carbs_z, plan_steps_z, tod_sin, tod_cos, cov_mask
    future_covs = np.stack([
        carbs_z,
        steps_z,
        tod_sin,
        tod_cos,
        mask_ones,
    ], axis=1).astype(np.float32)
    
    return dynamic_channels, future_covs, glucose


def extract_sliding_windows(
    dynamic_channels: np.ndarray,
    future_covs: np.ndarray,
    raw_glucose: np.ndarray,
    static_vec: np.ndarray,
    patient_id: str,
    stride: int = 2,
) -> Dict[str, np.ndarray]:
    """Extract history window (96 steps) and horizon (8 steps) samples."""
    n = len(raw_glucose)
    h_steps = cfg.history_steps  # 96
    f_steps = cfg.horizon_steps  # 8
    thresh = cfg.event_threshold # 180.0
    min_run = cfg.event_min_run  # 3
    
    histories = []
    covariates = []
    statics = []
    targets_p = []
    events = []
    already_highs = []
    sample_pts = []
    raw_futures = []
    
    # Valid t index range: [h_steps - 1, n - f_steps - 1]
    for t in range(h_steps - 1, n - f_steps, stride):
        # History window: [t - 95 : t + 1] -> 96 steps
        hist = dynamic_channels[t - h_steps + 1 : t + 1]
        # Future covariates: [t + 1 : t + 1 + 8] -> 8 steps
        cov = future_covs[t + 1 : t + 1 + f_steps]
        # Future raw glucose
        future_g = raw_glucose[t + 1 : t + 1 + f_steps]
        
        # Target scaled path
        target_path = (future_g - cfg.glucose_mean) / cfg.glucose_std
        
        # Event definition: any run of >= min_run consecutive readings > thresh
        above = (future_g > thresh)
        event = 0
        for i in range(len(above) - min_run + 1):
            if np.all(above[i : i + min_run]):
                event = 1
                break
                
        is_already_high = (raw_glucose[t] > thresh)
        
        histories.append(hist)
        covariates.append(cov)
        statics.append(static_vec)
        targets_p.append(target_path)
        events.append(event)
        already_highs.append(is_already_high)
        sample_pts.append(patient_id)
        raw_futures.append(future_g)
        
    return {
        "history": np.array(histories, dtype=np.float32),
        "covariates": np.array(covariates, dtype=np.float32),
        "static": np.array(statics, dtype=np.float32),
        "targets": np.array(targets_p, dtype=np.float32),
        "events": np.array(events, dtype=np.float32),
        "already_high": np.array(already_highs, dtype=bool),
        "patient_id": np.array(sample_pts),
        "raw_future": np.array(raw_futures, dtype=np.float32),
    }
