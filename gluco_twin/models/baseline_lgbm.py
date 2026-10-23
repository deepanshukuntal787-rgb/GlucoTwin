"""LightGBM Baseline Model for GlucoTwin (TRD Section 5.1)."""
import os
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from typing import Dict, Tuple
from gluco_twin.config import cfg, PROCESSED_DATA_DIR, CURRENT_MODEL_DIR

def extract_tabular_features(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract engineered tabular features from window history and static features:
    - Last glucose, deltas over 15, 30, 60 min (steps 1, 2, 4)
    - Rolling stats (mean, std, min, max) over 1h (4 steps), 3h (12 steps), 6h (24 steps), 24h (96 steps)
    - Time since meal, meal carbs
    - HR and steps rolling stats
    - Sleep fractions (deep, rem, light)
    - Time of day (sin, cos)
    - All 21 static features
    """
    data = np.load(npz_path, allow_pickle=True)
    history = data["history"]      # (N, 96, 16)
    static = data["static"]        # (N, 21)
    targets = data["targets"]      # (N, 8) in scaled units
    raw_future = data["raw_future"]# (N, 8) in mg/dL
    events = data["events"]        # (N,)
    already_high = data["already_high"] # (N,)

    # Un-normalize glucose from history[:, :, 0]
    glucose_hist = history[:, :, 0] * cfg.glucose_std + cfg.glucose_mean  # (N, 96)
    hr_hist = history[:, :, 2] * 11.0 + 72.0                              # (N, 96)
    steps_hist = np.expm1(history[:, :, 3] * 2.0 + 1.5)                  # (N, 96)

    # Engineered features
    g_last = glucose_hist[:, -1]
    g_delta_15 = g_last - glucose_hist[:, -2]
    g_delta_30 = g_last - glucose_hist[:, -3]
    g_delta_60 = g_last - glucose_hist[:, -5]

    # Rolling glucose
    g_mean_1h = glucose_hist[:, -4:].mean(axis=1)
    g_std_1h = glucose_hist[:, -4:].std(axis=1)
    g_min_1h = glucose_hist[:, -4:].min(axis=1)
    g_max_1h = glucose_hist[:, -4:].max(axis=1)

    g_mean_3h = glucose_hist[:, -12:].mean(axis=1)
    g_std_3h = glucose_hist[:, -12:].std(axis=1)
    g_min_3h = glucose_hist[:, -12:].min(axis=1)
    g_max_3h = glucose_hist[:, -12:].max(axis=1)

    g_mean_6h = glucose_hist[:, -24:].mean(axis=1)
    g_std_6h = glucose_hist[:, -24:].std(axis=1)

    g_mean_24h = glucose_hist.mean(axis=1)
    g_std_24h = glucose_hist.std(axis=1)
    g_min_24h = glucose_hist.min(axis=1)
    g_max_24h = glucose_hist.max(axis=1)

    # HR & steps stats
    hr_mean_1h = hr_hist[:, -4:].mean(axis=1)
    hr_mean_6h = hr_hist[:, -24:].mean(axis=1)
    steps_sum_1h = steps_hist[:, -4:].sum(axis=1)
    steps_sum_6h = steps_hist[:, -24:].sum(axis=1)

    # Sleep fractions over 24h
    deep_sleep_frac = history[:, :, 6].mean(axis=1)
    rem_sleep_frac = history[:, :, 7].mean(axis=1)
    total_sleep_frac = (history[:, :, 4] == 0).mean(axis=1)

    # Meals
    recent_carbs = history[:, -8:, 8].sum(axis=1) * 20.0
    since_meal = history[:, -1, 9] * 100.0 + 180.0

    # Time of day
    tod_sin = history[:, -1, 10]
    tod_cos = history[:, -1, 11]

    # Tabular feature matrix
    X = np.column_stack([
        g_last, g_delta_15, g_delta_30, g_delta_60,
        g_mean_1h, g_std_1h, g_min_1h, g_max_1h,
        g_mean_3h, g_std_3h, g_min_3h, g_max_3h,
        g_mean_6h, g_std_6h,
        g_mean_24h, g_std_24h, g_min_24h, g_max_24h,
        hr_mean_1h, hr_mean_6h,
        steps_sum_1h, steps_sum_6h,
        deep_sleep_frac, rem_sleep_frac, total_sleep_frac,
        recent_carbs, since_meal,
        tod_sin, tod_cos,
        static,
    ])

    return X, events, raw_future, already_high


def train_and_eval_lightgbm():
    """Train LightGBM event classifier and horizon regressors (+30m, +60m, +120m)."""
    print("=== Training LightGBM Baseline ===")
    
    X_train, y_train_event, y_train_future, train_ah = extract_tabular_features(PROCESSED_DATA_DIR / "train.npz")
    X_val, y_val_event, y_val_future, val_ah = extract_tabular_features(PROCESSED_DATA_DIR / "val.npz")
    X_test, y_test_event, y_test_future, test_ah = extract_tabular_features(PROCESSED_DATA_DIR / "test.npz")

    # Positive class weighting for event
    pos_weight = (len(y_train_event) - y_train_event.sum()) / (y_train_event.sum() + 1e-5)
    print(f"Train samples: {len(X_train)}, Event pos_weight: {pos_weight:.2f}")

    # 1. LightGBM Classifier (Spike event within 120 min)
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=cfg.seed,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train_event,
        eval_set=[(X_val, y_val_event)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    # 2. LightGBM Regressors for +30m (idx 1), +60m (idx 3), +120m (idx 7)
    reg_30 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=cfg.seed, n_jobs=-1, verbose=-1)
    reg_30.fit(X_train, y_train_future[:, 1], eval_set=[(X_val, y_val_future[:, 1])], callbacks=[lgb.early_stopping(20, verbose=False)])

    reg_60 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=cfg.seed, n_jobs=-1, verbose=-1)
    reg_60.fit(X_train, y_train_future[:, 3], eval_set=[(X_val, y_val_future[:, 3])], callbacks=[lgb.early_stopping(20, verbose=False)])

    reg_120 = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=cfg.seed, n_jobs=-1, verbose=-1)
    reg_120.fit(X_train, y_train_future[:, 7], eval_set=[(X_val, y_val_future[:, 7])], callbacks=[lgb.early_stopping(20, verbose=False)])

    # Evaluate on Test Set (leakage rule TRD 4.8: already_high excluded for classifier metric)
    test_mask = ~test_ah
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    test_auroc = roc_auc_score(y_test_event[test_mask], test_probs[test_mask])
    test_auprc = average_precision_score(y_test_event[test_mask], test_probs[test_mask])

    pred_30 = reg_30.predict(X_test)
    pred_60 = reg_60.predict(X_test)
    pred_120 = reg_120.predict(X_test)

    rmse_30 = float(np.sqrt(mean_squared_error(y_test_future[:, 1], pred_30)))
    rmse_60 = float(np.sqrt(mean_squared_error(y_test_future[:, 3], pred_60)))
    rmse_120 = float(np.sqrt(mean_squared_error(y_test_future[:, 7], pred_120)))

    metrics = {
        "auroc": float(test_auroc),
        "auprc": float(test_auprc),
        "rmse_30": rmse_30,
        "rmse_60": rmse_60,
        "rmse_120": rmse_120,
    }

    print("\n--- LightGBM Baseline Test Results ---")
    print(f"AUROC (event <= 120m): {test_auroc:.4f}")
    print(f"AUPRC (event <= 120m): {test_auprc:.4f}")
    print(f"RMSE @ +30m:  {rmse_30:.2f} mg/dL")
    print(f"RMSE @ +60m:  {rmse_60:.2f} mg/dL")
    print(f"RMSE @ +120m: {rmse_120:.2f} mg/dL")

    # Save models
    joblib.dump(clf, CURRENT_MODEL_DIR / "lgbm_classifier.joblib")
    joblib.dump({"reg_30": reg_30, "reg_60": reg_60, "reg_120": reg_120}, CURRENT_MODEL_DIR / "lgbm_regressors.joblib")
    
    return metrics

if __name__ == "__main__":
    train_and_eval_lightgbm()
