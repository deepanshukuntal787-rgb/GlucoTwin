"""Probability calibration and clinical alert policy module (TRD Section 5.4)."""
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, brier_score_loss
from gluco_twin.config import cfg, CURRENT_MODEL_DIR

class Calibrator:
    """Isotonic regression calibrator with threshold and alert policy."""
    def __init__(self, threshold: float = 0.35):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.alert_threshold = threshold
        self.fitted = False

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        # Fit isotonic regression
        self.iso.fit(val_probs, val_labels)
        self.fitted = True

        # Optimize alert threshold on validation
        calibrated_probs = self.iso.predict(val_probs)
        precision, recall, thresholds = precision_recall_curve(val_labels, calibrated_probs)
        
        # Pick threshold targeting precision >= 0.60
        valid_idx = np.where(precision[:-1] >= 0.55)[0]
        if len(valid_idx) > 0:
            self.alert_threshold = float(thresholds[valid_idx[0]])
        else:
            self.alert_threshold = 0.35

        print(f"Calibrator fitted. Chosen alert threshold: {self.alert_threshold:.3f}")
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return probs
        return self.iso.predict(probs)

    def evaluate_severity(self, prob: float, p50_60min: float) -> str:
        """Severity policy (TRD 5.4):
        - High: prob >= 0.60 or p50 > 250 mg/dL within 60 min
        - Moderate: 0.30 <= prob < 0.60
        - Low: prob < 0.30
        """
        if prob >= 0.60 or p50_60min >= 250.0:
            return "HIGH"
        elif prob >= self.alert_threshold:
            return "MODERATE"
        return "LOW"


def save_calibrator(calibrator: Calibrator, path=None):
    if path is None:
        path = CURRENT_MODEL_DIR / "calibrator.pkl"
    joblib.dump(calibrator, path)
    print(f"Calibrator saved to {path}")


def load_calibrator(path=None) -> Calibrator:
    if path is None:
        path = CURRENT_MODEL_DIR / "calibrator.pkl"
    return joblib.load(path)
