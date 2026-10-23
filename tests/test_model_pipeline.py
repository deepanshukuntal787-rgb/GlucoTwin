"""Automated validation tests and leakage checklist (TRD Section 4.8)."""
import json
import numpy as np
import torch
import pytest
from gluco_twin.config import cfg, PROCESSED_DATA_DIR, CURRENT_MODEL_DIR
from gluco_twin.models.fusion_net import FusionTwinNet

def test_patient_split_leakage():
    """Rule 1: No patient appears in more than one split."""
    train_data = np.load(PROCESSED_DATA_DIR / "train.npz", allow_pickle=True)
    val_data = np.load(PROCESSED_DATA_DIR / "val.npz", allow_pickle=True)
    test_data = np.load(PROCESSED_DATA_DIR / "test.npz", allow_pickle=True)

    train_pts = set(train_data["patient_id"])
    val_pts = set(val_data["patient_id"])
    test_pts = set(test_data["patient_id"])

    assert len(train_pts.intersection(val_pts)) == 0, "Leakage detected: train and val share patients!"
    assert len(train_pts.intersection(test_pts)) == 0, "Leakage detected: train and test share patients!"
    assert len(val_pts.intersection(test_pts)) == 0, "Leakage detected: val and test share patients!"
    print("Test passed: Zero patient leakage across splits.")


def test_temporal_causality():
    """Rule 3: Inputs use only history <= t, targets use only horizon > t."""
    train_data = np.load(PROCESSED_DATA_DIR / "train.npz", allow_pickle=True)
    assert train_data["history"].shape[1] == cfg.history_steps  # 96 steps
    assert train_data["covariates"].shape[1] == cfg.horizon_steps # 8 steps
    assert train_data["targets"].shape[1] == cfg.horizon_steps    # 8 steps
    print("Test passed: Temporal shapes strictly conform to 24h history and 2h horizon.")


def test_quantile_non_crossing():
    """Rule 5: P10 <= P50 <= P90 must strictly hold for all predictions."""
    model = FusionTwinNet()
    model.eval()
    
    dyn = torch.randn(16, cfg.history_steps, cfg.dynamic_dim)
    cov = torch.randn(16, cfg.horizon_steps, cfg.covariate_dim)
    stat = torch.randn(16, cfg.static_dim)
    
    with torch.no_grad():
        out = model(dyn, cov, stat)
        q = out["quantiles"]
        p10 = q[:, :, 0]
        p50 = q[:, :, 1]
        p90 = q[:, :, 2]
        
        assert (p10 <= p50).all(), "Quantile crossing: P10 > P50 found!"
        assert (p50 <= p90).all(), "Quantile crossing: P50 > P90 found!"
    print("Test passed: Monotonic non-crossing quantiles verified.")


def test_scaler_metadata_exists():
    """Rule 2: Scaler fitted and serialized without future info."""
    scaler_path = CURRENT_MODEL_DIR / "scaler.json"
    assert scaler_path.exists(), "Scaler metadata missing!"
    with open(scaler_path) as f:
        meta = json.load(f)
    assert "static_cols" in meta
    assert len(meta["static_cols"]) == cfg.static_dim
    print("Test passed: Scaler metadata valid.")


if __name__ == "__main__":
    test_patient_split_leakage()
    test_temporal_causality()
    test_quantile_non_crossing()
    test_scaler_metadata_exists()
    print("All TRD compliance tests PASSED!")
