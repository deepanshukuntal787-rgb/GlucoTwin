"""Personalization module for patient-specific digital twin adaptation (TRD Section 5.5)."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from gluco_twin.config import cfg
from gluco_twin.models.fusion_net import FusionTwinNet

class PatientAdapter(nn.Module):
    """Learnable per-patient parameter vector: bias b_p (128-d) + output scale and shift."""
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.bias_p = nn.Parameter(torch.zeros(hidden_dim))
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))


def train_patient_adapter(
    base_model: FusionTwinNet,
    patient_windows: Dict[str, torch.Tensor],
    device: torch.device,
    steps: int = 100,
    lr: float = 1e-2,
    l2_reg: float = 1e-3,
) -> Tuple[PatientAdapter, Dict[str, float]]:
    """Fine-tune a per-patient adapter on a patient's historical calibration data."""
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    adapter = PatientAdapter(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    dyn_x = patient_windows["history"].to(device)
    cov_x = patient_windows["covariates"].to(device)
    stat_x = patient_windows["static"].to(device)
    targets = patient_windows["targets"].to(device)

    # Base RMSE before personalization
    with torch.no_grad():
        base_out = base_model(dyn_x, cov_x, stat_x)
        base_rmse = np.sqrt(np.mean(((base_out["quantiles"][:, :, 1].cpu().numpy() - targets.cpu().numpy()) * cfg.glucose_std) ** 2))

    # Optimization loop
    for step in range(steps):
        optimizer.zero_grad()
        out = base_model(
            dyn_x,
            cov_x,
            stat_x,
            adapter_b=adapter.bias_p.unsqueeze(0),
            output_scale=adapter.scale,
            output_shift=adapter.shift,
        )
        p50 = out["quantiles"][:, :, 1]
        mse_loss = nn.functional.mse_loss(p50, targets)
        reg_loss = l2_reg * torch.sum(adapter.bias_p ** 2)
        total_loss = mse_loss + reg_loss

        total_loss.backward()
        optimizer.step()

    # Adapted RMSE
    with torch.no_grad():
        adapted_out = base_model(
            dyn_x,
            cov_x,
            stat_x,
            adapter_b=adapter.bias_p.unsqueeze(0),
            output_scale=adapter.scale,
            output_shift=adapter.shift,
        )
        adapted_rmse = np.sqrt(np.mean(((adapted_out["quantiles"][:, :, 1].cpu().numpy() - targets.cpu().numpy()) * cfg.glucose_std) ** 2))

    metrics = {
        "base_rmse": float(base_rmse),
        "adapted_rmse": float(adapted_rmse),
        "delta_rmse": float(base_rmse - adapted_rmse),
    }

    return adapter, metrics
