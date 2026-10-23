"""What-If Counterfactual Simulation Engine and Monotonicity Sanity Checks (TRD Section 5.6)."""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from gluco_twin.config import cfg
from gluco_twin.models.fusion_net import FusionTwinNet

class WhatIfSimulator:
    """Simulates counterfactual scenarios (e.g. planned meals, exercise) on patient digital twin."""
    def __init__(self, model: FusionTwinNet, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def simulate(
        self,
        dynamic_x: torch.Tensor,
        base_cov: torch.Tensor,
        static_x: torch.Tensor,
        add_carbs_g: float = 0.0,
        carb_offset_step: int = 1,
        add_steps: float = 0.0,
        step_duration_steps: int = 2,
        adapter_b: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Run counterfactual simulation with hypothetical interventions.
        Args:
            dynamic_x: (1, 96, 16)
            base_cov: (1, 8, 5)
            static_x: (1, 21)
            add_carbs_g: hypothetical meal carbs added
            carb_offset_step: step in horizon [0..7] where meal occurs
            add_steps: hypothetical steps added
        """
        cov_mod = base_cov.clone()
        
        # Add carbs to horizon covariate (cov[:, :, 0] is plan_carbs_z)
        if add_carbs_g > 0 and 0 <= carb_offset_step < cfg.horizon_steps:
            cov_mod[0, carb_offset_step, 0] += (add_carbs_g / 20.0)
            cov_mod[0, carb_offset_step, 4] = 1.0  # cov_mask

        # Add steps to horizon covariate (cov[:, :, 1] is plan_steps_z)
        if add_steps > 0:
            for s in range(carb_offset_step, min(cfg.horizon_steps, carb_offset_step + step_duration_steps)):
                added_log = np.log1p(add_steps / step_duration_steps) / 2.0
                cov_mod[0, s, 1] += added_log
                cov_mod[0, s, 4] = 1.0

        with torch.no_grad():
            out = self.model(
                dynamic_x.to(self.device),
                cov_mod.to(self.device),
                static_x.to(self.device),
                adapter_b=adapter_b.to(self.device) if adapter_b is not None else None,
            )
            
            # Unscale quantiles to mg/dL
            quantiles_scaled = out["quantiles"][0].cpu().numpy()  # (8, 3)
            quantiles_mgdl = quantiles_scaled * cfg.glucose_std + cfg.glucose_mean
            
            event_logit = out["event_logit"][0, 0].item()
            event_prob = 1.0 / (1.0 + np.exp(-event_logit))

        return {
            "p10": quantiles_mgdl[:, 0],
            "p50": quantiles_mgdl[:, 1],
            "p90": quantiles_mgdl[:, 2],
            "event_prob": event_prob,
            "peak_glucose": float(np.max(quantiles_mgdl[:, 1])),
            "delta_carbs": add_carbs_g,
            "delta_steps": add_steps,
        }

    def run_monotonicity_sanity_checks(
        self,
        test_dataset,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """TRD 5.6: Test that adding carbs increases predicted peak and adding steps lowers it."""
        indices = np.random.RandomState(cfg.seed).choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
        
        carb_monotonic_count = 0
        step_monotonic_count = 0

        for idx in indices:
            item = test_dataset[idx]
            dyn_x = item["history"].unsqueeze(0)
            cov_x = item["covariates"].unsqueeze(0)
            stat_x = item["static"].unsqueeze(0)

            # Baseline
            base_res = self.simulate(dyn_x, cov_x, stat_x)
            base_peak = base_res["peak_glucose"]

            # Scenario 1: Add 40g carbs
            carb_res = self.simulate(dyn_x, cov_x, stat_x, add_carbs_g=40.0)
            if carb_res["peak_glucose"] >= base_peak - 1.0:  # Allow 1 mg/dL numerical tolerance
                carb_monotonic_count += 1

            # Scenario 2: Add 3000 steps of brisk walking
            step_res = self.simulate(dyn_x, cov_x, stat_x, add_steps=3000.0)
            if step_res["peak_glucose"] <= base_peak + 1.0:
                step_monotonic_count += 1

        carb_pct = (carb_monotonic_count / len(indices)) * 100.0
        step_pct = (step_monotonic_count / len(indices)) * 100.0

        return {
            "carb_monotonic_pct": carb_pct,
            "step_monotonic_pct": step_pct,
            "target_achieved": (carb_pct >= 90.0 and step_pct >= 90.0),
        }
