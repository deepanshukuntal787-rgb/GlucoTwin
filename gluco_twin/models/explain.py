"""Explainability and feature attribution module (TRD Section 5.7)."""
import torch
import numpy as np
from typing import Dict, List, Tuple
from gluco_twin.config import cfg
from gluco_twin.models.fusion_net import FusionTwinNet

CLINICAL_FACTOR_NAMES = [
    "Meal Carbohydrates",
    "Baseline Glycemia",
    "Circadian / Time of Day",
    "Physical Activity & Sleep",
    "EHR / Metabolic Profile",
]

class TwinExplainer:
    """Grouped feature attribution explainer for clinical interpretability."""
    def __init__(self, model: FusionTwinNet, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def explain_prediction(
        self,
        dynamic_x: torch.Tensor,
        future_cov: torch.Tensor,
        static_x: torch.Tensor,
    ) -> Dict[str, any]:
        """Explain event risk by group occlusion sensitivity.
        Calculates the delta in event logit when each group of features is ablated to baseline.
        """
        dyn_x = dynamic_x.clone().to(self.device)
        cov_x = future_cov.clone().to(self.device)
        stat_x = static_x.clone().to(self.device)

        with torch.no_grad():
            base_out = self.model(dyn_x, cov_x, stat_x)
            base_logit = base_out["event_logit"][0, 0].item()
            base_prob = 1.0 / (1.0 + np.exp(-base_logit))

        impacts = {}

        # 1. Meal Carbs ablation (dynamic channel 8, 9, cov channel 0)
        dyn_occ = dyn_x.clone()
        cov_occ = cov_x.clone()
        dyn_occ[:, :, 8] = 0.0  # carbs
        dyn_occ[:, :, 9] = 1.8  # long time since meal
        cov_occ[:, :, 0] = 0.0  # future carbs
        with torch.no_grad():
            out_occ = self.model(dyn_occ, cov_occ, stat_x)
            impacts["Meal Carbohydrates"] = abs(base_logit - out_occ["event_logit"][0, 0].item())

        # 2. Baseline Glycemia ablation (dynamic channel 0, 1 -> set to mean baseline)
        dyn_occ = dyn_x.clone()
        dyn_occ[:, :, 0] = 0.0  # glucose z=0 (140 mg/dL)
        dyn_occ[:, :, 1] = 0.0  # diff = 0
        with torch.no_grad():
            out_occ = self.model(dyn_occ, cov_x, stat_x)
            impacts["Baseline Glycemia"] = abs(base_logit - out_occ["event_logit"][0, 0].item())

        # 3. Circadian / Time of Day ablation (channels 10, 11)
        dyn_occ = dyn_x.clone()
        cov_occ = cov_x.clone()
        dyn_occ[:, :, 10:12] = 0.0
        cov_occ[:, :, 2:4] = 0.0
        with torch.no_grad():
            out_occ = self.model(dyn_occ, cov_occ, stat_x)
            impacts["Circadian / Time of Day"] = abs(base_logit - out_occ["event_logit"][0, 0].item())

        # 4. Activity & Sleep ablation (channels 2, 3, 4:8)
        dyn_occ = dyn_x.clone()
        cov_occ = cov_x.clone()
        dyn_occ[:, :, 2:8] = 0.0
        cov_occ[:, :, 1] = 0.0
        with torch.no_grad():
            out_occ = self.model(dyn_occ, cov_occ, stat_x)
            impacts["Physical Activity & Sleep"] = abs(base_logit - out_occ["event_logit"][0, 0].item())

        # 5. Static EHR Profile ablation
        stat_occ = torch.zeros_like(stat_x)
        with torch.no_grad():
            out_occ = self.model(dyn_x, cov_x, stat_occ)
            impacts["EHR / Metabolic Profile"] = abs(base_logit - out_occ["event_logit"][0, 0].item())

        # Normalize to percentages
        total_impact = sum(impacts.values()) + 1e-6
        percentages = {k: round((v / total_impact) * 100.0, 1) for k, v in impacts.items()}

        # Top drivers for human-readable narrative
        sorted_factors = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = sorted_factors[0], sorted_factors[1]
        
        narrative = (
            f"Risk is primarily driven by {top1[0]} ({top1[1]}%) "
            f"and {top2[0]} ({top2[1]}%)."
        )

        return {
            "base_prob": round(base_prob, 4),
            "factor_percentages": percentages,
            "sorted_factors": sorted_factors,
            "narrative": narrative,
        }
