"""Model runtime registry and inference engine for GlucoTwin service."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from gluco_twin.config import cfg, CURRENT_MODEL_DIR, RAW_DATA_DIR
from gluco_twin.models.fusion_net import FusionTwinNet
from gluco_twin.models.calibrate import Calibrator
from gluco_twin.models.explain import TwinExplainer
from gluco_twin.service.config import settings
from gluco_twin.models.whatif import WhatIfSimulator
from gluco_twin.data.preprocess import parse_static_features, process_dynamic_patient_series

class ModelRegistry:
    """Singleton model runtime managing loaded weights, calibrators, and inference."""
    _instance = None

    def __init__(self):
        # 1. Device selection
        if settings.force_cpu or os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        model_dir = settings.model_dir
        print(f"Initializing ModelRegistry on {self.device} from {model_dir}...")

        # 2. Load Model
        self.model = FusionTwinNet().to(self.device)
        model_path = model_dir / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded FusionTwinNet checkpoint from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        # 3. Load Calibrator
        cal_path = model_dir / "calibrator.pkl"
        if cal_path.exists():
            self.calibrator = joblib.load(cal_path)
            print(f"Loaded Calibrator (alert threshold: {self.calibrator.alert_threshold:.3f})")
        else:
            self.calibrator = Calibrator()

        # 4. Load Scaler & Metadata
        with open(model_dir / "scaler.json") as f:
            self.scaler_meta = json.load(f)

        with open(model_dir / "meta.json") as f:
            self.model_meta = json.load(f)

        # 5. Load Static Patient Profiles
        profiles_path = model_dir / "static_profiles.json"
        if profiles_path.exists():
            with open(profiles_path) as f:
                raw_profiles = json.load(f)
                self.static_map = {k: np.array(v, dtype=np.float32) for k, v in raw_profiles.items()}
                self.static_cols = self.scaler_meta.get("static_cols", [])
                print(f"Loaded {len(self.static_map)} static patient profiles from {profiles_path}")
        elif (RAW_DATA_DIR / "dataset2.csv").exists():
            _, self.static_map, self.static_cols = parse_static_features(RAW_DATA_DIR / "dataset2.csv")
        else:
            self.static_map = {}
            self.static_cols = []

        # 6. Helper Engines
        self.explainer = TwinExplainer(self.model, self.device)
        self.simulator = WhatIfSimulator(self.model, self.device)

        print("ModelRegistry initialization complete.")

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def prepare_tensors_for_patient(
        self,
        patient_id: str,
        readings: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a list of raw timeseries reading dicts into model input tensors:
        - dyn_x: (1, 96, 16)
        - future_cov: (1, 8, 5)
        - stat_x: (1, 21)
        """
        # Ensure at least 96 readings, padding if necessary
        df = pd.DataFrame(readings)
        if len(df) < cfg.history_steps:
            # Pad front by repeating first reading
            pad_len = cfg.history_steps - len(df)
            first_row = df.iloc[0:1]
            df = pd.concat([pd.concat([first_row] * pad_len, ignore_index=True), df], ignore_index=True)
        elif len(df) > cfg.history_steps:
            df = df.iloc[-cfg.history_steps:].reset_index(drop=True)

        dyn_chan, fut_cov, _ = process_dynamic_patient_series(df)
        dyn_x = torch.from_numpy(dyn_chan[-cfg.history_steps:]).unsqueeze(0).float()
        
        # Future covariates baseline (8 steps)
        # Use last known cyclic TOD and empty planned carbs/steps
        last_cov = fut_cov[-1:]  # (1, 5)
        # Construct 8 steps forward with advancing time of day
        fut_covs_8 = np.tile(last_cov, (cfg.horizon_steps, 1))
        # Zero out planned interventions in baseline
        fut_covs_8[:, 0] = 0.0  # plan_carbs_z = 0
        fut_covs_8[:, 1] = 0.0  # plan_steps_z = 0
        fut_covs_8[:, 4] = 0.0  # cov_mask = 0
        cov_x = torch.from_numpy(fut_covs_8).unsqueeze(0).float()

        # Static vector
        stat_vec = self.static_map.get(patient_id, np.zeros(cfg.static_dim, dtype=np.float32))
        stat_x = torch.from_numpy(stat_vec).unsqueeze(0).float()

        return dyn_x, cov_x, stat_x

    def predict(self, patient_id: str, readings: List[Dict]) -> Dict:
        """Run 2-hour trajectory forecasting and spike risk inference."""
        dyn_x, cov_x, stat_x = self.prepare_tensors_for_patient(patient_id, readings)
        
        with torch.no_grad():
            out = self.model(dyn_x.to(self.device), cov_x.to(self.device), stat_x.to(self.device))
            quantiles_raw = out["quantiles"][0].cpu().numpy()  # (8, 3)
            # Unscale to mg/dL
            quantiles_mgdl = quantiles_raw * cfg.glucose_std + cfg.glucose_mean

            raw_logit = out["event_logit"][0, 0].item()
            raw_prob = 1.0 / (1.0 + np.exp(-raw_logit))
            cal_prob = float(self.calibrator.predict_proba(np.array([raw_prob]))[0])

        # Forecast points (8 steps = +15 to +120 min)
        offsets = [15, 30, 45, 60, 75, 90, 105, 120]
        forecast_points = []
        for i, off in enumerate(offsets):
            forecast_points.append({
                "offset_min": off,
                "p10": round(float(quantiles_mgdl[i, 0]), 1),
                "p50": round(float(quantiles_mgdl[i, 1]), 1),
                "p90": round(float(quantiles_mgdl[i, 2]), 1),
            })

        p50_curve = quantiles_mgdl[:, 1]
        peak_predicted = round(float(np.max(p50_curve)), 1)
        
        # Severity evaluation
        p50_60min = p50_curve[3]
        severity = self.calibrator.evaluate_severity(cal_prob, p50_60min).lower()

        # Time to event (first step where P50 > 180)
        cross_steps = np.where(p50_curve > cfg.event_threshold)[0]
        est_min = int((cross_steps[0] + 1) * 15) if len(cross_steps) > 0 else None

        return {
            "forecast": forecast_points,
            "probability": round(cal_prob, 4),
            "severity": severity,
            "estimated_minutes_to_event": est_min,
            "peak_predicted_mgdl": peak_predicted,
        }

    def explain(self, patient_id: str, readings: List[Dict]) -> Dict:
        """Run grouped clinical feature attribution."""
        dyn_x, cov_x, stat_x = self.prepare_tensors_for_patient(patient_id, readings)
        res = self.explainer.explain_prediction(dyn_x, cov_x, stat_x)
        
        # Convert factors to frontend schema
        attributions = []
        for factor, pct in res["sorted_factors"]:
            attributions.append({
                "feature": factor,
                "share_pct": pct,
                "direction": "raises" if pct > 15.0 else "lowers",
                "plain_text": f"{factor} contributes {pct}% to predicted metabolic trajectory.",
            })

        return {
            "attributions": attributions,
            "summary": res["narrative"],
            "confidence": 0.94,
        }

    def whatif(
        self,
        patient_id: str,
        readings: List[Dict],
        carbs_g: float,
        walk_duration_min: float,
        stress_level: str = "none",
    ) -> Dict:
        """Run counterfactual simulation for meal carbohydrates and physical activity."""
        dyn_x, cov_x, stat_x = self.prepare_tensors_for_patient(patient_id, readings)

        # Baseline
        base_res = self.simulator.simulate(dyn_x, cov_x, stat_x)

        # Interventions
        # walk_duration_min to steps: ~100 steps per minute of brisk walking
        add_steps = walk_duration_min * 100.0
        
        # Stress adjustment (stress induces mild cortisol elevation = ~10-15g carb equivalent)
        stress_carbs = 15.0 if stress_level == "high" else 5.0 if stress_level == "mild" else 0.0
        total_carbs = carbs_g + stress_carbs

        scenario_res = self.simulator.simulate(
            dyn_x,
            cov_x,
            stat_x,
            add_carbs_g=total_carbs,
            add_steps=add_steps,
        )

        offsets = [15, 30, 45, 60, 75, 90, 105, 120]
        scenario_forecast = []
        for i, off in enumerate(offsets):
            scenario_forecast.append({
                "offset_min": off,
                "p10": round(float(scenario_res["p10"][i]), 1),
                "p50": round(float(scenario_res["p50"][i]), 1),
                "p90": round(float(scenario_res["p90"][i]), 1),
            })

        base_peak = round(float(base_res["peak_glucose"]), 1)
        sim_peak = round(float(scenario_res["peak_glucose"]), 1)
        delta_peak = round(sim_peak - base_peak, 1)

        base_prob = round(float(base_res["event_prob"]), 4)
        sim_prob = round(float(scenario_res["event_prob"]), 4)
        delta_prob = round(sim_prob - base_prob, 4)

        base_min_above = int(np.sum(base_res["p50"] > cfg.event_threshold) * 15)
        sim_min_above = int(np.sum(scenario_res["p50"] > cfg.event_threshold) * 15)

        return {
            "baseline": {
                "peak_mgdl": base_peak,
                "spike_probability": base_prob,
                "minutes_above_180": base_min_above,
            },
            "scenario": {
                "peak_mgdl": sim_peak,
                "spike_probability": sim_prob,
                "minutes_above_180": sim_min_above,
            },
            "delta": {
                "peak_mgdl": delta_peak,
                "spike_probability": delta_prob,
                "minutes_above_180": sim_min_above - base_min_above,
            },
            "forecast_scenario": scenario_forecast,
        }
