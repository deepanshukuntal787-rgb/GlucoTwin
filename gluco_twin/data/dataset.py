"""PyTorch Dataset and caching pipeline for GlucoTwin."""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
from gluco_twin.config import cfg, RAW_DATA_DIR, PROCESSED_DATA_DIR, CURRENT_MODEL_DIR
from gluco_twin.data.preprocess import (
    parse_static_features,
    build_patient_splits,
    process_dynamic_patient_series,
    extract_sliding_windows,
)

class TwinDataset(Dataset):
    """PyTorch Dataset wrapping sliding window tensors for GlucoTwin."""
    def __init__(self, npz_path: str, is_train: bool = False, augment: bool = False):
        data = np.load(npz_path, allow_pickle=True)
        self.history = torch.from_numpy(data["history"]).float()       # (N, 96, 16)
        self.covariates = torch.from_numpy(data["covariates"]).float() # (N, 8, 5)
        self.static = torch.from_numpy(data["static"]).float()         # (N, 21)
        self.targets = torch.from_numpy(data["targets"]).float()       # (N, 8)
        self.events = torch.from_numpy(data["events"]).float()         # (N,)
        self.already_high = torch.from_numpy(data["already_high"])     # (N,)
        self.patient_id = data["patient_id"]                           # (N,)
        self.raw_future = torch.from_numpy(data["raw_future"]).float() # (N, 8)
        self.is_train = is_train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hist = self.history[idx].clone()
        cov = self.covariates[idx].clone()
        stat = self.static[idx].clone()
        targets = self.targets[idx].clone()
        event = self.events[idx]
        already_high = self.already_high[idx]

        # Online Augmentations (TRD 5.3) for training
        if self.is_train and self.augment:
            # 1. Gaussian noise on normalized glucose (channel 0)
            noise = torch.randn_like(hist[:, 0]) * (2.0 / cfg.glucose_std)
            hist[:, 0] = hist[:, 0] + noise

            # 2. Random channel dropout for wearable sensors (HR, steps, sleep)
            if torch.rand(1).item() < 0.15:
                hist[:, 2] = 0.0  # hr
                hist[:, 13] = 0.0 # mask_hr
            if torch.rand(1).item() < 0.15:
                hist[:, 3] = 0.0  # steps
                hist[:, 14] = 0.0 # mask_steps
            if torch.rand(1).item() < 0.15:
                hist[:, 4:8] = 0.0 # sleep
                hist[:, 15] = 0.0  # mask_sleep

            # 3. Future covariate masking (probability 0.5)
            if torch.rand(1).item() < 0.5:
                cov[:, 0:2] = 0.0 # carbs and steps
                cov[:, 4] = 0.0   # cov_mask

        return {
            "history": hist,
            "covariates": cov,
            "static": stat,
            "targets": targets,
            "event": event,
            "already_high": already_high,
            "raw_future": self.raw_future[idx],
            "patient_id": self.patient_id[idx],
        }


def prepare_and_cache_datasets(stride_train: int = 3, stride_eval: int = 4, force: bool = False):
    """Process raw CSV files and cache train.npz, val.npz, and test.npz."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    train_npz = PROCESSED_DATA_DIR / "train.npz"
    val_npz = PROCESSED_DATA_DIR / "val.npz"
    test_npz = PROCESSED_DATA_DIR / "test.npz"

    if train_npz.exists() and val_npz.exists() and test_npz.exists() and not force:
        print(f"Cached datasets found in {PROCESSED_DATA_DIR}, skipping extraction.")
        return str(train_npz), str(val_npz), str(test_npz)

    print("Parsing static EHR features from dataset2.csv...")
    static_df, static_map, static_cols = parse_static_features()
    
    # Save feature metadata
    with open(CURRENT_MODEL_DIR / "scaler.json", "w") as f:
        json.dump({
            "static_cols": static_cols,
            "glucose_mean": cfg.glucose_mean,
            "glucose_std": cfg.glucose_std,
            "grid_minutes": cfg.grid_minutes,
            "history_steps": cfg.history_steps,
            "horizon_steps": cfg.horizon_steps,
        }, f, indent=2)

    patients = sorted(list(static_map.keys()))
    train_pts, val_pts, test_pts = build_patient_splits(patients, seed=cfg.seed)
    
    print(f"Splits created: Train={len(train_pts)}, Val={len(val_pts)}, Test={len(test_pts)}")
    
    print("Reading dynamic time-series from xxxdepanshu.csv...")
    dyn_df = pd.read_csv(RAW_DATA_DIR / "xxxdepanshu.csv")
    
    splits_dict = {
        "train": (train_pts, stride_train, train_npz),
        "val": (val_pts, stride_eval, val_npz),
        "test": (test_pts, stride_eval, test_npz),
    }

    for split_name, (split_pts, stride, out_path) in splits_dict.items():
        print(f"Extracting {split_name} samples (patients={len(split_pts)}, stride={stride})...")
        hists, covs, stats, targets, events, already_highs, p_ids, raw_futs = [], [], [], [], [], [], [], []
        
        for pid in split_pts:
            pdf = dyn_df[dyn_df["patient_id"] == pid]
            if len(pdf) == 0:
                continue
            dyn_chan, fut_cov, raw_g = process_dynamic_patient_series(pdf)
            stat_v = static_map[pid]
            w = extract_sliding_windows(dyn_chan, fut_cov, raw_g, stat_v, pid, stride=stride)
            
            hists.append(w["history"])
            covs.append(w["covariates"])
            stats.append(w["static"])
            targets.append(w["targets"])
            events.append(w["events"])
            already_highs.append(w["already_high"])
            p_ids.append(w["patient_id"])
            raw_futs.append(w["raw_future"])

        cat_hist = np.concatenate(hists, axis=0)
        cat_cov = np.concatenate(covs, axis=0)
        cat_stat = np.concatenate(stats, axis=0)
        cat_targets = np.concatenate(targets, axis=0)
        cat_events = np.concatenate(events, axis=0)
        cat_already_high = np.concatenate(already_highs, axis=0)
        cat_pids = np.concatenate(p_ids, axis=0)
        cat_raw_futs = np.concatenate(raw_futs, axis=0)

        print(f"  {split_name}: {len(cat_events)} samples, {cat_events.sum()} events ({cat_events.mean()*100:.2f}% prevalence)")

        np.savez_compressed(
            out_path,
            history=cat_hist,
            covariates=cat_cov,
            static=cat_stat,
            targets=cat_targets,
            events=cat_events,
            already_high=cat_already_high,
            patient_id=cat_pids,
            raw_future=cat_raw_futs,
        )

    return str(train_npz), str(val_npz), str(test_npz)
