"""GlucoTwin configuration module."""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
CURRENT_MODEL_DIR = MODELS_DIR / "current"
REPORTS_DIR = BASE_DIR / "reports"

@dataclass
class Config:
    # Sampling & Horizons
    grid_minutes: int = 15
    history_steps: int = 96      # 24 hours (96 * 15 min)
    horizon_steps: int = 8       # 2 hours (8 * 15 min)
    event_threshold: float = 180.0  # mg/dL
    event_min_run: int = 3       # 3 consecutive steps (45 min) > threshold

    # Quantiles
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    glucose_mean: float = 140.0
    glucose_std: float = 50.0

    # Model architecture
    static_dim: int = 21
    dynamic_dim: int = 16
    covariate_dim: int = 5
    hidden_dim: int = 128
    decoder_dim: int = 64
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 30
    patience: int = 6
    seed: int = 42

    # Split
    train_patients: int = 70
    val_patients: int = 15
    test_patients: int = 15

cfg = Config()
