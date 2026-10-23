"""Service configuration and environment variable management for GlucoTwin."""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
from gluco_twin.config import DATA_DIR, CURRENT_MODEL_DIR

class Settings(BaseSettings):
    env: str = Field(default="production", alias="ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    
    # Database
    database_url: str = Field(
        default=f"sqlite:///{DATA_DIR / 'glucotwin.db'}",
        alias="DATABASE_URL",
    )
    
    # Model storage
    model_dir: Path = Field(default=CURRENT_MODEL_DIR, alias="MODEL_DIR")
    force_cpu: bool = Field(default=False, alias="FORCE_CPU")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
