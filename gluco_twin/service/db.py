"""SQLAlchemy database models and session manager for GlucoTwin."""
import json
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from gluco_twin.service.config import settings
from sqlalchemy import event

DATABASE_URL = settings.database_url

connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args["check_same_thread"] = False

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String(32), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(1), nullable=False)  # 'M' or 'F'
    diabetes_type = Column(String(16), default="T2D")
    duration_years = Column(Float, default=5.0)
    
    # Vitals
    bmi = Column(Float, default=27.5)
    weight_kg = Column(Float, default=75.0)
    height_cm = Column(Float, default=168.0)
    sbp_mmhg = Column(Integer, default=128)
    dbp_mmhg = Column(Integer, default=82)
    heart_rate_bpm = Column(Integer, default=72)

    # Labs
    hba1c_pct = Column(Float, default=7.4)
    fasting_glucose_mgdl = Column(Float, default=145.0)
    egfr_ml_min = Column(Float, default=85.0)
    ldl_mgdl = Column(Float, default=110.0)
    hdl_mgdl = Column(Float, default=42.0)
    triglycerides_mgdl = Column(Float, default=160.0)
    
    # Metadata
    medications_raw = Column(Text, default="")
    comorbidities_raw = Column(Text, default="")
    clinician = Column(String(128), default="Dr. A. Sharma (Endocrinology)")
    diagnose_date = Column(String(32), default="2018-05-12")
    target_low = Column(Float, default=70.0)
    target_high = Column(Float, default=180.0)
    twin_status = Column(String(32), default="personalized")

    timeseries = relationship("TimeseriesReading", back_populates="patient", cascade="all, delete-orphan")


class TimeseriesReading(Base):
    __tablename__ = "timeseries_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(32), ForeignKey("patients.id"), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    glucose = Column(Float, nullable=False)
    heart_rate = Column(Float, default=72.0)
    steps = Column(Integer, default=0)
    sleep_stage = Column(Integer, default=0)
    carbs_g = Column(Float, default=0.0)

    patient = relationship("Patient", back_populates="timeseries")

    __table_args__ = (
        Index("idx_patient_timestamp", "patient_id", "timestamp"),
    )


def init_db():
    Base.metadata.create_all(bind=engine)
