"""Database seeder: populates SQLite database from dataset2.csv and xxxdepanshu.csv."""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from gluco_twin.config import RAW_DATA_DIR, DATA_DIR
from gluco_twin.service.db import DB_PATH, init_db

PATIENT_NAMES_M = [
    "Rajesh Verma", "Amit Kumar", "Suresh Patel", "Vikram Singh", "Manoj Sharma",
    "Anil Gupta", "Ramesh Joshi", "Deepak Rao", "Sanjay Mehta", "Pradeep Nair",
    "Sunil Pillai", "Arun Bhatt", "Vijay Deshmukh", "Prakash Kulkarni", "Kishore Reddy",
    "Gopal Iyer", "Mahesh Hegde", "Ashok Choudhury", "Naveen Saxena", "Dinesh Aggarwal",
]

PATIENT_NAMES_F = [
    "Sunita Sharma", "Priya Nair", "Anjali Deshmukh", "Kavita Patel", "Rekha Verma",
    "Meena Gupta", "Shobha Rao", "Geeta Joshi", "Pooja Mehta", "Anita Pillai",
    "Deepa Bhatt", "Usha Kulkarni", "Lakshmi Reddy", "Padma Iyer", "Neeta Hegde",
    "Sarita Choudhury", "Vandana Saxena", "Radha Aggarwal", "Maya Singh", "Indira Kumar",
]

def generate_patient_name(patient_id: str, sex: str, seed: int) -> str:
    rng = np.random.RandomState(seed)
    if sex.upper() == "F":
        name = rng.choice(PATIENT_NAMES_F)
    else:
        name = rng.choice(PATIENT_NAMES_M)
    # Add unique initial based on ID to avoid duplicate display names
    num = int(patient_id.replace("P", ""))
    return f"{name} ({patient_id})"


def seed_database():
    """Populate database with 100 patients and continuous timeseries readings."""
    init_db()
    print("=== Seeding GlucoTwin Database ===")

    # 1. Load static patients
    static_df = pd.read_csv(RAW_DATA_DIR / "dataset2.csv")
    static_df["medications"] = static_df["medications"].fillna("None")
    static_df["comorbidities"] = static_df["comorbidities"].fillna("None")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM timeseries_readings")
    cursor.execute("DELETE FROM patients")
    conn.commit()

    print(f"Inserting {len(static_df)} patients...")
    patient_rows = []
    for idx, row in static_df.iterrows():
        pid = row["patient_id"]
        sex = str(row["sex"]).upper()
        age = int(row["age"])
        bmi = float(row["bmi"])
        hba1c = float(row["hba1c_pct"])
        duration = float(row["diabetes_duration_years"])
        dtype = "T1D" if "1" in str(row["diabetes_type"]) else "T2D"
        name = generate_patient_name(pid, sex, idx + 100)

        # Height & Weight
        height_cm = 162.0 if sex == "F" else 172.0
        weight_kg = round(bmi * ((height_cm / 100.0) ** 2), 1)

        # Blood pressure & labs
        is_htn = "htn" in str(row["comorbidities"]).lower()
        sbp = 138 if is_htn else int(115 + (age * 0.2))
        dbp = 88 if is_htn else int(75 + (age * 0.1))
        hr = 74 if is_htn else 70

        fasting_g = round(28.7 * hba1c - 46.7, 1)
        egfr = round(max(45.0, 110.0 - (age * 0.5) - (duration * 1.2)), 1)
        ldl = round(100.0 + (bmi * 1.2), 1)
        hdl = round(52.0 - (bmi * 0.3), 1)
        trig = round(130.0 + (bmi * 2.0), 1)

        diagnose_year = max(2002, 2023 - int(duration))
        diagnose_date = f"{diagnose_year}-04-15"

        patient_rows.append((
            pid, name, age, sex, dtype, duration,
            bmi, weight_kg, height_cm, sbp, dbp, hr,
            hba1c, fasting_g, egfr, ldl, hdl, trig,
            str(row["medications"]), str(row["comorbidities"]),
            "Dr. V. Raman (Consultant Diabetologist)", diagnose_date,
            70.0, 180.0, "personalized",
        ))

    cursor.executemany("""
        INSERT INTO patients (
            id, name, age, sex, diabetes_type, duration_years,
            bmi, weight_kg, height_cm, sbp_mmhg, dbp_mmhg, heart_rate_bpm,
            hba1c_pct, fasting_glucose_mgdl, egfr_ml_min, ldl_mgdl, hdl_mgdl, triglycerides_mgdl,
            medications_raw, comorbidities_raw, clinician, diagnose_date,
            target_low, target_high, twin_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, patient_rows)
    conn.commit()
    print("Patients successfully inserted.")

    # 2. Load dynamic timeseries
    print("Loading and indexing timeseries from xxxdepanshu.csv...")
    dyn_df = pd.read_csv(RAW_DATA_DIR / "xxxdepanshu.csv")

    # Fast bulk insert via sqlite3 executemany
    records = []
    for row in dyn_df.itertuples(index=False):
        records.append((
            row.patient_id,
            row.timestamp,
            float(row.glucose),
            float(row.heart_rate),
            int(row.steps),
            int(row.sleep_stage),
            float(row.carbs_g),
        ))

    print(f"Inserting {len(records):,} time-series observations...")
    cursor.executemany("""
        INSERT INTO timeseries_readings (
            patient_id, timestamp, glucose, heart_rate, steps, sleep_stage, carbs_g
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()

    # Index optimization
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts_patient_time ON timeseries_readings (patient_id, timestamp)")
    conn.commit()
    conn.close()

    print("Database seeding completed successfully!")

if __name__ == "__main__":
    seed_database()
