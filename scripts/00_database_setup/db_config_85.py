"""
Shared configuration for the 85-sample fatigue DBMS workflow.
Edit only this file if your PostgreSQL password, database name, or project path changes.
"""
from pathlib import Path
from sqlalchemy import create_engine

# ---------------- PostgreSQL configuration ----------------
DB_USER = "postgres"
DB_PASS = "505505"          # Change only if your PostgreSQL password is different
DB_HOST = "localhost"
DB_PORT = "5432"

# Recommended: keep the 85-sample work in a separate DB to avoid mixing with the old 17-route DB.
# Create this database once in DBeaver/pgAdmin, or change this back to fatigue_dbms_v1 if needed.
DB_NAME = "fatigue_dbms_85"

# ---------------- Project folder configuration ----------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "01_input"
CLEANED_DIR = DATA_DIR / "02_cleaned"
DB_OUTPUT_DIR = DATA_DIR / "03_db_outputs"

MASTER_PATH = INPUT_DIR / "master_specimen_85_samples_consolidated.csv"
CYCLE_PATH = CLEANED_DIR / "cycle_summary_85_samples_consolidated.csv"

# Optional. If this file is absent, load_loop_manifest_85.py will generate a summary-level manifest.
LOOP_MANIFEST_PATH = CLEANED_DIR / "loop_manifest_85.csv"

EXPECTED_ROUTES = 17
EXPECTED_SAMPLES = 85
EXPECTED_SAMPLES_PER_ROUTE = 5

# Use True for the first clean load of the 85-sample dataset.
FULL_REFRESH = True


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        future=True,
    )

