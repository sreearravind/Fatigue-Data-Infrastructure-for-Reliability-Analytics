import re
import numpy as np
import pandas as pd
from sqlalchemy import text
from db_config_85 import (
    get_engine,
    MASTER_PATH,
    DB_OUTPUT_DIR,
    EXPECTED_ROUTES,
    EXPECTED_SAMPLES,
    EXPECTED_SAMPLES_PER_ROUTE,
    FULL_REFRESH,
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    rename_map = {
        "sampleid": "sample_id",
        "sample": "sample_id",
        "specimen": "specimen_id",
        "specimenid": "specimen_id",
        "specimen_id_": "specimen_id",
        "cyclces_to_fatilure": "cycles_to_failure",
        "cycles_to_failures": "cycles_to_failure",
        "nf": "cycles_to_failure",
        "n_f": "cycles_to_failure",
        "ys": "ys_mpa",
        "yield_strength_mpa": "ys_mpa",
        "uts": "uts_mpa",
        "ultimate_tensile_strength_mpa": "uts_mpa",
        "elongation_%": "elongation_percent",
        "elongation": "elongation_percent",
        "hardness": "hardness_hv",
        "grain_size": "grain_size_um",
        "grain_size_µm": "grain_size_um",
        "grain_size_um_": "grain_size_um",
        "frequency": "frequency_hz",
        "temperature": "temperature_c",
        "route": "route_id",
        "process_type": "process_subtype",
        "processing_type": "process_subtype",
    }
    return df.rename(columns=rename_map)


def read_table(path) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    return pd.read_csv(path)


def infer_sample_no(specimen_id: str):
    text_id = str(specimen_id)
    match = re.search(r"(?:_s|_sample|sample|[-_])?(\d+)$", text_id, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def to_float_or_none(value):
    if pd.isna(value) or value == "":
        return None
    return float(value)


def main():
    print("=== START load_specimen_master_85 ===")
    DB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_PATH}")

    raw = read_table(MASTER_PATH)
    print("Raw columns:", raw.columns.tolist())
    df = normalize_columns(raw)
    print("Normalized columns:", df.columns.tolist())

    if "specimen_id" not in df.columns:
        if "sample_id" in df.columns:
            df["specimen_id"] = df["sample_id"]
        else:
            raise ValueError("Master file must contain specimen_id or sample_id for the 85-sample workflow.")

    required_cols = [
        "specimen_id",
        "route_id",
        "route_family",
        "process_subtype",
        "soak_hours",
        "ecap_angle_deg",
        "ys_mpa",
        "uts_mpa",
        "elongation_percent",
        "hardness_hv",
        "grain_size_um",
        "cycles_to_failure",
        "tsa",
        "frequency_hz",
        "temperature_c",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master file: {missing}")

    df["specimen_id"] = df["specimen_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["sample_no"] = df["specimen_id"].apply(infer_sample_no)

    # For this manuscript workflow, treat all 85 as experimental unless the file explicitly says otherwise.
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)
    df["source_file"] = str(MASTER_PATH)

    # Numeric conversion with strict checking.
    numeric_cols = [
        "soak_hours", "ecap_angle_deg", "ys_mpa", "uts_mpa", "elongation_percent",
        "hardness_hv", "grain_size_um", "cycles_to_failure", "tsa", "frequency_hz", "temperature_c"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    critical_numeric = [
        "ys_mpa", "uts_mpa", "elongation_percent", "hardness_hv", "grain_size_um",
        "cycles_to_failure", "tsa", "frequency_hz", "temperature_c"
    ]
    bad_numeric = {col: int(df[col].isna().sum()) for col in critical_numeric if df[col].isna().any()}
    if bad_numeric:
        raise ValueError(f"Critical numeric columns contain missing/non-numeric values: {bad_numeric}")

    if df["specimen_id"].duplicated().any():
        duplicates = df.loc[df["specimen_id"].duplicated(keep=False), "specimen_id"].tolist()
        raise ValueError(f"Duplicate specimen_id values found in master file: {duplicates[:20]}")

    n_samples = df["specimen_id"].nunique()
    n_routes = df["route_id"].nunique()
    route_counts = df.groupby("route_id")["specimen_id"].nunique().sort_index()
    print(f"Unique samples: {n_samples}")
    print(f"Unique routes : {n_routes}")
    print("Samples per route:")
    print(route_counts)

    if n_samples != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} samples, observed {n_samples}.")
    if n_routes != EXPECTED_ROUTES:
        raise ValueError(f"Expected {EXPECTED_ROUTES} routes, observed {n_routes}.")
    bad_routes = route_counts[route_counts != EXPECTED_SAMPLES_PER_ROUTE]
    if not bad_routes.empty:
        raise ValueError(f"Routes not having {EXPECTED_SAMPLES_PER_ROUTE} samples:\n{bad_routes}")

    route_df = df[["route_id", "route_family", "process_subtype", "soak_hours", "ecap_angle_deg"]].drop_duplicates("route_id")
    specimen_df = df[[
        "specimen_id", "route_id", "sample_no", "ys_mpa", "uts_mpa", "elongation_percent",
        "hardness_hv", "grain_size_um", "cycles_to_failure", "tsa", "frequency_hz",
        "temperature_c", "is_synthetic", "source_file"
    ]].copy()

    engine = get_engine()
    with engine.begin() as conn:
        dbname = conn.execute(text("SELECT current_database();")).scalar()
        print(f"\nConnected to database: {dbname}")

        if FULL_REFRESH:
            print("FULL_REFRESH=True -> clearing old DB rows before loading master data.")
            conn.execute(text("TRUNCATE TABLE loop_manifest, cycle_summary, specimen_processed, specimen, processing_route RESTART IDENTITY;"))

        route_df.to_sql("processing_route", conn, if_exists="append", index=False, method="multi")
        specimen_df.to_sql("specimen", conn, if_exists="append", index=False, method="multi")

        route_after = conn.execute(text("SELECT COUNT(*) FROM processing_route;")).scalar()
        specimen_after = conn.execute(text("SELECT COUNT(*) FROM specimen;")).scalar()
        print(f"After load -> processing_route: {route_after}, specimen: {specimen_after}")

    report_path = DB_OUTPUT_DIR / "load_specimen_master_85_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("load_specimen_master_85 report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Master file: {MASTER_PATH}\n")
        f.write(f"Unique samples: {n_samples}\n")
        f.write(f"Unique routes: {n_routes}\n")
        f.write("\nSamples per route:\n")
        f.write(route_counts.to_string())
        f.write("\n\nStatus: PASS for specimen master DB load\n")

    print(f"\n✅ Done load_specimen_master_85. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
