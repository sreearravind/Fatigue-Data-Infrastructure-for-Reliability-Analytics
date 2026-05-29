import numpy as np
import pandas as pd
from sqlalchemy import text
from db_config_85 import get_engine, MASTER_PATH, CYCLE_PATH, CLEANED_DIR, DB_OUTPUT_DIR, EXPECTED_SAMPLES


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def normalize_column_names(columns) -> pd.Index:
    """Standardise raw column names before applying rename rules."""
    return (
        pd.Index(columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )


def coalesce_duplicate_columns(df: pd.DataFrame):
    """
    After normalization/renaming, multiple raw columns may collapse into the same
    target name, e.g. 'cycles' and 'cycle_no' both becoming 'cycle_no'.

    Pandas returns a DataFrame when selecting a duplicated column name; this causes
    pd.to_numeric(df[col]) to fail. This function combines duplicated columns into
    one column by taking the first non-null/non-empty value across the duplicates.
    """
    cols = list(df.columns)
    duplicate_names = sorted({c for c in cols if cols.count(c) > 1})

    if not duplicate_names:
        return df, duplicate_names

    cleaned = pd.DataFrame(index=df.index)
    ordered_unique_cols = list(dict.fromkeys(cols))

    for col in ordered_unique_cols:
        subset = df.loc[:, df.columns == col]
        if subset.shape[1] == 1:
            cleaned[col] = subset.iloc[:, 0]
        else:
            # Treat empty strings as missing, then keep first available value.
            subset = subset.replace(r"^\s*$", np.nan, regex=True)
            cleaned[col] = subset.bfill(axis=1).iloc[:, 0]

    return cleaned, duplicate_names


def normalize_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = normalize_column_names(df.columns)

    rename_map = {
        "sample_id": "specimen_id",
        "sampleid": "specimen_id",
        "specimenid": "specimen_id",
        "cyclces_to_failure": "cycles_to_failure",
        "cyclces_to_fatilure": "cycles_to_failure",
        "cycles_to_fatilure": "cycles_to_failure",
        "cyclces_to_failures": "cycles_to_failure",
        "cycles_to_failure_nf": "cycles_to_failure",
        "nf": "cycles_to_failure",
        "n_f": "cycles_to_failure",
        "ys": "ys_mpa",
        "yield_strength_mpa": "ys_mpa",
        "yield_strength": "ys_mpa",
        "uts": "uts_mpa",
        "ultimate_tensile_strength_mpa": "uts_mpa",
        "ultimate_tensile_strength": "uts_mpa",
        "hardness": "hardness_hv",
        "hardness_vickers": "hardness_hv",
        "grain_size": "grain_size_um",
        "grain_size_µm": "grain_size_um",
        "grain_size_um_": "grain_size_um",
        "cycles": "cycle_no",
        "cycle": "cycle_no",
        "cycle_number": "cycle_no",
        "plastic_strain_amplitude": "psa",
        "plastic_strain_amp": "psa",
        "mean_stress": "mean_stress_mpa",
        "mean_stress_(mpa)": "mean_stress_mpa",
        "max_stress": "max_stress_mpa",
        "min_stress": "min_stress_mpa",
        "unloading_modulus": "unloading_modulus_mpa",
        "stress_amplitude": "stress_amplitude_mpa",
        "stress_amp_mpa": "stress_amplitude_mpa",
        "route": "route_id",
    }

    df = df.rename(columns=rename_map)
    df, duplicate_names = coalesce_duplicate_columns(df)
    return df, duplicate_names


def read_master(path) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required_cols, label: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}\nAvailable columns: {df.columns.tolist()}")


def to_numeric_required(df: pd.DataFrame, cols, label: str) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = {col: int(df[col].isna().sum()) for col in cols if df[col].isna().any()}
    if bad:
        raise ValueError(f"Missing/non-numeric values in {label}: {bad}")
    return df


def aggregate_cycle_features(master: pd.DataFrame, cycle: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cycle-level descriptors to one row per specimen.

    Important fix:
    The consolidated cycle file may already contain a column named
    'cycles_to_failure'. If we merge master['cycles_to_failure'] directly,
    pandas creates cycles_to_failure_x / cycles_to_failure_y and the original
    name disappears, causing KeyError: 'cycles_to_failure'.

    To avoid this, the cycle-file copy is treated as cycle-response data only:
    any existing cycles_to_failure-like column is dropped before merging the
    authoritative value from the master file.
    """
    cycle = cycle.copy()
    master_nf = master[["specimen_id", "cycles_to_failure"]].copy()

    # Keep master as the single authoritative source for Nf during stable-window calculation.
    cycle = cycle.drop(columns=["cycles_to_failure"], errors="ignore")

    cycle = cycle.merge(master_nf, on="specimen_id", how="left", validate="many_to_one")

    missing_nf = int(cycle["cycles_to_failure"].isna().sum())
    if missing_nf:
        raise ValueError(f"{missing_nf} cycle rows could not be matched with cycles_to_failure from master.")

    zero_or_negative_nf = int((cycle["cycles_to_failure"] <= 0).sum())
    if zero_or_negative_nf:
        raise ValueError(f"{zero_or_negative_nf} cycle rows have non-positive cycles_to_failure values.")

    cycle["cycle_fraction"] = cycle["cycle_no"] / cycle["cycles_to_failure"]
    cycle["is_stable"] = cycle["cycle_fraction"].between(0.30, 0.70, inclusive="both")

    feature_cols = ["psa", "mean_stress_mpa", "stress_amplitude_mpa", "unloading_modulus_mpa"]
    cycle = to_numeric_required(cycle, feature_cols, "cycle feature columns")

    overall = cycle.groupby("specimen_id").agg(
        psa_mean=("psa", "mean"),
        mean_stress_mean=("mean_stress_mpa", "mean"),
        stress_amp_mean=("stress_amplitude_mpa", "mean"),
        unloading_modulus_mean=("unloading_modulus_mpa", "mean"),
    )

    stable = cycle[cycle["is_stable"]].groupby("specimen_id").agg(
        psa_stable_mean=("psa", "mean"),
        psa_stable_std=("psa", "std"),
        mean_stress_stable_mean=("mean_stress_mpa", "mean"),
        stress_amp_stable_mean=("stress_amplitude_mpa", "mean"),
        unloading_modulus_stable_mean=("unloading_modulus_mpa", "mean"),
    )

    agg = overall.join(stable, how="left").reset_index()
    return agg


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------
def main():
    print("=== START load_specimen_processed_85 | v3 fixed ===")
    DB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_PATH}")
    if not CYCLE_PATH.exists():
        raise FileNotFoundError(f"Cycle summary file not found: {CYCLE_PATH}")

    raw_master = read_master(MASTER_PATH)
    raw_cycle = pd.read_csv(CYCLE_PATH)

    print("Raw master columns:", raw_master.columns.tolist())
    print("Raw cycle columns:", raw_cycle.columns.tolist())

    master, master_duplicate_cols = normalize_columns(raw_master)
    cycle, cycle_duplicate_cols = normalize_columns(raw_cycle)

    print("Normalized master columns:", master.columns.tolist())
    print("Normalized cycle columns:", cycle.columns.tolist())

    # Save duplicate-column diagnostic if needed.
    duplicate_report_path = DB_OUTPUT_DIR / "load_specimen_processed_85_duplicate_column_names.txt"
    if master_duplicate_cols or cycle_duplicate_cols:
        with open(duplicate_report_path, "w", encoding="utf-8") as f:
            f.write("Duplicate column names after normalization/renaming\n")
            f.write("=" * 60 + "\n")
            f.write(f"Master duplicate columns: {master_duplicate_cols}\n")
            f.write(f"Cycle duplicate columns: {cycle_duplicate_cols}\n")
            f.write("\nThe script coalesced duplicate columns using the first non-null/non-empty value.\n")
        print(f"Duplicate-column diagnostic saved to: {duplicate_report_path}")

    required_master = ["specimen_id", "cycles_to_failure", "ys_mpa", "uts_mpa", "grain_size_um"]
    required_cycle = [
        "specimen_id", "cycle_no", "psa", "mean_stress_mpa",
        "stress_amplitude_mpa", "unloading_modulus_mpa"
    ]
    require_columns(master, required_master, "master")
    require_columns(cycle, required_cycle, "cycle")

    master["specimen_id"] = master["specimen_id"].astype(str).str.strip()
    cycle["specimen_id"] = cycle["specimen_id"].astype(str).str.strip()

    master = to_numeric_required(
        master,
        ["cycles_to_failure", "ys_mpa", "uts_mpa", "grain_size_um"],
        "master numeric columns",
    )
    cycle = to_numeric_required(
        cycle,
        ["cycle_no"],
        "cycle number column",
    )
    cycle["cycle_no"] = cycle["cycle_no"].astype(int)

    master_sample_count = master["specimen_id"].nunique()
    cycle_sample_count = cycle["specimen_id"].nunique()

    if master_sample_count != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} master samples, observed {master_sample_count}")
    if cycle_sample_count != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} cycle samples, observed {cycle_sample_count}")

    missing_in_cycle = sorted(set(master["specimen_id"]) - set(cycle["specimen_id"]))
    extra_in_cycle = sorted(set(cycle["specimen_id"]) - set(master["specimen_id"]))
    if missing_in_cycle or extra_in_cycle:
        raise ValueError(
            "Master-cycle specimen mismatch found. "
            f"Missing in cycle: {missing_in_cycle[:30]}; Extra in cycle: {extra_in_cycle[:30]}"
        )

    proc = master[["specimen_id", "cycles_to_failure", "ys_mpa", "uts_mpa", "grain_size_um"]].copy()
    proc["log_nf"] = np.log10(proc["cycles_to_failure"])
    proc["d_inv_sqrt"] = 1 / np.sqrt(proc["grain_size_um"])
    proc["strength_ratio"] = proc["ys_mpa"] / proc["uts_mpa"]
    proc["fatigue_efficiency"] = proc["log_nf"] / proc["ys_mpa"]

    cycle_agg = aggregate_cycle_features(proc, cycle)
    proc = proc.merge(cycle_agg, on="specimen_id", how="left")

    proc_cols = [
        "specimen_id", "log_nf", "d_inv_sqrt", "strength_ratio", "fatigue_efficiency",
        "psa_mean", "psa_stable_mean", "psa_stable_std",
        "mean_stress_mean", "mean_stress_stable_mean",
        "stress_amp_mean", "stress_amp_stable_mean",
        "unloading_modulus_mean", "unloading_modulus_stable_mean",
    ]
    require_columns(proc, proc_cols, "processed output")
    proc_df = proc[proc_cols].copy()

    # Check aggregated feature completeness before DB insertion.
    feature_null_counts = proc_df.drop(columns=["specimen_id"]).isna().sum()
    feature_null_counts = feature_null_counts[feature_null_counts > 0]
    if not feature_null_counts.empty:
        null_report_path = DB_OUTPUT_DIR / "load_specimen_processed_85_feature_null_counts.csv"
        feature_null_counts.rename("null_count").to_csv(null_report_path)
        print(f"Warning: some processed features contain nulls. Saved count report to: {null_report_path}")

    out_features = CLEANED_DIR / "sample_level_features_85_basic_from_db_stage.csv"
    proc.to_csv(out_features, index=False)
    print(f"Saved basic sample-level feature preview to: {out_features}")

    engine = get_engine()
    with engine.begin() as conn:
        dbname = conn.execute(text("SELECT current_database();")).scalar()
        print(f"\nConnected to database: {dbname}")

        valid_ids = set(pd.read_sql("SELECT specimen_id FROM specimen;", conn)["specimen_id"])
        unknown_ids = sorted(set(proc_df["specimen_id"]) - valid_ids)
        if unknown_ids:
            raise ValueError(f"Processed file has specimen_ids not present in specimen table: {unknown_ids[:30]}")

        before_count = conn.execute(text("SELECT COUNT(*) FROM specimen_processed;")).scalar()
        print(f"Before load -> specimen_processed: {before_count}")
        conn.execute(text("TRUNCATE TABLE specimen_processed;"))
        proc_df.to_sql("specimen_processed", conn, if_exists="append", index=False, chunksize=5000, method="multi")
        after_count = conn.execute(text("SELECT COUNT(*) FROM specimen_processed;")).scalar()
        print(f"After load  -> specimen_processed: {after_count}")

    report_path = DB_OUTPUT_DIR / "load_specimen_processed_85_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("load_specimen_processed_85 report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Master samples: {master_sample_count}\n")
        f.write(f"Cycle samples: {cycle_sample_count}\n")
        f.write(f"Rows loaded to specimen_processed: {len(proc_df)}\n")
        f.write(f"Feature preview saved: {out_features}\n")
        f.write("Stable window: 30% to 70% of fatigue life\n")
        if master_duplicate_cols or cycle_duplicate_cols:
            f.write(f"Duplicate-column diagnostic: {duplicate_report_path}\n")
        f.write("Status: PASS for specimen_processed DB load\n")

    print(f"\n✅ Done load_specimen_processed_85. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
