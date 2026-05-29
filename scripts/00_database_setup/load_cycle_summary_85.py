import pandas as pd
import numpy as np
from sqlalchemy import text
from db_config_85 import get_engine, CYCLE_PATH, DB_OUTPUT_DIR, EXPECTED_SAMPLES


def coalesce_duplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Pandas returns a DataFrame instead of a Series when df['column'] has duplicate
    column names. This caused pd.to_numeric(df[col]) to fail. This helper merges
    duplicate columns by taking the first non-empty value across the duplicated
    columns row-wise.
    """
    duplicate_names = sorted(df.columns[df.columns.duplicated()].unique().tolist())
    if not duplicate_names:
        return df, []

    out = pd.DataFrame(index=df.index)

    for col in pd.unique(df.columns):
        selected = df.loc[:, col]

        if isinstance(selected, pd.DataFrame):
            selected = selected.replace(r"^\s*$", np.nan, regex=True)
            out[col] = selected.bfill(axis=1).iloc[:, 0]
        else:
            out[col] = selected

    return out, duplicate_names


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percent", regex=False)
    )

    rename_map = {
        "sample_id": "specimen_id",
        "sampleid": "specimen_id",
        "sample": "specimen_id",
        "specimenid": "specimen_id",

        "cycles": "cycle_no",
        "cycle": "cycle_no",
        "cycle_number": "cycle_no",
        "cycle_number_": "cycle_no",

        "plastic_strain_amplitude": "psa",
        "plastic_strain_amp": "psa",
        "plastic_strain_amplitude_": "psa",

        "mean_stress": "mean_stress_mpa",
        "mean_stress_mpa_": "mean_stress_mpa",
        "mean_stress__mpa": "mean_stress_mpa",

        "max_stress": "max_stress_mpa",
        "maximum_stress_mpa": "max_stress_mpa",

        "min_stress": "min_stress_mpa",
        "minimum_stress_mpa": "min_stress_mpa",

        "unloading_modulus": "unloading_modulus_mpa",
        "unloading_modulus_mpa_": "unloading_modulus_mpa",

        "stress_amplitude": "stress_amplitude_mpa",
        "stress_amp_mpa": "stress_amplitude_mpa",
        "stress_amplitude_mpa_": "stress_amplitude_mpa",

        "route": "route_id",
        "routeid": "route_id",
    }

    df = df.rename(columns=rename_map)
    df, duplicate_names = coalesce_duplicate_columns(df)
    return df, duplicate_names


def main():
    print("=== START load_cycle_summary_85 ===")
    DB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CYCLE_PATH.exists():
        raise FileNotFoundError(f"Cycle summary file not found: {CYCLE_PATH}")

    raw = pd.read_csv(CYCLE_PATH, low_memory=False)
    print("Raw columns:", raw.columns.tolist())

    df, duplicate_names = normalize_columns(raw)
    print("Normalized columns:", df.columns.tolist())

    if duplicate_names:
        duplicate_report_path = DB_OUTPUT_DIR / "load_cycle_summary_85_duplicate_column_names.txt"
        with open(duplicate_report_path, "w", encoding="utf-8") as f:
            f.write("Duplicate column names were found after normalization and safely coalesced.\n")
            f.write("This usually happens when both cycle_no and cycles, or sample_id and specimen_id, exist in the same file.\n\n")
            f.write("Duplicate names:\n")
            for name in duplicate_names:
                f.write(f"- {name}\n")
        print(f"⚠️ Duplicate normalized columns coalesced: {duplicate_names}")
        print(f"   Duplicate-column report saved to: {duplicate_report_path}")

    required_cols = [
        "specimen_id",
        "cycle_no",
        "psa",
        "mean_stress_mpa",
        "max_stress_mpa",
        "min_stress_mpa",
        "unloading_modulus_mpa",
        "stress_amplitude_mpa",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in cycle summary: {missing}")

    if "route_id" not in df.columns:
        df["route_id"] = np.nan

    # Keep only required / useful columns to prevent accidental duplicated or unrelated fields.
    df = df[[
        "specimen_id",
        "route_id",
        "cycle_no",
        "psa",
        "mean_stress_mpa",
        "max_stress_mpa",
        "min_stress_mpa",
        "unloading_modulus_mpa",
        "stress_amplitude_mpa",
    ]].copy()

    df["specimen_id"] = df["specimen_id"].astype(str).str.strip()

    df["route_id"] = df["route_id"].replace(r"^\s*$", np.nan, regex=True)
    if df["route_id"].notna().any():
        df["route_id"] = df["route_id"].astype(str).str.strip()
    else:
        df["route_id"] = None

    numeric_cols = [
        "cycle_no",
        "psa",
        "mean_stress_mpa",
        "max_stress_mpa",
        "min_stress_mpa",
        "unloading_modulus_mpa",
        "stress_amplitude_mpa",
    ]

    for col in numeric_cols:
        # Safety fallback: if a duplicate column survived for any reason, coalesce here also.
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True).bfill(axis=1).iloc[:, 0]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    critical_cols = ["specimen_id", *numeric_cols]
    null_counts = {
        col: int(df[col].isna().sum())
        for col in critical_cols
        if df[col].isna().any()
    }
    if null_counts:
        null_report_path = DB_OUTPUT_DIR / "load_cycle_summary_85_null_or_non_numeric_rows.csv"
        bad_mask = pd.Series(False, index=df.index)
        for col in critical_cols:
            bad_mask = bad_mask | df[col].isna()
        df.loc[bad_mask].to_csv(null_report_path, index=False)
        raise ValueError(
            f"Missing/non-numeric critical cycle values found: {null_counts}. "
            f"Problem rows saved to {null_report_path}"
        )

    df["cycle_no"] = df["cycle_no"].astype(int)

    duplicate_count = int(df.duplicated(["specimen_id", "cycle_no"]).sum())
    if duplicate_count:
        dup_path = DB_OUTPUT_DIR / "load_cycle_summary_85_duplicate_rows.csv"
        df.loc[df.duplicated(["specimen_id", "cycle_no"], keep=False)].to_csv(dup_path, index=False)
        raise ValueError(
            f"Duplicate specimen_id + cycle_no rows found: {duplicate_count}. "
            f"Saved to {dup_path}"
        )

    cycle_df = df.copy()

    engine = get_engine()

    with engine.begin() as conn:
        dbname = conn.execute(text("SELECT current_database();")).scalar()
        print(f"\nConnected to database: {dbname}")

        specimen_ref = pd.read_sql(
            "SELECT specimen_id, route_id, cycles_to_failure FROM specimen;",
            conn,
        )

        known_ids = set(specimen_ref["specimen_id"].astype(str))
        cycle_ids = set(cycle_df["specimen_id"].astype(str))
        unknown_ids = sorted(cycle_ids - known_ids)

        if unknown_ids:
            unknown_path = DB_OUTPUT_DIR / "load_cycle_summary_85_unknown_specimen_ids.csv"
            pd.DataFrame({"unknown_specimen_id": unknown_ids}).to_csv(unknown_path, index=False)
            raise ValueError(
                f"Cycle file contains specimen_ids not present in specimen table. "
                f"First few: {unknown_ids[:30]}. Full list saved to {unknown_path}"
            )

        observed_samples = int(cycle_df["specimen_id"].nunique())
        if observed_samples != EXPECTED_SAMPLES:
            raise ValueError(
                f"Expected {EXPECTED_SAMPLES} unique specimens in cycle file, "
                f"observed {observed_samples}"
            )

        # Fill route_id from specimen table to keep DB rows consistent.
        cycle_df = (
            cycle_df
            .drop(columns=["route_id"])
            .merge(specimen_ref, on="specimen_id", how="left")
        )

        cycle_df["cycle_fraction"] = cycle_df["cycle_no"] / cycle_df["cycles_to_failure"]

        # Light sanity check: cycle_fraction should normally be near 0 to 1.
        # Some datasets may include exactly failure cycle or a few machine-export edge rows,
        # so this is recorded only as a warning report, not a hard stop.
        fraction_warning_mask = (cycle_df["cycle_fraction"] < 0) | (cycle_df["cycle_fraction"] > 1.05)
        fraction_warning_count = int(fraction_warning_mask.sum())
        if fraction_warning_count:
            frac_path = DB_OUTPUT_DIR / "load_cycle_summary_85_cycle_fraction_warnings.csv"
            cycle_df.loc[fraction_warning_mask].to_csv(frac_path, index=False)
            print(f"⚠️ Cycle fraction warning rows: {fraction_warning_count}. Saved to {frac_path}")

        cycle_df = cycle_df.drop(columns=["cycles_to_failure"])
        cycle_df = cycle_df[[
            "specimen_id",
            "route_id",
            "cycle_no",
            "cycle_fraction",
            "psa",
            "mean_stress_mpa",
            "max_stress_mpa",
            "min_stress_mpa",
            "unloading_modulus_mpa",
            "stress_amplitude_mpa",
        ]]

        before_count = conn.execute(text("SELECT COUNT(*) FROM cycle_summary;")).scalar()
        print(f"Before load -> cycle_summary: {before_count}")

        conn.execute(text("TRUNCATE TABLE loop_manifest, cycle_summary;"))

        cycle_df.to_sql(
            "cycle_summary",
            conn,
            if_exists="append",
            index=False,
            chunksize=5000,
            method="multi",
        )

        after_count = conn.execute(text("SELECT COUNT(*) FROM cycle_summary;")).scalar()
        print(f"After load  -> cycle_summary: {after_count}")

    route_counts = cycle_df.groupby("route_id")["specimen_id"].nunique().sort_index()
    specimen_cycle_counts = (
        cycle_df.groupby(["route_id", "specimen_id"])["cycle_no"]
        .agg(["min", "max", "count"])
        .reset_index()
        .sort_values(["route_id", "specimen_id"])
    )

    specimen_count_path = DB_OUTPUT_DIR / "load_cycle_summary_85_specimen_cycle_counts.csv"
    specimen_cycle_counts.to_csv(specimen_count_path, index=False)

    report_path = DB_OUTPUT_DIR / "load_cycle_summary_85_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("load_cycle_summary_85 report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Cycle file: {CYCLE_PATH}\n")
        f.write(f"Total cycle rows loaded: {len(cycle_df)}\n")
        f.write(f"Unique samples: {cycle_df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {cycle_df['route_id'].nunique()}\n")
        f.write(f"Duplicate specimen-cycle rows: {duplicate_count}\n")
        f.write(f"Duplicate normalized columns coalesced: {duplicate_names}\n")
        f.write(f"Cycle-fraction warning rows: {fraction_warning_count}\n")
        f.write("\nSamples per route:\n")
        f.write(route_counts.to_string())
        f.write(f"\n\nSpecimen cycle count file: {specimen_count_path}\n")
        f.write("\nStatus: PASS for cycle summary DB load\n")

    print(f"\n✅ Done load_cycle_summary_85. Report saved to: {report_path}")
    print(f"Specimen cycle-count report saved to: {specimen_count_path}")


if __name__ == "__main__":
    main()
