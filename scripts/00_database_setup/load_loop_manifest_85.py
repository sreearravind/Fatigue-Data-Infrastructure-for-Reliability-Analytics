"""
load_loop_manifest_85.py

85-sample workflow loader for loop_manifest.

This script supports two modes:
1. If loop_manifest_85.csv exists, load it.
2. If loop_manifest_85.csv does not exist, generate a summary-level manifest
   from cycle_summary_85_samples_consolidated.csv.

The script is robust to duplicate column names created during normalization,
for example when both cycle_no and cycles exist in the same CSV.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from db_config_85 import get_engine, LOOP_MANIFEST_PATH, CYCLE_PATH, DB_OUTPUT_DIR


RENAME_MAP = {
    "sample_id": "specimen_id",
    "sampleid": "specimen_id",
    "specimenid": "specimen_id",
    "cycles": "cycle_no",
    "cycle": "cycle_no",
    "cycle_number": "cycle_no",
    "rows": "n_rows",
    "row_count": "n_rows",
    "path": "file_path",
}


def _clean_col_name(col: object) -> str:
    """Return a normalized lowercase DB-friendly column name."""
    return (
        str(col)
        .strip()
        .lower()
        .replace("\ufeff", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def coalesce_duplicate_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Pandas returns a DataFrame instead of a Series when a column name is duplicated.
    This function merges duplicated columns row-wise using the first non-null value.
    """
    duplicate_names = sorted(set(df.columns[df.columns.duplicated()].tolist()))
    if not duplicate_names:
        return df

    DB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = DB_OUTPUT_DIR / f"load_loop_manifest_85_duplicate_columns_{label}.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write(f"Duplicate normalized column names detected in {label}\n")
        f.write("=" * 70 + "\n")
        for name in duplicate_names:
            f.write(f"{name}: {int((df.columns == name).sum())} columns\n")

    print(f"Duplicate-column diagnostic saved to: {diag_path}")

    result = pd.DataFrame(index=df.index)
    for name in pd.Index(df.columns).drop_duplicates():
        same = df.loc[:, df.columns == name]
        if same.shape[1] == 1:
            result[name] = same.iloc[:, 0]
        else:
            # First non-null value across duplicate columns.
            result[name] = same.bfill(axis=1).iloc[:, 0]
    return result


def normalize_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_col_name(c) for c in df.columns]
    df = df.rename(columns=RENAME_MAP)
    df = coalesce_duplicate_columns(df, label=label)
    return df


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {label} columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def build_summary_manifest_from_cycle_summary() -> pd.DataFrame:
    print(
        "No loop_manifest_85.csv found. "
        "Generating summary-level manifest from consolidated cycle summary."
    )

    if not CYCLE_PATH.exists():
        raise FileNotFoundError(f"Cycle summary file not found: {CYCLE_PATH}")

    raw_cycle = pd.read_csv(CYCLE_PATH)
    print("Raw cycle columns:", raw_cycle.columns.tolist())
    cycle = normalize_columns(raw_cycle, label="cycle_summary")
    print("Normalized cycle columns:", cycle.columns.tolist())

    ensure_columns(cycle, ["specimen_id", "cycle_no"], "cycle summary")

    manifest = cycle[["specimen_id", "cycle_no"]].copy()
    manifest["specimen_id"] = manifest["specimen_id"].astype(str).str.strip()
    manifest["cycle_no"] = pd.to_numeric(manifest["cycle_no"], errors="coerce")

    bad_cycle_no = int(manifest["cycle_no"].isna().sum())
    bad_specimen = int((manifest["specimen_id"] == "").sum())
    if bad_cycle_no or bad_specimen:
        bad_path = DB_OUTPUT_DIR / "load_loop_manifest_85_bad_generated_manifest_rows.csv"
        manifest.loc[manifest["cycle_no"].isna() | (manifest["specimen_id"] == "")].to_csv(bad_path, index=False)
        raise ValueError(
            f"Invalid generated manifest rows found. "
            f"bad_cycle_no={bad_cycle_no}, bad_specimen_id={bad_specimen}. "
            f"Saved to: {bad_path}"
        )

    manifest["cycle_no"] = manifest["cycle_no"].astype(int)
    manifest["file_path"] = str(CYCLE_PATH)
    manifest["n_rows"] = 1
    manifest["issues"] = "summary_level_manifest_generated_from_cycle_summary"
    manifest["file_sha256"] = None

    return manifest.drop_duplicates(["specimen_id", "cycle_no"]).reset_index(drop=True)


def read_existing_manifest() -> pd.DataFrame:
    print(f"Reading loop manifest from: {LOOP_MANIFEST_PATH}")
    raw = pd.read_csv(LOOP_MANIFEST_PATH)
    print("Raw manifest columns:", raw.columns.tolist())
    df = normalize_columns(raw, label="loop_manifest")
    print("Normalized manifest columns:", df.columns.tolist())

    required = ["specimen_id", "cycle_no", "file_path", "n_rows", "issues"]
    ensure_columns(df, required, "loop manifest")
    if "file_sha256" not in df.columns:
        df["file_sha256"] = None
    return df[["specimen_id", "cycle_no", "file_path", "n_rows", "issues", "file_sha256"]].copy()


def clean_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    manifest_df = manifest_df.copy()
    manifest_df["specimen_id"] = manifest_df["specimen_id"].astype(str).str.strip()
    manifest_df["cycle_no"] = pd.to_numeric(manifest_df["cycle_no"], errors="coerce")
    manifest_df["n_rows"] = pd.to_numeric(manifest_df["n_rows"], errors="coerce").fillna(1)

    null_counts = {
        col: int(manifest_df[col].isna().sum())
        for col in ["cycle_no", "n_rows"]
        if int(manifest_df[col].isna().sum()) > 0
    }
    blank_specimen = int((manifest_df["specimen_id"] == "").sum())
    if null_counts or blank_specimen:
        bad_path = DB_OUTPUT_DIR / "load_loop_manifest_85_invalid_rows.csv"
        mask = manifest_df["cycle_no"].isna() | manifest_df["n_rows"].isna() | (manifest_df["specimen_id"] == "")
        manifest_df.loc[mask].to_csv(bad_path, index=False)
        raise ValueError(
            f"Invalid manifest rows found. null_counts={null_counts}, "
            f"blank_specimen_id={blank_specimen}. Saved to: {bad_path}"
        )

    manifest_df["cycle_no"] = manifest_df["cycle_no"].astype(int)
    manifest_df["n_rows"] = manifest_df["n_rows"].astype(int)
    manifest_df["file_path"] = manifest_df["file_path"].fillna("").astype(str)
    manifest_df["issues"] = manifest_df["issues"].fillna("").astype(str)
    manifest_df["file_sha256"] = manifest_df["file_sha256"].where(manifest_df["file_sha256"].notna(), None)

    manifest_df = manifest_df[["specimen_id", "cycle_no", "file_path", "n_rows", "issues", "file_sha256"]]

    dup_count = int(manifest_df.duplicated(["specimen_id", "cycle_no"]).sum())
    if dup_count:
        dup_path = DB_OUTPUT_DIR / "load_loop_manifest_85_duplicate_rows.csv"
        manifest_df.loc[manifest_df.duplicated(["specimen_id", "cycle_no"], keep=False)].to_csv(dup_path, index=False)
        raise ValueError(
            f"Duplicate specimen_id + cycle_no manifest rows found: {dup_count}. "
            f"Saved to: {dup_path}"
        )

    return manifest_df


def validate_against_db(manifest_df: pd.DataFrame, conn) -> None:
    valid_specimens = set(pd.read_sql("SELECT specimen_id FROM specimen;", conn)["specimen_id"].astype(str))
    unknown_specimens = sorted(set(manifest_df["specimen_id"]) - valid_specimens)
    if unknown_specimens:
        raise ValueError(
            "Manifest contains specimen_ids not present in specimen table. "
            f"Examples: {unknown_specimens[:30]}"
        )

    valid_pairs = pd.read_sql("SELECT specimen_id, cycle_no FROM cycle_summary;", conn)
    valid_pairs["specimen_id"] = valid_pairs["specimen_id"].astype(str)
    valid_pairs["cycle_no"] = valid_pairs["cycle_no"].astype(int)

    valid_pairs_set = set(zip(valid_pairs["specimen_id"], valid_pairs["cycle_no"]))
    manifest_pairs = set(zip(manifest_df["specimen_id"], manifest_df["cycle_no"]))

    extra_pairs = list(manifest_pairs - valid_pairs_set)
    if extra_pairs:
        raise ValueError(
            "Manifest contains specimen-cycle pairs not present in cycle_summary. "
            f"Examples: {extra_pairs[:10]}"
        )

    missing_manifest_pairs = list(valid_pairs_set - manifest_pairs)
    if missing_manifest_pairs:
        print(
            "Warning: loop_manifest has fewer pairs than cycle_summary. "
            f"Missing examples: {missing_manifest_pairs[:10]}"
        )


def main() -> None:
    print("=== START load_loop_manifest_85 ===")
    DB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if LOOP_MANIFEST_PATH.exists():
        manifest_df = read_existing_manifest()
        manifest_source = "existing loop_manifest_85.csv"
    else:
        manifest_df = build_summary_manifest_from_cycle_summary()
        manifest_source = "generated from cycle_summary_85_samples_consolidated.csv"

    manifest_df = clean_manifest(manifest_df)

    engine = get_engine()
    with engine.begin() as conn:
        dbname = conn.execute(text("SELECT current_database();")).scalar()
        print(f"\nConnected to database: {dbname}")

        validate_against_db(manifest_df, conn)

        before_count = conn.execute(text("SELECT COUNT(*) FROM loop_manifest;")).scalar()
        print(f"Before load -> loop_manifest: {before_count}")

        conn.execute(text("TRUNCATE TABLE loop_manifest;"))
        manifest_df.to_sql(
            "loop_manifest",
            conn,
            if_exists="append",
            index=False,
            chunksize=5000,
            method="multi",
        )

        after_count = conn.execute(text("SELECT COUNT(*) FROM loop_manifest;")).scalar()
        print(f"After load  -> loop_manifest: {after_count}")

    report_path = DB_OUTPUT_DIR / "load_loop_manifest_85_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("load_loop_manifest_85 report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Rows loaded: {len(manifest_df)}\n")
        f.write(f"Manifest source: {manifest_source}\n")
        f.write(f"Unique specimens: {manifest_df['specimen_id'].nunique()}\n")
        f.write("Status: PASS for loop_manifest DB load\n")

    print(f"\n✅ Done load_loop_manifest_85. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
