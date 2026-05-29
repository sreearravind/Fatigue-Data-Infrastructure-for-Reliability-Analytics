"""
00_prepare_cycle_summary_85.py

Purpose
-------
Consolidate the cycle-summary data for the modified 85-sample fatigue dataset.

Input files
-----------
1) Original 17 experimental route-level cycle summary file, treated as Sample-1
2) Consolidated 68 augmented cycle summary file, treated as Samples-2 to -5

Output
------
cycle_summary_85_samples_consolidated.csv

The script also writes validation reports into the same output folder.

Author workflow note
--------------------
This script is intended to be the first step in the 85-sample workflow before
DB loading, descriptor aggregation, statistical analysis, and ML modelling.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. USER CONFIGURATION
# -----------------------------------------------------------------------------

INPUT_SAMPLE1_CYCLE_SUMMARY = Path(
    r"data/01_input/Cycle summary_17 routes_samples_no_1.csv"
)

INPUT_AUGMENTED_68_CYCLE_SUMMARY = Path(
    r"data/01_input/synthetic_cycle_summary_68_samples_consolidated.csv"
)

OUTPUT_DIR = Path(
    r"data\02_cleaned"
)

OUTPUT_FILE = OUTPUT_DIR / "cycle_summary_85_samples_consolidated.csv"

EXPECTED_ROUTE_COUNT = 17
EXPECTED_SAMPLE_COUNT = 85
EXPECTED_SAMPLES_PER_ROUTE = 5

# If True, validation failure stops the script after writing report files.
STRICT_VALIDATION = True

# For the original 17-route file, if sample_id is absent, Sample-1 IDs will be
# created as route_id + this suffix. Example: T5_S1, CT6_S1, ECAP90_S1.
SAMPLE1_SUFFIX = "S1"


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read CSV with common encodings and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error: Optional[Exception] = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:  # pragma: no cover - practical runtime fallback
            last_error = exc

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def clean_column_name(col: object) -> str:
    """Clean column names while preserving readability."""
    text = str(col).replace("\ufeff", "").replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM/spaces from column names."""
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def canonical(col: str) -> str:
    """Canonical form for fuzzy column matching."""
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Find a column using flexible candidate matching."""
    lookup = {canonical(c): c for c in df.columns}
    for cand in candidates:
        key = canonical(cand)
        if key in lookup:
            return lookup[key]
    return None


def infer_route_from_text(value: object) -> str:
    """
    Infer processing route from a sample/specimen identifier.

    Examples:
    T5 -> T5
    T5_S1 -> T5
    T5-1 -> T5
    CT6_S4 -> CT6
    ECAP90_S2 -> ECAP90
    ECAP120-5 -> ECAP120
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()
    text = text.replace(" ", "_")

    # Strong match for known route families.
    strong = re.match(r"^(ECAP90|ECAP120|T6A|T6W|T5|CT\d+)", text, flags=re.IGNORECASE)
    if strong:
        return strong.group(1).upper().replace("ECAP", "ECAP")

    # Fallback: remove trailing sample notation such as _S1, -S1, _1, -1.
    text = re.sub(r"([_\-]?S?\d+)$", "", text, flags=re.IGNORECASE)
    return text.upper()


def infer_sample_no_from_text(value: object) -> Optional[int]:
    """Infer sample number from a sample/specimen identifier if present."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(?:[_\-]?S|[_\-])([1-5])$", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def standardize_route_sample_columns(
    df: pd.DataFrame,
    source_label: str,
    default_sample_no: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ensure route_id and sample_id exist.

    The function preserves existing columns where possible and adds/standardizes:
    - route_id
    - sample_no
    - sample_id
    - source_group
    """
    df = df.copy()

    route_col = find_column(
        df,
        [
            "route_id", "route", "processing_route", "processing route",
            "route family", "route_family", "specimen route",
        ],
    )
    sample_col = find_column(
        df,
        [
            "sample_id", "sample id", "specimen_id", "specimen id",
            "sample", "specimen", "sample_name", "sample name",
        ],
    )
    sample_no_col = find_column(
        df,
        ["sample_no", "sample no", "sample_number", "sample number", "replicate", "replicate_no"],
    )

    # Build route_id.
    if route_col is not None:
        df["route_id"] = df[route_col].astype(str).str.strip()
    elif sample_col is not None:
        df["route_id"] = df[sample_col].apply(infer_route_from_text)
    else:
        raise ValueError(
            "Could not identify route/sample information. "
            "Expected one of: route_id, route, sample_id, specimen_id."
        )

    df["route_id"] = df["route_id"].apply(infer_route_from_text)

    # Build sample_no.
    if sample_no_col is not None:
        df["sample_no"] = pd.to_numeric(df[sample_no_col], errors="coerce")
    elif sample_col is not None:
        inferred = df[sample_col].apply(infer_sample_no_from_text)
        df["sample_no"] = pd.to_numeric(inferred, errors="coerce")
    else:
        df["sample_no"] = np.nan

    if default_sample_no is not None:
        df["sample_no"] = df["sample_no"].fillna(default_sample_no)

    # Build sample_id.
    if sample_col is not None:
        existing_sample_id = df[sample_col].astype(str).str.strip()
    else:
        existing_sample_id = pd.Series([None] * len(df), index=df.index, dtype="object")

    # If sample id is only the route id or unavailable, create route_Sn.
    def build_sample_id(row) -> str:
        route = str(row["route_id"]).strip()
        sample_no = row.get("sample_no", np.nan)
        existing = str(existing_sample_id.loc[row.name]).strip() if existing_sample_id.loc[row.name] is not None else ""

        existing_route = infer_route_from_text(existing) if existing and existing.lower() != "nan" else ""
        existing_sample_no = infer_sample_no_from_text(existing)

        if existing and existing.lower() != "nan" and existing_sample_no is not None:
            # Normalize separator only mildly: keep existing sample ID to match master files.
            return existing

        if existing and existing.lower() != "nan" and existing_route and canonical(existing) != canonical(route):
            # Existing sample identifier contains more than route-level information.
            return existing

        if pd.notna(sample_no):
            return f"{route}_S{int(sample_no)}"

        # Last fallback; validation will catch if this creates only route-level IDs.
        return route

    df["sample_id"] = df.apply(build_sample_id, axis=1)
    df["source_group"] = source_label

    return df


def standardize_cycle_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure cycle_no exists if a recognizable cycle column is present."""
    df = df.copy()
    cycle_col = find_column(
        df,
        [
            "cycle_no", "cycle no", "cycle", "cycles", "cycle_number",
            "cycle number", "N", "n_cycle", "n cycles",
        ],
    )

    if cycle_col is None:
        # Keep script flexible; validation will report cycle_no as missing.
        df["cycle_no"] = np.nan
    else:
        df["cycle_no"] = pd.to_numeric(df[cycle_col], errors="coerce")

    return df


def add_traceability_columns(df: pd.DataFrame, source_file: Path, source_label: str) -> pd.DataFrame:
    """Add traceability metadata without disturbing scientific columns."""
    df = df.copy()
    df["source_file"] = source_file.name
    df["source_group"] = source_label
    return df


def build_route_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create route-wise sample and cycle-count summary."""
    grouped = (
        df.groupby(["route_id", "sample_id"], dropna=False)
        .agg(
            cycle_rows=("cycle_no", "size"),
            valid_cycle_no_count=("cycle_no", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            min_cycle_no=("cycle_no", "min"),
            max_cycle_no=("cycle_no", "max"),
            source_groups=("source_group", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )

    return grouped.sort_values(["route_id", "sample_id"]).reset_index(drop=True)


def find_missing_cycle_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check missing cycle numbers within each sample if cycle numbers are integer-like.

    This report is diagnostic. It does not assume every fatigue test must have the
    same number of cycles; it only checks gaps inside each sample's observed range.
    """
    records = []
    for (route_id, sample_id), g in df.groupby(["route_id", "sample_id"], dropna=False):
        cycles = pd.to_numeric(g["cycle_no"], errors="coerce").dropna()
        if cycles.empty:
            records.append(
                {
                    "route_id": route_id,
                    "sample_id": sample_id,
                    "status": "NO_VALID_CYCLE_NUMBERS",
                    "min_cycle_no": np.nan,
                    "max_cycle_no": np.nan,
                    "observed_cycle_count": 0,
                    "missing_cycle_count_within_range": np.nan,
                    "first_missing_cycles": "",
                }
            )
            continue

        # Only check sequence if all values are integer-like.
        integer_like = np.all(np.isclose(cycles, np.round(cycles)))
        if not integer_like:
            records.append(
                {
                    "route_id": route_id,
                    "sample_id": sample_id,
                    "status": "NON_INTEGER_CYCLE_NUMBERS_SEQUENCE_NOT_CHECKED",
                    "min_cycle_no": cycles.min(),
                    "max_cycle_no": cycles.max(),
                    "observed_cycle_count": len(cycles),
                    "missing_cycle_count_within_range": np.nan,
                    "first_missing_cycles": "",
                }
            )
            continue

        unique_cycles = set(cycles.astype(int).tolist())
        min_cycle = int(min(unique_cycles))
        max_cycle = int(max(unique_cycles))
        expected_cycles = set(range(min_cycle, max_cycle + 1))
        missing = sorted(expected_cycles - unique_cycles)

        records.append(
            {
                "route_id": route_id,
                "sample_id": sample_id,
                "status": "OK" if not missing else "MISSING_CYCLES_WITHIN_RANGE",
                "min_cycle_no": min_cycle,
                "max_cycle_no": max_cycle,
                "observed_cycle_count": len(unique_cycles),
                "missing_cycle_count_within_range": len(missing),
                "first_missing_cycles": ",".join(map(str, missing[:30])),
            }
        )

    return pd.DataFrame(records).sort_values(["route_id", "sample_id"]).reset_index(drop=True)


def validation_overview(df: pd.DataFrame, route_sample_summary: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Create high-level validation overview and return pass/fail."""
    route_count = int(df["route_id"].nunique(dropna=True))
    sample_count = int(df["sample_id"].nunique(dropna=True))
    total_rows = int(len(df))
    missing_route_id = int(df["route_id"].isna().sum() + (df["route_id"].astype(str).str.strip() == "").sum())
    missing_sample_id = int(df["sample_id"].isna().sum() + (df["sample_id"].astype(str).str.strip() == "").sum())
    missing_cycle_no = int(pd.to_numeric(df["cycle_no"], errors="coerce").isna().sum())

    samples_per_route = route_sample_summary.groupby("route_id")["sample_id"].nunique()
    routes_with_expected_samples = int((samples_per_route == EXPECTED_SAMPLES_PER_ROUTE).sum())
    routes_with_unexpected_samples = int((samples_per_route != EXPECTED_SAMPLES_PER_ROUTE).sum())

    duplicate_subset = ["sample_id", "cycle_no"]
    duplicate_sample_cycle_rows = int(df.duplicated(subset=duplicate_subset, keep=False).sum())

    checks = [
        {
            "check": "Total row count",
            "expected": "More than 0",
            "observed": total_rows,
            "status": "PASS" if total_rows > 0 else "FAIL",
        },
        {
            "check": "Unique route count",
            "expected": EXPECTED_ROUTE_COUNT,
            "observed": route_count,
            "status": "PASS" if route_count == EXPECTED_ROUTE_COUNT else "FAIL",
        },
        {
            "check": "Unique sample count",
            "expected": EXPECTED_SAMPLE_COUNT,
            "observed": sample_count,
            "status": "PASS" if sample_count == EXPECTED_SAMPLE_COUNT else "FAIL",
        },
        {
            "check": "Routes with exactly five samples",
            "expected": EXPECTED_ROUTE_COUNT,
            "observed": routes_with_expected_samples,
            "status": "PASS" if routes_with_expected_samples == EXPECTED_ROUTE_COUNT else "FAIL",
        },
        {
            "check": "Routes with unexpected sample count",
            "expected": 0,
            "observed": routes_with_unexpected_samples,
            "status": "PASS" if routes_with_unexpected_samples == 0 else "FAIL",
        },
        {
            "check": "Missing route_id values",
            "expected": 0,
            "observed": missing_route_id,
            "status": "PASS" if missing_route_id == 0 else "FAIL",
        },
        {
            "check": "Missing sample_id values",
            "expected": 0,
            "observed": missing_sample_id,
            "status": "PASS" if missing_sample_id == 0 else "FAIL",
        },
        {
            "check": "Missing cycle_no values",
            "expected": 0,
            "observed": missing_cycle_no,
            "status": "PASS" if missing_cycle_no == 0 else "FAIL",
        },
        {
            "check": "Duplicate sample_id + cycle_no rows",
            "expected": 0,
            "observed": duplicate_sample_cycle_rows,
            "status": "PASS" if duplicate_sample_cycle_rows == 0 else "WARNING",
        },
    ]

    overview = pd.DataFrame(checks)
    hard_fail = bool((overview["status"] == "FAIL").any())
    return overview, not hard_fail


def write_text_report(
    output_path: Path,
    overview: pd.DataFrame,
    route_sample_summary: pd.DataFrame,
    missing_sequence_report: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """Write a compact human-readable validation report."""
    route_counts = route_sample_summary.groupby("route_id")["sample_id"].nunique().sort_index()
    sequence_issue_count = int((missing_sequence_report["status"] != "OK").sum()) if not missing_sequence_report.empty else 0

    lines = []
    lines.append("cycle_summary_85_samples_consolidated.csv validation report")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"Output file: {OUTPUT_FILE}")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Unique routes: {df['route_id'].nunique(dropna=True)}")
    lines.append(f"Unique samples: {df['sample_id'].nunique(dropna=True)}")
    lines.append("")
    lines.append("Validation overview:")
    lines.append(overview.to_string(index=False))
    lines.append("")
    lines.append("Samples per route:")
    lines.append(route_counts.to_string())
    lines.append("")
    lines.append(f"Sequence diagnostic issue count: {sequence_issue_count}")
    lines.append("See cycle_summary_85_missing_cycle_sequence_report.csv for details.")
    lines.append("")
    lines.append("Important ML note:")
    lines.append("Use this consolidated cycle file for descriptor aggregation only.")
    lines.append("Do not train ML directly on cycle-level rows as independent samples.")
    lines.append("The ML-ready dataset should be one row per sample, i.e., 85 rows after aggregation.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# 3. MAIN WORKFLOW
# -----------------------------------------------------------------------------

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading input files...")
    sample1 = normalize_columns(read_csv_flexible(INPUT_SAMPLE1_CYCLE_SUMMARY))
    augmented68 = normalize_columns(read_csv_flexible(INPUT_AUGMENTED_68_CYCLE_SUMMARY))

    print(f"Sample-1 file rows: {len(sample1):,}")
    print(f"Augmented-68 file rows: {len(augmented68):,}")

    sample1 = standardize_route_sample_columns(
        sample1,
        source_label="sample_1_original_17_routes",
        default_sample_no=1,
    )
    augmented68 = standardize_route_sample_columns(
        augmented68,
        source_label="samples_2_to_5_augmented_68",
        default_sample_no=None,
    )

    sample1 = standardize_cycle_column(sample1)
    augmented68 = standardize_cycle_column(augmented68)

    sample1 = add_traceability_columns(
        sample1,
        source_file=INPUT_SAMPLE1_CYCLE_SUMMARY,
        source_label="sample_1_original_17_routes",
    )
    augmented68 = add_traceability_columns(
        augmented68,
        source_file=INPUT_AUGMENTED_68_CYCLE_SUMMARY,
        source_label="samples_2_to_5_augmented_68",
    )

    print("Concatenating files...")
    combined = pd.concat([sample1, augmented68], ignore_index=True, sort=False)

    # Keep important columns first for easier inspection.
    priority_cols = ["route_id", "sample_no", "sample_id", "cycle_no", "source_group", "source_file"]
    other_cols = [c for c in combined.columns if c not in priority_cols]
    combined = combined[priority_cols + other_cols]

    # Sort for deterministic output.
    combined["sample_no_sort"] = pd.to_numeric(combined["sample_no"], errors="coerce")
    combined["cycle_no_sort"] = pd.to_numeric(combined["cycle_no"], errors="coerce")
    combined = combined.sort_values(
        by=["route_id", "sample_no_sort", "sample_id", "cycle_no_sort"],
        kind="mergesort",
    ).drop(columns=["sample_no_sort", "cycle_no_sort"])

    print("Building validation reports...")
    route_sample_summary = build_route_sample_summary(combined)
    missing_sequence_report = find_missing_cycle_sequences(combined)
    overview, validation_passed = validation_overview(combined, route_sample_summary)

    duplicates = combined[combined.duplicated(subset=["sample_id", "cycle_no"], keep=False)].copy()

    print("Writing outputs...")
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    overview.to_csv(OUTPUT_DIR / "cycle_summary_85_validation_overview.csv", index=False, encoding="utf-8-sig")
    route_sample_summary.to_csv(OUTPUT_DIR / "cycle_summary_85_route_sample_counts.csv", index=False, encoding="utf-8-sig")
    missing_sequence_report.to_csv(OUTPUT_DIR / "cycle_summary_85_missing_cycle_sequence_report.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(OUTPUT_DIR / "cycle_summary_85_duplicate_sample_cycle_rows.csv", index=False, encoding="utf-8-sig")

    write_text_report(
        OUTPUT_DIR / "cycle_summary_85_validation_report.txt",
        overview=overview,
        route_sample_summary=route_sample_summary,
        missing_sequence_report=missing_sequence_report,
        df=combined,
    )

    print("\nValidation overview:")
    print(overview.to_string(index=False))
    print(f"\nSaved consolidated file:\n{OUTPUT_FILE}")
    print(f"Saved validation reports in:\n{OUTPUT_DIR}")

    if not validation_passed:
        message = (
            "\nSTRICT VALIDATION FAILED. The consolidated file and reports were written, "
            "but one or more required checks failed. Inspect the validation reports before DB loading."
        )
        print(message)
        if STRICT_VALIDATION:
            return 1

    print("\nSUCCESS: 85-sample cycle summary preparation completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

