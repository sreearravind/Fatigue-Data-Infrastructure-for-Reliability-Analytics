"""
task3.3_cycle_aggregated_85.py
================================

Purpose
-------
Build the 85-sample ML-ready dataset by aggregating cycle-level fatigue
records from PostgreSQL into one row per fatigue sample.

This script is the 85-sample extension of the earlier task3.3_cycle_aggregated.py.
The old script aggregated cycle-level values by route_id. This version aggregates
by specimen_id/sample_id first, preserving five samples per route for statistics,
Weibull/bootstrap analysis, and ML validation.

Input source
------------
PostgreSQL database: fatigue_dbms_85
Required tables:
    processing_route
    specimen
    specimen_processed
    cycle_summary

Main output
-----------
    02_cleaned/sample_level_features_85.csv

Additional outputs
------------------
    02_cleaned/task3_3_cyclic_aggregated_85.csv
    02_cleaned/route_level_feature_summary_85.csv
    02_cleaned/sample_level_missing_feature_report_85.csv
    02_cleaned/task3_3_cycle_aggregated_85_validation_overview.csv
    02_cleaned/task3_3_cycle_aggregated_85_validation_report.txt

Author workflow note
--------------------
Do not train ML directly on cycle_summary rows. Use the output of this script:
85 rows x sample-level PSPP/cyclic features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "scripts" / "00_database_setup"))

try:
    from db_config_85 import (
        get_engine,
        CLEANED_DIR,
        DB_OUTPUT_DIR,
        EXPECTED_ROUTES,
        EXPECTED_SAMPLES,
        EXPECTED_SAMPLES_PER_ROUTE,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import db_config_85.py from scripts/00_database_setup. "
        "Run this repository from its root so the relative script layout is preserved."
    ) from exc


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
SAMPLE_FEATURE_PATH = CLEANED_DIR / "sample_level_features_85.csv"
LEGACY_STYLE_AGG_PATH = CLEANED_DIR / "task3_3_cyclic_aggregated_85.csv"
ROUTE_SUMMARY_PATH = CLEANED_DIR / "route_level_feature_summary_85.csv"
MISSING_REPORT_PATH = CLEANED_DIR / "sample_level_missing_feature_report_85.csv"
VALIDATION_OVERVIEW_PATH = CLEANED_DIR / "task3_3_cycle_aggregated_85_validation_overview.csv"
VALIDATION_TEXT_PATH = CLEANED_DIR / "task3_3_cycle_aggregated_85_validation_report.txt"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def ensure_output_dirs() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    DB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def to_numeric_series(series: pd.Series, col_name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    n_bad = int(converted.isna().sum())
    if n_bad:
        print(f"⚠️ Column '{col_name}' has {n_bad} missing/non-numeric values after conversion.")
    return converted


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def compute_slope(group: pd.DataFrame, value_col: str, x_col: str = "cycle_fraction") -> float:
    """Return normalized linear slope of value_col against cycle_fraction."""
    subset = group[[x_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 2:
        return np.nan
    if subset[x_col].nunique() < 2:
        return np.nan
    try:
        return float(np.polyfit(subset[x_col].to_numpy(), subset[value_col].to_numpy(), 1)[0])
    except Exception:
        return np.nan


def coefficient_of_variation(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan
    mean_val = series.mean()
    if np.isclose(mean_val, 0.0):
        return np.nan
    return float(series.std(ddof=1) / abs(mean_val))


# ---------------------------------------------------------------------------
# Database extraction
# ---------------------------------------------------------------------------
def read_base_specimen_table() -> pd.DataFrame:
    """Read master PSPP specimen information from DB."""
    engine = get_engine()
    query = """
        SELECT
            s.specimen_id,
            s.route_id,
            pr.route_family,
            pr.process_subtype,
            pr.soak_hours,
            pr.ecap_angle_deg,
            s.sample_no,
            s.is_synthetic,
            s.ys_mpa,
            s.uts_mpa,
            s.elongation_percent,
            s.hardness_hv,
            s.grain_size_um,
            s.cycles_to_failure,
            s.tsa,
            s.frequency_hz,
            s.temperature_c,
            sp.log_nf AS log_nf_db,
            sp.d_inv_sqrt AS d_inv_sqrt_db,
            sp.strength_ratio AS strength_ratio_db,
            sp.fatigue_efficiency AS fatigue_efficiency_db
        FROM specimen s
        LEFT JOIN processing_route pr
            ON s.route_id = pr.route_id
        LEFT JOIN specimen_processed sp
            ON s.specimen_id = sp.specimen_id
        ORDER BY s.route_id, s.specimen_id;
    """
    with engine.begin() as conn:
        df = pd.read_sql(query, conn)
    return df


def read_cycle_summary_table() -> pd.DataFrame:
    """Read cycle-level fatigue descriptors from DB."""
    engine = get_engine()
    query = """
        SELECT
            specimen_id,
            route_id,
            cycle_no,
            cycle_fraction,
            psa,
            mean_stress_mpa,
            max_stress_mpa,
            min_stress_mpa,
            unloading_modulus_mpa,
            stress_amplitude_mpa
        FROM cycle_summary
        ORDER BY specimen_id, cycle_no;
    """
    with engine.begin() as conn:
        df = pd.read_sql(query, conn)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def prepare_numeric_inputs(base: pd.DataFrame, cycle: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert core columns to numeric and compute derived PSPP columns."""
    base = base.copy()
    cycle = cycle.copy()

    base_numeric_cols = [
        "soak_hours",
        "ecap_angle_deg",
        "sample_no",
        "ys_mpa",
        "uts_mpa",
        "elongation_percent",
        "hardness_hv",
        "grain_size_um",
        "cycles_to_failure",
        "tsa",
        "frequency_hz",
        "temperature_c",
        "log_nf_db",
        "d_inv_sqrt_db",
        "strength_ratio_db",
        "fatigue_efficiency_db",
    ]
    for col in base_numeric_cols:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    cycle_numeric_cols = [
        "cycle_no",
        "cycle_fraction",
        "psa",
        "mean_stress_mpa",
        "max_stress_mpa",
        "min_stress_mpa",
        "unloading_modulus_mpa",
        "stress_amplitude_mpa",
    ]
    for col in cycle_numeric_cols:
        if col in cycle.columns:
            cycle[col] = pd.to_numeric(cycle[col], errors="coerce")

    # Authoritative derived PSPP features are recalculated from specimen table
    # to ensure consistency after the 85-sample update.
    base["log10_nf"] = np.log10(base["cycles_to_failure"])
    base["d_inv_sqrt"] = 1.0 / np.sqrt(base["grain_size_um"])
    base["strength_ratio"] = safe_divide(base["ys_mpa"], base["uts_mpa"])
    base["fatigue_efficiency_logNf_per_MPa"] = safe_divide(base["log10_nf"], base["ys_mpa"])

    # Ensure cycle_fraction exists and is linked to master cycles_to_failure.
    cycle = cycle.merge(
        base[["specimen_id", "cycles_to_failure"]],
        on="specimen_id",
        how="left",
        validate="many_to_one",
    )
    if "cycle_fraction" not in cycle.columns:
        cycle["cycle_fraction"] = np.nan
    missing_fraction = cycle["cycle_fraction"].isna()
    cycle.loc[missing_fraction, "cycle_fraction"] = safe_divide(
        cycle.loc[missing_fraction, "cycle_no"],
        cycle.loc[missing_fraction, "cycles_to_failure"],
    )

    cycle["stress_range_mpa"] = cycle["max_stress_mpa"] - cycle["min_stress_mpa"]

    # Energy proxy only: cycle_summary does not contain full stress-strain loop coordinates.
    # Unit note: MPa × strain is numerically equivalent to MJ/m^3.
    cycle["energy_proxy_mj_m3"] = 4.0 * cycle["stress_amplitude_mpa"] * cycle["psa"]

    return base, cycle


def aggregate_region_features(cycle: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cycle-level features into one row per specimen."""
    cycle = cycle.copy()

    # Region definitions based on fraction of life.
    cycle["is_initial"] = cycle["cycle_fraction"].between(0.0, 0.10, inclusive="both")
    cycle["is_stable"] = cycle["cycle_fraction"].between(0.30, 0.70, inclusive="both")
    cycle["is_final"] = cycle["cycle_fraction"].between(0.90, 1.00, inclusive="both")

    records: List[Dict[str, float]] = []

    metrics = {
        "psa": "psa",
        "mean_stress_mpa": "mean_stress",
        "stress_amplitude_mpa": "stress_amplitude",
        "unloading_modulus_mpa": "unloading_modulus",
        "energy_proxy_mj_m3": "energy_proxy",
    }

    for specimen_id, g in cycle.groupby("specimen_id", sort=True):
        g = g.sort_values("cycle_no").copy()
        route_id = g["route_id"].dropna().iloc[0] if g["route_id"].notna().any() else np.nan
        initial = g[g["is_initial"]]
        stable = g[g["is_stable"]]
        final = g[g["is_final"]]

        row: Dict[str, float] = {
            "specimen_id": specimen_id,
            "route_id_from_cycle": route_id,
            "n_cycle_rows": int(len(g)),
            "cycle_no_min": int(g["cycle_no"].min()) if g["cycle_no"].notna().any() else np.nan,
            "cycle_no_max": int(g["cycle_no"].max()) if g["cycle_no"].notna().any() else np.nan,
            "cycle_fraction_min": float(g["cycle_fraction"].min()),
            "cycle_fraction_max": float(g["cycle_fraction"].max()),
            "initial_region_rows": int(len(initial)),
            "stable_region_rows": int(len(stable)),
            "final_region_rows": int(len(final)),
            "max_stress_peak": float(g["max_stress_mpa"].max()),
            "min_stress_minimum": float(g["min_stress_mpa"].min()),
            "stress_range_mean": float(g["stress_range_mpa"].mean()),
            "stress_range_stable_mean": float(stable["stress_range_mpa"].mean()) if len(stable) else np.nan,
        }

        for col, prefix in metrics.items():
            row[f"{prefix}_mean"] = float(g[col].mean())
            row[f"{prefix}_std"] = float(g[col].std(ddof=1))
            row[f"{prefix}_min"] = float(g[col].min())
            row[f"{prefix}_max"] = float(g[col].max())
            row[f"{prefix}_initial_mean"] = float(initial[col].mean()) if len(initial) else np.nan
            row[f"{prefix}_stable_mean"] = float(stable[col].mean()) if len(stable) else np.nan
            row[f"{prefix}_stable_std"] = float(stable[col].std(ddof=1)) if len(stable) > 1 else np.nan
            row[f"{prefix}_stable_min"] = float(stable[col].min()) if len(stable) else np.nan
            row[f"{prefix}_stable_max"] = float(stable[col].max()) if len(stable) else np.nan
            row[f"{prefix}_stable_cov"] = coefficient_of_variation(stable[col]) if len(stable) else np.nan
            row[f"{prefix}_final_mean"] = float(final[col].mean()) if len(final) else np.nan
            row[f"{prefix}_slope"] = compute_slope(g, col, x_col="cycle_fraction")

            # Stability ratio: final / initial. Useful for cyclic softening/hardening trends.
            initial_mean = row[f"{prefix}_initial_mean"]
            final_mean = row[f"{prefix}_final_mean"]
            if pd.notna(initial_mean) and not np.isclose(initial_mean, 0.0):
                row[f"{prefix}_final_to_initial_ratio"] = float(final_mean / initial_mean)
            else:
                row[f"{prefix}_final_to_initial_ratio"] = np.nan

        # Mean stress ratio in the stabilized regime.
        if pd.notna(row.get("stress_amplitude_stable_mean")) and not np.isclose(row["stress_amplitude_stable_mean"], 0.0):
            row["mean_stress_ratio_stable"] = float(
                row["mean_stress_stable_mean"] / row["stress_amplitude_stable_mean"]
            )
        else:
            row["mean_stress_ratio_stable"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


def build_sample_level_features(base: pd.DataFrame, cycle_agg: pd.DataFrame) -> pd.DataFrame:
    """Merge specimen PSPP features with aggregated cycle descriptors."""
    df = base.merge(cycle_agg, on="specimen_id", how="left", validate="one_to_one")

    # Check route consistency between specimen and cycle_summary.
    df["route_id_cycle_match"] = df["route_id"].astype(str) == df["route_id_from_cycle"].astype(str)

    # Coverage ratio should usually be close to 1.0, but can differ if post-failure
    # rows were removed or if cycle summaries were cleaned.
    df["cycle_coverage_ratio"] = safe_divide(df["cycle_no_max"], df["cycles_to_failure"])

    preferred_order = [
        "specimen_id",
        "route_id",
        "route_family",
        "process_subtype",
        "sample_no",
        "is_synthetic",
        "soak_hours",
        "ecap_angle_deg",
        "ys_mpa",
        "uts_mpa",
        "elongation_percent",
        "hardness_hv",
        "grain_size_um",
        "cycles_to_failure",
        "log10_nf",
        "d_inv_sqrt",
        "strength_ratio",
        "fatigue_efficiency_logNf_per_MPa",
        "tsa",
        "frequency_hz",
        "temperature_c",
        "n_cycle_rows",
        "cycle_no_min",
        "cycle_no_max",
        "cycle_coverage_ratio",
        "cycle_fraction_min",
        "cycle_fraction_max",
        "initial_region_rows",
        "stable_region_rows",
        "final_region_rows",
        "psa_initial_mean",
        "psa_stable_mean",
        "psa_stable_std",
        "psa_stable_cov",
        "psa_final_mean",
        "psa_slope",
        "psa_final_to_initial_ratio",
        "mean_stress_initial_mean",
        "mean_stress_stable_mean",
        "mean_stress_stable_std",
        "mean_stress_stable_cov",
        "mean_stress_final_mean",
        "mean_stress_slope",
        "mean_stress_final_to_initial_ratio",
        "mean_stress_ratio_stable",
        "stress_amplitude_initial_mean",
        "stress_amplitude_stable_mean",
        "stress_amplitude_stable_std",
        "stress_amplitude_stable_cov",
        "stress_amplitude_final_mean",
        "stress_amplitude_slope",
        "stress_amplitude_final_to_initial_ratio",
        "unloading_modulus_initial_mean",
        "unloading_modulus_stable_mean",
        "unloading_modulus_stable_std",
        "unloading_modulus_stable_cov",
        "unloading_modulus_final_mean",
        "unloading_modulus_slope",
        "unloading_modulus_final_to_initial_ratio",
        "max_stress_peak",
        "min_stress_minimum",
        "stress_range_mean",
        "stress_range_stable_mean",
        "energy_proxy_initial_mean",
        "energy_proxy_stable_mean",
        "energy_proxy_stable_std",
        "energy_proxy_stable_cov",
        "energy_proxy_final_mean",
        "energy_proxy_slope",
        "energy_proxy_final_to_initial_ratio",
        "route_id_cycle_match",
    ]

    existing_order = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_order]
    df = df[existing_order + remaining]

    return df.sort_values(["route_id", "specimen_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------
def build_missing_feature_report(features: pd.DataFrame) -> pd.DataFrame:
    report = []
    for col in features.columns:
        missing = int(features[col].isna().sum())
        if missing > 0:
            report.append(
                {
                    "column": col,
                    "missing_count": missing,
                    "missing_percent": round(100 * missing / len(features), 3) if len(features) else np.nan,
                }
            )
    return pd.DataFrame(report).sort_values("missing_count", ascending=False) if report else pd.DataFrame(
        columns=["column", "missing_count", "missing_percent"]
    )


def validate_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    route_counts = features.groupby("route_id")["specimen_id"].nunique().sort_index()

    checks = []

    def add_check(check: str, expected, observed, passed: bool) -> None:
        checks.append(
            {
                "check": check,
                "expected": expected,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
        )

    add_check("Total sample-level rows", EXPECTED_SAMPLES, len(features), len(features) == EXPECTED_SAMPLES)
    add_check(
        "Unique specimen_id count",
        EXPECTED_SAMPLES,
        features["specimen_id"].nunique(),
        features["specimen_id"].nunique() == EXPECTED_SAMPLES,
    )
    add_check(
        "Unique route_id count",
        EXPECTED_ROUTES,
        features["route_id"].nunique(),
        features["route_id"].nunique() == EXPECTED_ROUTES,
    )
    add_check(
        "Routes with exactly five samples",
        EXPECTED_ROUTES,
        int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()),
        bool((route_counts == EXPECTED_SAMPLES_PER_ROUTE).all()),
    )
    add_check(
        "Duplicate specimen_id rows",
        0,
        int(features.duplicated("specimen_id").sum()),
        int(features.duplicated("specimen_id").sum()) == 0,
    )
    add_check(
        "Missing cycles_to_failure",
        0,
        int(features["cycles_to_failure"].isna().sum()),
        int(features["cycles_to_failure"].isna().sum()) == 0,
    )
    add_check(
        "Missing log10_nf",
        0,
        int(features["log10_nf"].isna().sum()),
        int(features["log10_nf"].isna().sum()) == 0,
    )
    add_check(
        "Missing stable PSA mean",
        0,
        int(features["psa_stable_mean"].isna().sum()),
        int(features["psa_stable_mean"].isna().sum()) == 0,
    )
    add_check(
        "Missing stable mean stress mean",
        0,
        int(features["mean_stress_stable_mean"].isna().sum()),
        int(features["mean_stress_stable_mean"].isna().sum()) == 0,
    )
    add_check(
        "Route ID match between specimen and cycle_summary",
        "all TRUE",
        int(features["route_id_cycle_match"].sum()),
        bool(features["route_id_cycle_match"].all()),
    )

    numeric = features.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0
    add_check("Infinite numeric values", 0, inf_count, inf_count == 0)

    overview = pd.DataFrame(checks)
    final_status = "PASS" if (overview["status"] == "PASS").all() else "FAIL"
    return overview, final_status


def build_route_level_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = (
        features.groupby(["route_id", "route_family", "process_subtype"], dropna=False)
        .agg(
            n_samples=("specimen_id", "nunique"),
            nf_mean=("cycles_to_failure", "mean"),
            nf_std=("cycles_to_failure", "std"),
            log10_nf_mean=("log10_nf", "mean"),
            log10_nf_std=("log10_nf", "std"),
            ys_mpa_mean=("ys_mpa", "mean"),
            hardness_hv_mean=("hardness_hv", "mean"),
            grain_size_um_mean=("grain_size_um", "mean"),
            psa_stable_mean=("psa_stable_mean", "mean"),
            mean_stress_stable_mean=("mean_stress_stable_mean", "mean"),
            stress_amplitude_stable_mean=("stress_amplitude_stable_mean", "mean"),
            unloading_modulus_stable_mean=("unloading_modulus_stable_mean", "mean"),
            energy_proxy_stable_mean=("energy_proxy_stable_mean", "mean"),
        )
        .reset_index()
        .sort_values("route_id")
    )
    summary["nf_cov"] = safe_divide(summary["nf_std"], summary["nf_mean"])
    return summary


def write_reports(
    features: pd.DataFrame,
    route_summary: pd.DataFrame,
    missing_report: pd.DataFrame,
    validation_overview: pd.DataFrame,
    final_status: str,
) -> None:
    features.to_csv(SAMPLE_FEATURE_PATH, index=False)
    features.to_csv(LEGACY_STYLE_AGG_PATH, index=False)
    route_summary.to_csv(ROUTE_SUMMARY_PATH, index=False)
    missing_report.to_csv(MISSING_REPORT_PATH, index=False)
    validation_overview.to_csv(VALIDATION_OVERVIEW_PATH, index=False)

    route_counts = features.groupby("route_id")["specimen_id"].nunique().sort_index()

    with open(VALIDATION_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write("task3.3_cycle_aggregated_85 validation report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Main output: {SAMPLE_FEATURE_PATH}\n")
        f.write(f"Legacy-style output: {LEGACY_STYLE_AGG_PATH}\n")
        f.write(f"Route summary: {ROUTE_SUMMARY_PATH}\n")
        f.write(f"Missing feature report: {MISSING_REPORT_PATH}\n\n")
        f.write(f"Sample-level rows: {len(features)}\n")
        f.write(f"Unique routes: {features['route_id'].nunique()}\n")
        f.write(f"Unique specimens: {features['specimen_id'].nunique()}\n")
        f.write(f"Total engineered columns: {features.shape[1]}\n")
        f.write(f"Final validation status: {final_status}\n\n")
        f.write("Validation overview:\n")
        f.write(validation_overview.to_string(index=False))
        f.write("\n\nSamples per route:\n")
        f.write(route_counts.to_string())
        f.write("\n\nKey ML note:\n")
        f.write(
            "Use sample_level_features_85.csv for ML. Do not use the 159,967 "
            "cycle_summary rows as independent ML samples, because cycle rows from "
            "the same specimen are not independent.\n"
        )
        f.write("\nEnergy note:\n")
        f.write(
            "energy_proxy_* columns are approximate cyclic energy proxies calculated "
            "as 4 * stress_amplitude_mpa * psa. They are not full closed-loop "
            "hysteresis energies from raw stress-strain coordinates.\n"
        )


def main() -> None:
    print("=== START task3.3_cycle_aggregated_85 ===")
    ensure_output_dirs()

    base = read_base_specimen_table()
    cycle = read_cycle_summary_table()

    print(f"Base specimen rows from DB: {len(base)}")
    print(f"Cycle summary rows from DB: {len(cycle)}")
    print(f"Base columns: {base.columns.tolist()}")
    print(f"Cycle columns: {cycle.columns.tolist()}")

    require_columns(
        base,
        [
            "specimen_id",
            "route_id",
            "route_family",
            "process_subtype",
            "ys_mpa",
            "uts_mpa",
            "hardness_hv",
            "grain_size_um",
            "cycles_to_failure",
        ],
        "base specimen table",
    )
    require_columns(
        cycle,
        [
            "specimen_id",
            "cycle_no",
            "psa",
            "mean_stress_mpa",
            "max_stress_mpa",
            "min_stress_mpa",
            "unloading_modulus_mpa",
            "stress_amplitude_mpa",
        ],
        "cycle_summary table",
    )

    base, cycle = prepare_numeric_inputs(base, cycle)
    cycle_agg = aggregate_region_features(cycle)
    features = build_sample_level_features(base, cycle_agg)

    missing_report = build_missing_feature_report(features)
    validation_overview, final_status = validate_features(features)
    route_summary = build_route_level_summary(features)

    write_reports(features, route_summary, missing_report, validation_overview, final_status)

    print("\nSample-level features preview:")
    print(features.head())

    print("\nValidation overview:")
    print(validation_overview)

    print(f"\nSaved main ML-ready file to: {SAMPLE_FEATURE_PATH}")
    print(f"Saved route summary to: {ROUTE_SUMMARY_PATH}")
    print(f"Saved validation report to: {VALIDATION_TEXT_PATH}")

    if final_status != "PASS":
        raise RuntimeError(
            "task3.3_cycle_aggregated_85 completed but validation status is FAIL. "
            f"Review: {VALIDATION_TEXT_PATH}"
        )

    print("\n✅ Done task3.3_cycle_aggregated_85. Status: PASS")


if __name__ == "__main__":
    main()
