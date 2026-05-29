"""
Phase4.8_FAIR_bootstrap_CI_by_route_85.py

Purpose
-------
Consolidate route-wise fatigue reliability metadata for the 85-sample workflow.

This script reads the ML-ready sample-level feature file:
    Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

It then creates a FAIR-ready route-level reliability metadata table containing:
    - route metadata
    - route-wise fatigue-life descriptive statistics
    - route-wise bootstrap 95% confidence intervals for mean Nf and mean log10(Nf)
    - route-wise Weibull parameters
    - route-wise B-life estimates: B1, B5, B10, B50, B90
    - reliability interpretation notes

Important interpretation
------------------------
Bootstrap resampling is performed at the sample level, not the cycle-row level.
Route-wise Weibull results are based on n = 5 samples per route and should be
reported as exploratory reliability metadata rather than full industrial
route-specific reliability qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42
CI_PERCENTILES = (2.5, 50.0, 97.5)

SCRIPT_NAME = "Phase4.8_FAIR_bootstrap_CI_by_route_85.py"


def resolve_project_root() -> Path:
    """
    Resolve the project root robustly.

    Expected placement:
        <project_root>/db_scripts_85/Phase4.8_FAIR_bootstrap_CI_by_route_85.py

    If the script is not inside db_scripts_85, the current working directory is
    checked as a fallback.
    """
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "db_scripts_85":
        return here.parent.parent
    if (here.parent / "Fatigue_85_augmented_dataset").exists():
        return here.parent
    if (Path.cwd() / "Fatigue_85_augmented_dataset").exists():
        return Path.cwd()
    return here.parent.parent


PROJECT_ROOT = resolve_project_root()
DATASET_DIR = PROJECT_ROOT / "Fatigue_85_augmented_dataset"
CLEANED_DIR = DATASET_DIR / "02_cleaned"
STATS_DIR = DATASET_DIR / "04_statistics_outputs"
FIG_DIR = DATASET_DIR / "06_figures"

INPUT_SAMPLE_FEATURES = CLEANED_DIR / "sample_level_features_85.csv"

OUT_FAIR_ROUTE_METADATA = STATS_DIR / "Phase4_8_FAIR_bootstrap_CI_by_route_85.csv"
OUT_FAIR_ROUTE_FAMILY_METADATA = STATS_DIR / "Phase4_8_FAIR_bootstrap_CI_by_route_family_85.csv"
OUT_COLUMN_DICTIONARY = STATS_DIR / "Phase4_8_FAIR_route_metadata_dictionary_85.csv"
OUT_VALIDATION = STATS_DIR / "Phase4_8_FAIR_bootstrap_validation_overview_85.csv"
OUT_JSON_METADATA = STATS_DIR / "Phase4_8_FAIR_analysis_metadata_85.json"
OUT_REPORT = STATS_DIR / "Phase4_8_FAIR_bootstrap_CI_by_route_report_85.txt"

FIG_ROUTE_MEAN_CI = FIG_DIR / "Phase4_8_routewise_mean_nf_bootstrap_CI_85.png"
FIG_ROUTE_B_LIFE = FIG_DIR / "Phase4_8_routewise_weibull_B10_B50_B90_85.png"
FIG_ROUTE_FAMILY_MEAN_CI = FIG_DIR / "Phase4_8_route_family_mean_nf_bootstrap_CI_85.png"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and standardise column names."""
    out = df.copy()
    out.columns = (
        pd.Index(out.columns)
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
        "sample_id": "specimen_id",
        "sampleid": "specimen_id",
        "specimenid": "specimen_id",
        "nf": "cycles_to_failure",
        "n_f": "cycles_to_failure",
        "log10nf": "log10_nf",
        "log_nf": "log10_nf",
        "log_nf_db": "log10_nf_db",
        "ys": "ys_mpa",
        "uts": "uts_mpa",
        "hardness": "hardness_hv",
        "grain_size": "grain_size_um",
        "fatigue_efficiency_db": "fatigue_efficiency_logNf_per_MPa_db",
        "fatigue_efficiency_lognf_per_mpa": "fatigue_efficiency_logNf_per_MPa",
    }
    out = out.rename(columns=rename_map)
    return out


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {label}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def safe_cv(values: pd.Series) -> float:
    mean = values.mean()
    sd = values.std(ddof=1)
    if pd.isna(mean) or abs(mean) < 1e-12:
        return np.nan
    return sd / mean


def bootstrap_ci_mean(
    data: np.ndarray | pd.Series,
    n_bootstrap: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    percentiles: tuple[float, float, float] = CI_PERCENTILES,
) -> dict[str, float]:
    """Bootstrap confidence interval for the mean."""
    arr = pd.to_numeric(pd.Series(data), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {
            "observed": np.nan,
            "ci_2p5": np.nan,
            "ci_50": np.nan,
            "ci_97p5": np.nan,
            "ci_width": np.nan,
            "n_samples": 0,
            "bootstrap_B": n_bootstrap,
        }

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)

    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci = np.percentile(boot_means, percentiles)
    return {
        "observed": float(arr.mean()),
        "ci_2p5": float(ci[0]),
        "ci_50": float(ci[1]),
        "ci_97p5": float(ci[2]),
        "ci_width": float(ci[2] - ci[0]),
        "n_samples": int(n),
        "bootstrap_B": int(n_bootstrap),
    }


def fit_weibull_2p(values: np.ndarray | pd.Series) -> dict[str, float | str]:
    """
    Fit two-parameter Weibull distribution with loc fixed at 0.

    Returns NaN parameters if fitting is unstable. This is intentionally conservative.
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    if len(arr) < 2:
        return {
            "shape_beta": np.nan,
            "scale_eta_cycles": np.nan,
            "weibull_fit_status": "FAIL: fewer than 2 positive observations",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape, loc, scale = weibull_min.fit(arr, floc=0)
        if not np.isfinite(shape) or not np.isfinite(scale) or shape <= 0 or scale <= 0:
            raise ValueError("Non-finite or non-positive Weibull parameters")
        return {
            "shape_beta": float(shape),
            "scale_eta_cycles": float(scale),
            "weibull_fit_status": "PASS",
        }
    except Exception as exc:
        return {
            "shape_beta": np.nan,
            "scale_eta_cycles": np.nan,
            "weibull_fit_status": f"FAIL: {exc}",
        }


def weibull_quantile(scale_eta: float, shape_beta: float, failure_probability: float) -> float:
    if pd.isna(scale_eta) or pd.isna(shape_beta):
        return np.nan
    if scale_eta <= 0 or shape_beta <= 0:
        return np.nan
    if failure_probability <= 0 or failure_probability >= 1:
        return np.nan
    return float(scale_eta * (-np.log(1.0 - failure_probability)) ** (1.0 / shape_beta))


def build_route_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build route-wise FAIR reliability metadata table."""
    rows: list[dict[str, object]] = []

    for route_id, sub in df.groupby("route_id", sort=True):
        nf = sub["cycles_to_failure"].astype(float)
        log_nf = sub["log10_nf"].astype(float)

        nf_ci = bootstrap_ci_mean(nf, seed=BOOTSTRAP_SEED)
        log_ci = bootstrap_ci_mean(log_nf, seed=BOOTSTRAP_SEED + 17)

        weibull = fit_weibull_2p(nf)
        shape = weibull["shape_beta"]
        scale = weibull["scale_eta_cycles"]

        route_family = sub["route_family"].dropna().iloc[0] if "route_family" in sub.columns and sub["route_family"].notna().any() else ""

        row: dict[str, object] = {
            "route_id": route_id,
            "route_family": route_family,
            "n_samples": int(len(sub)),
            "specimen_id_list": ";".join(sub["specimen_id"].astype(str).sort_values().tolist()),
            "nf_min": float(nf.min()),
            "nf_max": float(nf.max()),
            "nf_mean": float(nf.mean()),
            "nf_median": float(nf.median()),
            "nf_sd": float(nf.std(ddof=1)),
            "nf_cov": float(safe_cv(nf)),
            "log10_nf_mean": float(log_nf.mean()),
            "log10_nf_median": float(log_nf.median()),
            "log10_nf_sd": float(log_nf.std(ddof=1)),
            "bootstrap_mean_nf_ci_2p5": nf_ci["ci_2p5"],
            "bootstrap_mean_nf_ci_50": nf_ci["ci_50"],
            "bootstrap_mean_nf_ci_97p5": nf_ci["ci_97p5"],
            "bootstrap_mean_nf_ci_width": nf_ci["ci_width"],
            "bootstrap_mean_log10nf_ci_2p5": log_ci["ci_2p5"],
            "bootstrap_mean_log10nf_ci_50": log_ci["ci_50"],
            "bootstrap_mean_log10nf_ci_97p5": log_ci["ci_97p5"],
            "bootstrap_mean_log10nf_ci_width": log_ci["ci_width"],
            "bootstrap_B": BOOTSTRAP_B,
            "shape_beta": shape,
            "scale_eta_cycles": scale,
            "weibull_fit_status": weibull["weibull_fit_status"],
            "B1_life_cycles": weibull_quantile(scale, shape, 0.01),
            "B5_life_cycles": weibull_quantile(scale, shape, 0.05),
            "B10_life_cycles": weibull_quantile(scale, shape, 0.10),
            "B50_median_life_cycles": weibull_quantile(scale, shape, 0.50),
            "B90_life_cycles": weibull_quantile(scale, shape, 0.90),
            "analysis_level": "route",
            "data_granularity": "sample-level aggregation from cycle-level descriptors",
            "reliability_scope": "exploratory route-wise reliability metadata; n=5 per route",
            "fair_findable_id": f"FDIP85_ROUTE_{route_id}",
            "fair_accessible_source": "sample_level_features_85.csv",
            "fair_interoperable_schema": "route_id + specimen_id + PSPP descriptors + reliability metadata",
            "fair_reusable_note": "Bootstrap and Weibull metadata generated reproducibly with fixed random seed",
            "analysis_script": SCRIPT_NAME,
            "analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("nf_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_by_mean_nf", np.arange(1, len(out) + 1))
    return out


def build_route_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build route-family FAIR reliability metadata table."""
    rows: list[dict[str, object]] = []

    for family, sub in df.groupby("route_family", sort=True):
        nf = sub["cycles_to_failure"].astype(float)
        log_nf = sub["log10_nf"].astype(float)

        nf_ci = bootstrap_ci_mean(nf, seed=BOOTSTRAP_SEED)
        log_ci = bootstrap_ci_mean(log_nf, seed=BOOTSTRAP_SEED + 17)
        weibull = fit_weibull_2p(nf)
        shape = weibull["shape_beta"]
        scale = weibull["scale_eta_cycles"]

        row: dict[str, object] = {
            "route_family": family,
            "n_samples": int(len(sub)),
            "n_routes": int(sub["route_id"].nunique()),
            "route_id_list": ";".join(sorted(sub["route_id"].astype(str).unique().tolist())),
            "nf_min": float(nf.min()),
            "nf_max": float(nf.max()),
            "nf_mean": float(nf.mean()),
            "nf_median": float(nf.median()),
            "nf_sd": float(nf.std(ddof=1)),
            "nf_cov": float(safe_cv(nf)),
            "log10_nf_mean": float(log_nf.mean()),
            "log10_nf_sd": float(log_nf.std(ddof=1)),
            "bootstrap_mean_nf_ci_2p5": nf_ci["ci_2p5"],
            "bootstrap_mean_nf_ci_50": nf_ci["ci_50"],
            "bootstrap_mean_nf_ci_97p5": nf_ci["ci_97p5"],
            "bootstrap_mean_nf_ci_width": nf_ci["ci_width"],
            "bootstrap_mean_log10nf_ci_2p5": log_ci["ci_2p5"],
            "bootstrap_mean_log10nf_ci_50": log_ci["ci_50"],
            "bootstrap_mean_log10nf_ci_97p5": log_ci["ci_97p5"],
            "bootstrap_mean_log10nf_ci_width": log_ci["ci_width"],
            "bootstrap_B": BOOTSTRAP_B,
            "shape_beta": shape,
            "scale_eta_cycles": scale,
            "weibull_fit_status": weibull["weibull_fit_status"],
            "B10_life_cycles": weibull_quantile(scale, shape, 0.10),
            "B50_median_life_cycles": weibull_quantile(scale, shape, 0.50),
            "B90_life_cycles": weibull_quantile(scale, shape, 0.90),
            "analysis_level": "route_family",
            "reliability_scope": "exploratory route-family reliability metadata",
            "analysis_script": SCRIPT_NAME,
            "analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("nf_mean", ascending=False).reset_index(drop=True)
    out.insert(0, "rank_by_mean_nf", np.arange(1, len(out) + 1))
    return out


def make_column_dictionary() -> pd.DataFrame:
    """Create a FAIR-oriented data dictionary for the main route metadata table."""
    definitions = [
        ("rank_by_mean_nf", "Ranking of routes by mean fatigue life, descending", "index", "Reusable"),
        ("route_id", "Processing route identifier", "text", "Findable"),
        ("route_family", "Processing family such as AR, HT, DCT, or ECAP", "text", "Interoperable"),
        ("n_samples", "Number of sample-level records for the route", "count", "Reusable"),
        ("specimen_id_list", "Semicolon-separated specimen IDs contributing to the route summary", "text", "Findable"),
        ("nf_min", "Minimum fatigue life for the route", "cycles", "Reusable"),
        ("nf_max", "Maximum fatigue life for the route", "cycles", "Reusable"),
        ("nf_mean", "Mean fatigue life for the route", "cycles", "Reusable"),
        ("nf_median", "Median fatigue life for the route", "cycles", "Reusable"),
        ("nf_sd", "Sample standard deviation of fatigue life", "cycles", "Reusable"),
        ("nf_cov", "Coefficient of variation of fatigue life", "dimensionless", "Reusable"),
        ("log10_nf_mean", "Mean log10-transformed fatigue life", "dimensionless", "Reusable"),
        ("log10_nf_sd", "Sample standard deviation of log10-transformed fatigue life", "dimensionless", "Reusable"),
        ("bootstrap_mean_nf_ci_2p5", "Lower 2.5 percentile bootstrap CI for mean fatigue life", "cycles", "Reusable"),
        ("bootstrap_mean_nf_ci_50", "Median bootstrap estimate for mean fatigue life", "cycles", "Reusable"),
        ("bootstrap_mean_nf_ci_97p5", "Upper 97.5 percentile bootstrap CI for mean fatigue life", "cycles", "Reusable"),
        ("bootstrap_mean_log10nf_ci_2p5", "Lower 2.5 percentile bootstrap CI for mean log10 fatigue life", "dimensionless", "Reusable"),
        ("bootstrap_mean_log10nf_ci_50", "Median bootstrap estimate for mean log10 fatigue life", "dimensionless", "Reusable"),
        ("bootstrap_mean_log10nf_ci_97p5", "Upper 97.5 percentile bootstrap CI for mean log10 fatigue life", "dimensionless", "Reusable"),
        ("bootstrap_B", "Number of bootstrap resamples", "count", "Reusable"),
        ("shape_beta", "Two-parameter Weibull shape parameter", "dimensionless", "Reusable"),
        ("scale_eta_cycles", "Two-parameter Weibull scale/characteristic life", "cycles", "Reusable"),
        ("B1_life_cycles", "Weibull life at 1% failure probability", "cycles", "Reusable"),
        ("B5_life_cycles", "Weibull life at 5% failure probability", "cycles", "Reusable"),
        ("B10_life_cycles", "Weibull life at 10% failure probability", "cycles", "Reusable"),
        ("B50_median_life_cycles", "Weibull median life at 50% failure probability", "cycles", "Reusable"),
        ("B90_life_cycles", "Weibull life at 90% failure probability", "cycles", "Reusable"),
        ("reliability_scope", "Interpretation boundary for reliability metadata", "text", "Reusable"),
        ("fair_findable_id", "Route-level FAIR identifier generated for the metadata row", "text", "Findable"),
        ("fair_accessible_source", "Source file used to create the metadata table", "text", "Accessible"),
        ("fair_interoperable_schema", "Schema description for cross-study/database reuse", "text", "Interoperable"),
        ("fair_reusable_note", "Reuse/reproducibility note", "text", "Reusable"),
    ]
    return pd.DataFrame(definitions, columns=["column_name", "definition", "unit_or_type", "fair_principle"])


def build_validation(df: pd.DataFrame, route_df: pd.DataFrame) -> pd.DataFrame:
    """Create validation overview."""
    route_sample_counts = df.groupby("route_id")["specimen_id"].nunique()
    validations = [
        {
            "check": "Total sample rows",
            "expected": EXPECTED_SAMPLES,
            "observed": int(len(df)),
            "status": "PASS" if len(df) == EXPECTED_SAMPLES else "FAIL",
        },
        {
            "check": "Unique specimen_id count",
            "expected": EXPECTED_SAMPLES,
            "observed": int(df["specimen_id"].nunique()),
            "status": "PASS" if df["specimen_id"].nunique() == EXPECTED_SAMPLES else "FAIL",
        },
        {
            "check": "Unique route_id count",
            "expected": EXPECTED_ROUTES,
            "observed": int(df["route_id"].nunique()),
            "status": "PASS" if df["route_id"].nunique() == EXPECTED_ROUTES else "FAIL",
        },
        {
            "check": "Routes with exactly five samples",
            "expected": EXPECTED_ROUTES,
            "observed": int((route_sample_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()),
            "status": "PASS" if int((route_sample_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES else "FAIL",
        },
        {
            "check": "Missing cycles_to_failure",
            "expected": 0,
            "observed": int(df["cycles_to_failure"].isna().sum()),
            "status": "PASS" if df["cycles_to_failure"].isna().sum() == 0 else "FAIL",
        },
        {
            "check": "Missing log10_nf",
            "expected": 0,
            "observed": int(df["log10_nf"].isna().sum()),
            "status": "PASS" if df["log10_nf"].isna().sum() == 0 else "FAIL",
        },
        {
            "check": "Non-positive fatigue life",
            "expected": 0,
            "observed": int((df["cycles_to_failure"] <= 0).sum()),
            "status": "PASS" if int((df["cycles_to_failure"] <= 0).sum()) == 0 else "FAIL",
        },
        {
            "check": "Route metadata rows",
            "expected": EXPECTED_ROUTES,
            "observed": int(len(route_df)),
            "status": "PASS" if len(route_df) == EXPECTED_ROUTES else "FAIL",
        },
        {
            "check": "Missing bootstrap route CI values",
            "expected": 0,
            "observed": int(route_df[["bootstrap_mean_nf_ci_2p5", "bootstrap_mean_nf_ci_97p5"]].isna().sum().sum()),
            "status": "PASS" if int(route_df[["bootstrap_mean_nf_ci_2p5", "bootstrap_mean_nf_ci_97p5"]].isna().sum().sum()) == 0 else "FAIL",
        },
        {
            "check": "Missing Weibull route parameters",
            "expected": 0,
            "observed": int(route_df[["shape_beta", "scale_eta_cycles"]].isna().sum().sum()),
            "status": "PASS" if int(route_df[["shape_beta", "scale_eta_cycles"]].isna().sum().sum()) == 0 else "FAIL",
        },
    ]
    return pd.DataFrame(validations)


def plot_route_mean_ci(route_df: pd.DataFrame, out_path: Path) -> None:
    """Plot route-wise mean Nf with bootstrap CI."""
    plot_df = route_df.sort_values("nf_mean", ascending=True).copy()
    y = np.arange(len(plot_df))

    x = plot_df["nf_mean"].to_numpy(dtype=float)
    left = x - plot_df["bootstrap_mean_nf_ci_2p5"].to_numpy(dtype=float)
    right = plot_df["bootstrap_mean_nf_ci_97p5"].to_numpy(dtype=float) - x

    plt.figure(figsize=(10, 7))
    plt.errorbar(x, y, xerr=[left, right], fmt="o", capsize=3)
    plt.yticks(y, plot_df["route_id"])
    plt.xlabel("Mean fatigue life with 95% bootstrap CI (cycles)")
    plt.ylabel("Processing route")
    plt.title("FAIR route-wise fatigue-life metadata: mean Nf with bootstrap CI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_route_b_life(route_df: pd.DataFrame, out_path: Path) -> None:
    """Plot route-wise Weibull B10-B50-B90 life estimates."""
    plot_df = route_df.sort_values("nf_mean", ascending=True).copy()
    y = np.arange(len(plot_df))

    plt.figure(figsize=(10, 7))
    plt.scatter(plot_df["B50_median_life_cycles"], y, label="B50 median")
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        plt.vlines(row.B10_life_cycles, idx - 0.28, idx + 0.28, label="B10" if idx == 0 else None)
        plt.vlines(row.B90_life_cycles, idx - 0.28, idx + 0.28, label="B90" if idx == 0 else None)
    plt.yticks(y, plot_df["route_id"])
    plt.xlabel("Weibull life quantile (cycles)")
    plt.ylabel("Processing route")
    plt.title("FAIR route-wise Weibull life metadata: B10-B50-B90")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_route_family_mean_ci(family_df: pd.DataFrame, out_path: Path) -> None:
    """Plot route-family mean Nf with bootstrap CI."""
    plot_df = family_df.sort_values("nf_mean", ascending=True).copy()
    y = np.arange(len(plot_df))

    x = plot_df["nf_mean"].to_numpy(dtype=float)
    left = x - plot_df["bootstrap_mean_nf_ci_2p5"].to_numpy(dtype=float)
    right = plot_df["bootstrap_mean_nf_ci_97p5"].to_numpy(dtype=float) - x

    plt.figure(figsize=(9, 5))
    plt.errorbar(x, y, xerr=[left, right], fmt="o", capsize=3)
    plt.yticks(y, plot_df["route_family"])
    plt.xlabel("Mean fatigue life with 95% bootstrap CI (cycles)")
    plt.ylabel("Route family")
    plt.title("FAIR route-family fatigue-life metadata")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_report(
    df: pd.DataFrame,
    route_df: pd.DataFrame,
    family_df: pd.DataFrame,
    validation: pd.DataFrame,
    report_path: Path,
) -> None:
    status = "PASS" if (validation["status"] == "PASS").all() else "CHECK"

    top_routes = route_df[
        [
            "rank_by_mean_nf",
            "route_id",
            "route_family",
            "n_samples",
            "nf_mean",
            "bootstrap_mean_nf_ci_2p5",
            "bootstrap_mean_nf_ci_97p5",
            "shape_beta",
            "scale_eta_cycles",
            "B10_life_cycles",
            "B50_median_life_cycles",
            "B90_life_cycles",
        ]
    ].head(17)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Phase4.8_FAIR_bootstrap_CI_by_route_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {INPUT_SAMPLE_FEATURES}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        f.write(f"Bootstrap resamples B: {BOOTSTRAP_B}\n")
        f.write(f"Random seed: {BOOTSTRAP_SEED}\n")
        f.write(f"Analysis timestamp: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("Validation overview\n")
        f.write("-" * 78 + "\n")
        f.write(validation.to_string(index=False))
        f.write("\n\n")

        f.write("FAIR route-level reliability metadata preview\n")
        f.write("-" * 78 + "\n")
        f.write(top_routes.to_string(index=False))
        f.write("\n\n")

        f.write("FAIR route-family reliability metadata\n")
        f.write("-" * 78 + "\n")
        family_cols = [
            "rank_by_mean_nf",
            "route_family",
            "n_samples",
            "n_routes",
            "nf_mean",
            "bootstrap_mean_nf_ci_2p5",
            "bootstrap_mean_nf_ci_97p5",
            "shape_beta",
            "scale_eta_cycles",
            "B10_life_cycles",
            "B50_median_life_cycles",
            "B90_life_cycles",
        ]
        f.write(family_df[family_cols].to_string(index=False))
        f.write("\n\n")

        f.write("Generated files\n")
        f.write("-" * 78 + "\n")
        for path in [
            OUT_FAIR_ROUTE_METADATA,
            OUT_FAIR_ROUTE_FAMILY_METADATA,
            OUT_COLUMN_DICTIONARY,
            OUT_VALIDATION,
            OUT_JSON_METADATA,
            FIG_ROUTE_MEAN_CI,
            FIG_ROUTE_B_LIFE,
            FIG_ROUTE_FAMILY_MEAN_CI,
        ]:
            f.write(f"  {path}\n")

        f.write("\nInterpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "This table is a FAIR-ready reliability metadata layer derived from the "
            "85-row sample-level dataset. Bootstrap confidence intervals are computed "
            "at the sample level, not from cycle rows. Route-wise Weibull metadata are "
            "based on five samples per route and should be reported as exploratory "
            "route-wise reliability metadata rather than full industrial route-specific "
            "qualification.\n\n"
        )
        f.write(f"Final status: {status}\n")


def main() -> None:
    print("=== START Phase4.8_FAIR_bootstrap_CI_by_route_85 ===")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_SAMPLE_FEATURES.exists():
        raise FileNotFoundError(f"Input sample-level feature file not found: {INPUT_SAMPLE_FEATURES}")

    df = pd.read_csv(INPUT_SAMPLE_FEATURES)
    df = normalize_columns(df)
    print(f"Loaded sample-level rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    required = ["specimen_id", "route_id", "route_family", "cycles_to_failure"]
    require_columns(df, required, "sample_level_features_85.csv")

    df["specimen_id"] = df["specimen_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["route_family"] = df["route_family"].astype(str).str.strip()

    df["cycles_to_failure"] = pd.to_numeric(df["cycles_to_failure"], errors="coerce")
    if "log10_nf" not in df.columns:
        df["log10_nf"] = np.log10(df["cycles_to_failure"])
    else:
        df["log10_nf"] = pd.to_numeric(df["log10_nf"], errors="coerce")
        missing_log = df["log10_nf"].isna() & df["cycles_to_failure"].notna() & (df["cycles_to_failure"] > 0)
        df.loc[missing_log, "log10_nf"] = np.log10(df.loc[missing_log, "cycles_to_failure"])

    route_df = build_route_summary(df)
    family_df = build_route_family_summary(df)
    dictionary_df = make_column_dictionary()
    validation = build_validation(df, route_df)

    route_df.to_csv(OUT_FAIR_ROUTE_METADATA, index=False)
    family_df.to_csv(OUT_FAIR_ROUTE_FAMILY_METADATA, index=False)
    dictionary_df.to_csv(OUT_COLUMN_DICTIONARY, index=False)
    validation.to_csv(OUT_VALIDATION, index=False)

    analysis_metadata = {
        "script": SCRIPT_NAME,
        "input_file": str(INPUT_SAMPLE_FEATURES),
        "main_output": str(OUT_FAIR_ROUTE_METADATA),
        "n_samples": int(len(df)),
        "n_routes": int(df["route_id"].nunique()),
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "analysis_level": "route",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "interpretation_scope": (
            "Route-wise bootstrap CI and Weibull metadata derived from five samples per route; "
            "exploratory reliability metadata for FAIR database integration."
        ),
    }
    with open(OUT_JSON_METADATA, "w", encoding="utf-8") as f:
        json.dump(analysis_metadata, f, indent=2)

    plot_route_mean_ci(route_df, FIG_ROUTE_MEAN_CI)
    plot_route_b_life(route_df, FIG_ROUTE_B_LIFE)
    plot_route_family_mean_ci(family_df, FIG_ROUTE_FAMILY_MEAN_CI)

    write_report(df, route_df, family_df, validation, OUT_REPORT)

    print("\nValidation overview:")
    print(validation.to_string(index=False))

    print("\nFAIR route-level reliability metadata preview:")
    preview_cols = [
        "rank_by_mean_nf",
        "route_id",
        "route_family",
        "n_samples",
        "nf_mean",
        "bootstrap_mean_nf_ci_2p5",
        "bootstrap_mean_nf_ci_97p5",
        "shape_beta",
        "scale_eta_cycles",
        "B10_life_cycles",
        "B50_median_life_cycles",
        "B90_life_cycles",
    ]
    print(route_df[preview_cols].head(17).to_string(index=False))

    final_status = "PASS" if (validation["status"] == "PASS").all() else "CHECK"
    print(f"\nSaved FAIR route metadata to: {OUT_FAIR_ROUTE_METADATA}")
    print(f"Saved FAIR route-family metadata to: {OUT_FAIR_ROUTE_FAMILY_METADATA}")
    print(f"Saved report to: {OUT_REPORT}")
    print(f"\n✅ Done Phase4.8_FAIR_bootstrap_CI_by_route_85. Status: {final_status}")


if __name__ == "__main__":
    main()
