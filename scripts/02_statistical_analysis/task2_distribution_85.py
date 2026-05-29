"""
task2_distribution_85.py

Distribution-level statistical characterization for the 85-sample fatigue dataset.

Run after:
    task3.3_cycle_aggregated_85.py

Input:
    Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Outputs:
    Fatigue_85_augmented_dataset/04_statistics_outputs/
    Fatigue_85_augmented_dataset/06_figures/
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import shapiro, weibull_min, skew, kurtosis
except Exception as exc:
    raise ImportError("This script requires scipy. Install it using: pip install scipy") from exc


# -------------------------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------------------------
def find_project_root() -> Path:
    """Expected placement: <project_root>/db_scripts_85/task2_distribution_85.py"""
    current = Path(__file__).resolve()
    if current.parent.name.lower() == "db_scripts_85":
        return current.parent.parent
    return current.parent


PROJECT_ROOT = find_project_root()
DATASET_DIR = PROJECT_ROOT / "Fatigue_85_augmented_dataset"

CLEANED_DIR = DATASET_DIR / "02_cleaned"
STATS_DIR = DATASET_DIR / "04_statistics_outputs"
FIG_DIR = DATASET_DIR / "06_figures"

INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"

OUT_GLOBAL = STATS_DIR / "task2_distribution_global_summary_85.csv"
OUT_ROUTE = STATS_DIR / "task2_routewise_nf_summary_85.csv"
OUT_FAMILY = STATS_DIR / "task2_route_family_distribution_85.csv"
OUT_SYNTHETIC = STATS_DIR / "task2_is_synthetic_tracking_summary_85.csv"
OUT_VALIDATION = STATS_DIR / "task2_distribution_validation_overview_85.csv"
OUT_REPORT = STATS_DIR / "task2_distribution_report_85.txt"

FIG_HIST_NF = FIG_DIR / "task2_hist_nf_85.png"
FIG_HIST_LOG = FIG_DIR / "task2_hist_log10nf_85.png"
FIG_WEIBULL = FIG_DIR / "task2_weibull_probplot_85.png"
FIG_ROUTE_BOX = FIG_DIR / "task2_boxplot_routewise_log10nf_85.png"
FIG_FAMILY_BOX = FIG_DIR / "task2_boxplot_route_family_log10nf_85.png"


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
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
        "sample_id": "specimen_id",
        "sampleid": "specimen_id",
        "specimenid": "specimen_id",
        "nf": "cycles_to_failure",
        "n_f": "cycles_to_failure",
        "nf_cycles": "cycles_to_failure",
        "n_cycles": "cycles_to_failure",
        "log10nf": "log10_nf",
        "log_nf": "log10_nf",
        "log10_n_f": "log10_nf",
        "grain_size": "grain_size_um",
    }
    return df.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def safe_cv(series: pd.Series) -> float:
    mean_value = series.mean()
    if pd.isna(mean_value) or math.isclose(float(mean_value), 0.0):
        return np.nan
    return float(series.std(ddof=1) / mean_value)


def summary_stats(group: pd.DataFrame, value_col: str = "cycles_to_failure") -> pd.Series:
    nf = pd.to_numeric(group[value_col], errors="coerce").dropna()
    log_nf = pd.to_numeric(group["log10_nf"], errors="coerce").dropna()
    return pd.Series({
        "n_samples": int(len(nf)),
        "nf_min": nf.min(),
        "nf_q25": nf.quantile(0.25),
        "nf_median": nf.median(),
        "nf_q75": nf.quantile(0.75),
        "nf_max": nf.max(),
        "nf_mean": nf.mean(),
        "nf_sd": nf.std(ddof=1),
        "nf_cov": safe_cv(nf),
        "log10_nf_min": log_nf.min(),
        "log10_nf_mean": log_nf.mean(),
        "log10_nf_median": log_nf.median(),
        "log10_nf_sd": log_nf.std(ddof=1),
    })


def validation_row(check: str, expected: object, observed: object, passed: bool) -> dict:
    return {"check": check, "expected": expected, "observed": observed, "status": "PASS" if passed else "FAIL"}


def save_histogram_nf(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["cycles_to_failure"], bins=12)
    plt.xlabel("Fatigue life, Nf (cycles)")
    plt.ylabel("Sample count")
    plt.title("Distribution of fatigue life across 85 samples")
    plt.tight_layout()
    plt.savefig(FIG_HIST_NF, dpi=300)
    plt.close()


def save_histogram_log_nf(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["log10_nf"], bins=12)
    plt.xlabel("log10(Nf)")
    plt.ylabel("Sample count")
    plt.title("Distribution of log10(Nf) across 85 samples")
    plt.tight_layout()
    plt.savefig(FIG_HIST_LOG, dpi=300)
    plt.close()


def save_weibull_probability_plot(df: pd.DataFrame) -> tuple[float, float]:
    nf_values = pd.to_numeric(df["cycles_to_failure"], errors="coerce").dropna().astype(float).values
    nf_values = nf_values[nf_values > 0]
    if len(nf_values) < 3:
        raise ValueError("At least 3 positive Nf values are required for Weibull fitting.")

    shape, loc, scale = weibull_min.fit(nf_values, floc=0)

    nf_sorted = np.sort(nf_values)
    n = len(nf_sorted)
    failure_probability = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
    x = np.log(nf_sorted)
    y = np.log(-np.log(1 - failure_probability))

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = shape * x_line - shape * np.log(scale)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=20)
    plt.plot(x_line, y_line)
    plt.xlabel("ln(Nf)")
    plt.ylabel("ln(-ln(1-F))")
    plt.title("Weibull probability plot for 85-sample fatigue life")
    plt.tight_layout()
    plt.savefig(FIG_WEIBULL, dpi=300)
    plt.close()
    return float(shape), float(scale)


def save_routewise_boxplot(df: pd.DataFrame) -> None:
    route_order = sorted(df["route_id"].dropna().unique().tolist())
    data = [df.loc[df["route_id"] == route, "log10_nf"].dropna().values for route in route_order]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=route_order, showmeans=True)
    plt.xlabel("Processing route")
    plt.ylabel("log10(Nf)")
    plt.title("Route-wise fatigue life distribution across 85 samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_ROUTE_BOX, dpi=300)
    plt.close()


def save_family_boxplot(df: pd.DataFrame) -> None:
    if "route_family" not in df.columns:
        return
    family_order = sorted(df["route_family"].dropna().unique().tolist())
    data = [df.loc[df["route_family"] == family, "log10_nf"].dropna().values for family in family_order]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=family_order, showmeans=True)
    plt.xlabel("Route family")
    plt.ylabel("log10(Nf)")
    plt.title("Route-family fatigue life distribution across 85 samples")
    plt.tight_layout()
    plt.savefig(FIG_FAMILY_BOX, dpi=300)
    plt.close()


def write_report(
    df: pd.DataFrame,
    global_summary: pd.DataFrame,
    route_summary: pd.DataFrame,
    family_summary: Optional[pd.DataFrame],
    validation_df: pd.DataFrame,
) -> None:
    row = global_summary.iloc[0].to_dict()
    status = "PASS" if (validation_df["status"] == "PASS").all() else "FAIL"

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("task2_distribution_85 report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows in sample-level file: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n\n")

        f.write("Global fatigue-life distribution:\n")
        f.write(f"  Nf min: {row['nf_min']:.4f}\n")
        f.write(f"  Nf max: {row['nf_max']:.4f}\n")
        f.write(f"  Nf mean: {row['nf_mean']:.4f}\n")
        f.write(f"  Nf SD: {row['nf_sd']:.4f}\n")
        f.write(f"  Nf CoV: {row['nf_cov']:.4f}\n")
        f.write(f"  log10(Nf) mean: {row['log10_nf_mean']:.6f}\n")
        f.write(f"  log10(Nf) SD: {row['log10_nf_sd']:.6f}\n")
        f.write(f"  Shapiro-Wilk W on log10(Nf): {row['shapiro_w_log10_nf']:.6f}\n")
        f.write(f"  Shapiro-Wilk p on log10(Nf): {row['shapiro_p_log10_nf']:.6f}\n")
        f.write(f"  Weibull shape beta: {row['weibull_shape_beta']:.6f}\n")
        f.write(f"  Weibull scale eta: {row['weibull_scale_eta']:.6f}\n")
        f.write(f"  Skewness Nf: {row['nf_skewness']:.6f}\n")
        f.write(f"  Kurtosis Nf: {row['nf_kurtosis']:.6f}\n\n")

        f.write("Route-wise summary preview:\n")
        preview_cols = ["route_id", "n_samples", "nf_mean", "nf_sd", "nf_cov", "log10_nf_mean"]
        f.write(route_summary[preview_cols].to_string(index=False))
        f.write("\n\n")

        if family_summary is not None:
            f.write("Route-family summary:\n")
            f.write(family_summary.to_string(index=False))
            f.write("\n\n")

        f.write("Validation overview:\n")
        f.write(validation_df.to_string(index=False))
        f.write("\n\n")

        f.write("Generated files:\n")
        for path in [
            OUT_GLOBAL, OUT_ROUTE, OUT_FAMILY, OUT_VALIDATION,
            FIG_HIST_NF, FIG_HIST_LOG, FIG_WEIBULL, FIG_ROUTE_BOX, FIG_FAMILY_BOX,
        ]:
            f.write(f"  {path}\n")
        if OUT_SYNTHETIC.exists():
            f.write(f"  {OUT_SYNTHETIC}\n")
        f.write(f"\nFinal status: {status}\n")


def main() -> None:
    print("=== START task2_distribution_85 ===")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}\nRun task3.3_cycle_aggregated_85.py first.")

    raw = pd.read_csv(INPUT_FILE)
    print("Raw columns:", raw.columns.tolist())
    df = normalize_columns(raw)
    print("Normalized columns:", df.columns.tolist())

    require_columns(df, ["specimen_id", "route_id", "cycles_to_failure"])

    df["specimen_id"] = df["specimen_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["cycles_to_failure"] = pd.to_numeric(df["cycles_to_failure"], errors="coerce")

    if "log10_nf" not in df.columns:
        df["log10_nf"] = np.nan
    df["log10_nf"] = pd.to_numeric(df["log10_nf"], errors="coerce")
    recompute_mask = df["log10_nf"].isna() & df["cycles_to_failure"].gt(0)
    df.loc[recompute_mask, "log10_nf"] = np.log10(df.loc[recompute_mask, "cycles_to_failure"])

    invalid_mask = df["cycles_to_failure"].isna() | df["cycles_to_failure"].le(0)
    if invalid_mask.any():
        bad_path = STATS_DIR / "task2_invalid_nf_rows_85.csv"
        df.loc[invalid_mask].to_csv(bad_path, index=False)
        raise ValueError(f"Invalid cycles_to_failure rows found: {int(invalid_mask.sum())}. Saved to {bad_path}")

    nf = df["cycles_to_failure"].astype(float)
    log_nf = df["log10_nf"].astype(float)

    shapiro_w, shapiro_p = shapiro(log_nf)
    weibull_shape, weibull_scale = save_weibull_probability_plot(df)

    global_summary = pd.DataFrame([{
        "n_samples": int(df["specimen_id"].nunique()),
        "n_routes": int(df["route_id"].nunique()),
        "nf_min": nf.min(),
        "nf_q25": nf.quantile(0.25),
        "nf_median": nf.median(),
        "nf_q75": nf.quantile(0.75),
        "nf_max": nf.max(),
        "nf_mean": nf.mean(),
        "nf_sd": nf.std(ddof=1),
        "nf_cov": safe_cv(nf),
        "log10_nf_min": log_nf.min(),
        "log10_nf_q25": log_nf.quantile(0.25),
        "log10_nf_median": log_nf.median(),
        "log10_nf_q75": log_nf.quantile(0.75),
        "log10_nf_max": log_nf.max(),
        "log10_nf_mean": log_nf.mean(),
        "log10_nf_sd": log_nf.std(ddof=1),
        "shapiro_w_log10_nf": float(shapiro_w),
        "shapiro_p_log10_nf": float(shapiro_p),
        "weibull_shape_beta": weibull_shape,
        "weibull_scale_eta": weibull_scale,
        "nf_skewness": float(skew(nf, bias=False)),
        "nf_kurtosis": float(kurtosis(nf, bias=False)),
    }])

    route_summary = (
        df.groupby("route_id", as_index=False)
        .apply(summary_stats, include_groups=False)
        .reset_index(drop=True)
        .sort_values("route_id")
    )

    family_summary = None
    if "route_family" in df.columns:
        family_summary = (
            df.groupby("route_family", as_index=False)
            .apply(summary_stats, include_groups=False)
            .reset_index(drop=True)
            .sort_values("route_family")
        )

    if "is_synthetic" in df.columns:
        synthetic_summary = (
            df.groupby(["route_id", "is_synthetic"], dropna=False)
            .agg(n_samples=("specimen_id", "nunique"))
            .reset_index()
            .sort_values(["route_id", "is_synthetic"])
        )
        synthetic_summary.to_csv(OUT_SYNTHETIC, index=False)

    sample_counts_per_route = df.groupby("route_id")["specimen_id"].nunique()
    numeric_values = df.select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(numeric_values).sum().sum())
    validation_df = pd.DataFrame([
        validation_row("Total rows", 85, len(df), len(df) == 85),
        validation_row("Unique specimen_id count", 85, df["specimen_id"].nunique(), df["specimen_id"].nunique() == 85),
        validation_row("Unique route_id count", 17, df["route_id"].nunique(), df["route_id"].nunique() == 17),
        validation_row("Routes with exactly five samples", 17, int((sample_counts_per_route == 5).sum()), int((sample_counts_per_route == 5).sum()) == 17),
        validation_row("Duplicate specimen_id rows", 0, int(df.duplicated("specimen_id").sum()), int(df.duplicated("specimen_id").sum()) == 0),
        validation_row("Missing cycles_to_failure", 0, int(df["cycles_to_failure"].isna().sum()), int(df["cycles_to_failure"].isna().sum()) == 0),
        validation_row("Missing log10_nf", 0, int(df["log10_nf"].isna().sum()), int(df["log10_nf"].isna().sum()) == 0),
        validation_row("Non-positive fatigue life", 0, int((df["cycles_to_failure"] <= 0).sum()), int((df["cycles_to_failure"] <= 0).sum()) == 0),
        validation_row("Infinite numeric values", 0, infinite_count, infinite_count == 0),
    ])

    save_histogram_nf(df)
    save_histogram_log_nf(df)
    save_routewise_boxplot(df)
    save_family_boxplot(df)

    global_summary.to_csv(OUT_GLOBAL, index=False)
    route_summary.to_csv(OUT_ROUTE, index=False)
    if family_summary is not None:
        family_summary.to_csv(OUT_FAMILY, index=False)
    validation_df.to_csv(OUT_VALIDATION, index=False)
    write_report(df, global_summary, route_summary, family_summary, validation_df)

    print("\nGlobal distribution summary:")
    print(global_summary.to_string(index=False))
    print("\nRoute-wise summary preview:")
    print(route_summary.head().to_string(index=False))
    print("\nValidation overview:")
    print(validation_df.to_string(index=False))
    print(f"\nSaved global summary to: {OUT_GLOBAL}")
    print(f"Saved route-wise summary to: {OUT_ROUTE}")
    print(f"Saved report to: {OUT_REPORT}")
    print(f"Saved figures to: {FIG_DIR}")

    final_status = "PASS" if (validation_df["status"] == "PASS").all() else "FAIL"
    print(f"\n✅ Done task2_distribution_85. Status: {final_status}")


if __name__ == "__main__":
    main()
