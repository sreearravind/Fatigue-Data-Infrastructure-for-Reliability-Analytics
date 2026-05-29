"""
task3.5A_bootstrap_CI_85.py

Bootstrap confidence interval analysis for the 85-sample fatigue dataset.

Purpose
-------
This script reads the ML-ready sample-level feature file created by
`task3.3_cycle_aggregated_85.py` and estimates percentile bootstrap
confidence intervals for fatigue life statistics.

Main input
----------
Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Main outputs
------------
Fatigue_85_augmented_dataset/04_statistics_outputs/
    task3_5A_global_bootstrap_CI_85.csv
    task3_5A_route_family_bootstrap_CI_85.csv
    task3_5A_routewise_bootstrap_CI_85.csv
    task3_5A_bootstrap_validation_overview_85.csv
    task3_5A_bootstrap_CI_report_85.txt

Fatigue_85_augmented_dataset/06_figures/
    task3_5A_global_bootstrap_mean_nf_85.png
    task3_5A_global_bootstrap_mean_log10nf_85.png
    task3_5A_routewise_mean_nf_CI_85.png
    task3_5A_route_family_mean_nf_CI_85.png

Notes
-----
- Cycle-level rows are NOT bootstrapped here.
- Bootstrap resampling is performed at the sample/specimen level.
- Route-wise results use n = 5 per route, so their CI values should be
  interpreted as uncertainty of the five-sample route set, not as a final
  industrial qualification estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------
def find_project_root() -> Path:
    """Return the project root folder containing Fatigue_85_augmented_dataset."""
    here = Path(__file__).resolve().parent
    candidates = [here, *here.parents]
    for candidate in candidates:
        if (candidate / "Fatigue_85_augmented_dataset").exists():
            return candidate
    # Fallback for the expected user path when running from db_scripts_85
    return here.parent


PROJECT_ROOT = find_project_root()
DATASET_DIR = PROJECT_ROOT / "Fatigue_85_augmented_dataset"
CLEANED_DIR = DATASET_DIR / "02_cleaned"
STATS_DIR = DATASET_DIR / "04_statistics_outputs"
FIG_DIR = DATASET_DIR / "06_figures"

INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"

EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5
BOOTSTRAP_B = 5000
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names for safe downstream access."""
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
        "route": "route_id",
        "nf": "cycles_to_failure",
        "n_f": "cycles_to_failure",
        "log10nf": "log10_nf",
        "log_nf": "log10_nf",
        "log10_n_f": "log10_nf",
        "ys": "ys_mpa",
        "uts": "uts_mpa",
    }
    out = out.rename(columns=rename_map)
    return coalesce_duplicate_columns(out)


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicate column names using first non-null value across duplicates."""
    if not df.columns.duplicated().any():
        return df

    new_df = pd.DataFrame(index=df.index)
    for col in pd.unique(df.columns):
        same = df.loc[:, df.columns == col]
        if same.shape[1] == 1:
            new_df[col] = same.iloc[:, 0]
        else:
            new_df[col] = same.bfill(axis=1).iloc[:, 0]
    return new_df


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def safe_numeric(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        n_bad = int(values.isna().sum())
        raise ValueError(f"Column {name} contains {n_bad} missing/non-numeric values.")
    return values.astype(float)


@dataclass
class BootstrapResult:
    group_label: str
    group_value: str
    metric: str
    statistic: str
    observed: float
    ci_2p5: float
    ci_50: float
    ci_97p5: float
    ci_width: float
    n_samples: int
    bootstrap_B: int


def bootstrap_metric(
    values: np.ndarray,
    *,
    metric: str,
    statistic: str,
    group_label: str,
    group_value: str,
    B: int,
    rng: np.random.Generator,
) -> BootstrapResult:
    """Bootstrap a statistic using sample-level resampling with replacement."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        raise ValueError(f"No finite values for {group_label}={group_value}, metric={metric}")

    if statistic == "mean":
        stat_func = np.mean
    elif statistic == "median":
        stat_func = np.median
    elif statistic == "std":
        # std of one value is defined here as 0 for stability, though all groups should be n>=5.
        stat_func = lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
    elif statistic == "cov":
        def stat_func(x: np.ndarray) -> float:
            mean = np.mean(x)
            if np.isclose(mean, 0):
                return np.nan
            return np.std(x, ddof=1) / mean if len(x) > 1 else 0.0
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

    boot = np.empty(B, dtype=float)
    for i in range(B):
        sample = values[rng.integers(0, n, size=n)]
        boot[i] = stat_func(sample)

    boot = boot[np.isfinite(boot)]
    ci_2p5, ci_50, ci_97p5 = np.percentile(boot, [2.5, 50, 97.5])
    observed = float(stat_func(values))

    return BootstrapResult(
        group_label=group_label,
        group_value=str(group_value),
        metric=metric,
        statistic=statistic,
        observed=observed,
        ci_2p5=float(ci_2p5),
        ci_50=float(ci_50),
        ci_97p5=float(ci_97p5),
        ci_width=float(ci_97p5 - ci_2p5),
        n_samples=n,
        bootstrap_B=B,
    )


def bootstrap_group(df: pd.DataFrame, group_label: str, group_value: str, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap core fatigue-life metrics for one group."""
    results: list[BootstrapResult] = []

    metric_map = {
        "cycles_to_failure": df["cycles_to_failure"].to_numpy(dtype=float),
        "log10_nf": df["log10_nf"].to_numpy(dtype=float),
    }

    for metric, values in metric_map.items():
        for statistic in ["mean", "median", "std", "cov"]:
            # CoV for log scale is not physically very useful; keep it out of the main result.
            if metric == "log10_nf" and statistic == "cov":
                continue
            results.append(
                bootstrap_metric(
                    values,
                    metric=metric,
                    statistic=statistic,
                    group_label=group_label,
                    group_value=group_value,
                    B=BOOTSTRAP_B,
                    rng=rng,
                )
            )

    return pd.DataFrame([r.__dict__ for r in results])


def make_validation(df: pd.DataFrame) -> pd.DataFrame:
    route_counts = df.groupby("route_id")["specimen_id"].nunique() if "route_id" in df.columns else pd.Series(dtype=int)
    checks = [
        {
            "check": "Total sample rows",
            "expected": EXPECTED_SAMPLES,
            "observed": len(df),
            "status": "PASS" if len(df) == EXPECTED_SAMPLES else "FAIL",
        },
        {
            "check": "Unique specimen_id count",
            "expected": EXPECTED_SAMPLES,
            "observed": df["specimen_id"].nunique(),
            "status": "PASS" if df["specimen_id"].nunique() == EXPECTED_SAMPLES else "FAIL",
        },
        {
            "check": "Unique route_id count",
            "expected": EXPECTED_ROUTES,
            "observed": df["route_id"].nunique(),
            "status": "PASS" if df["route_id"].nunique() == EXPECTED_ROUTES else "FAIL",
        },
        {
            "check": "Routes with exactly five samples",
            "expected": EXPECTED_ROUTES,
            "observed": int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()),
            "status": "PASS" if int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES else "FAIL",
        },
        {
            "check": "Duplicate specimen_id rows",
            "expected": 0,
            "observed": int(df.duplicated("specimen_id").sum()),
            "status": "PASS" if int(df.duplicated("specimen_id").sum()) == 0 else "FAIL",
        },
        {
            "check": "Missing cycles_to_failure",
            "expected": 0,
            "observed": int(df["cycles_to_failure"].isna().sum()),
            "status": "PASS" if int(df["cycles_to_failure"].isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Missing log10_nf",
            "expected": 0,
            "observed": int(df["log10_nf"].isna().sum()),
            "status": "PASS" if int(df["log10_nf"].isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Non-positive fatigue life",
            "expected": 0,
            "observed": int((df["cycles_to_failure"] <= 0).sum()),
            "status": "PASS" if int((df["cycles_to_failure"] <= 0).sum()) == 0 else "FAIL",
        },
    ]
    return pd.DataFrame(checks)


def save_global_figures(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Save bootstrap distributions for global mean Nf and mean log10(Nf)."""
    nf = df["cycles_to_failure"].to_numpy(dtype=float)
    log_nf = df["log10_nf"].to_numpy(dtype=float)
    n = len(nf)

    boot_nf = np.empty(BOOTSTRAP_B, dtype=float)
    boot_log = np.empty(BOOTSTRAP_B, dtype=float)
    for i in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        boot_nf[i] = nf[idx].mean()
        boot_log[i] = log_nf[idx].mean()

    ci_nf = np.percentile(boot_nf, [2.5, 50, 97.5])
    plt.figure(figsize=(8, 5))
    plt.hist(boot_nf, bins=40)
    plt.axvline(ci_nf[0], linestyle="--")
    plt.axvline(ci_nf[1], linestyle="-")
    plt.axvline(ci_nf[2], linestyle="--")
    plt.xlabel("Bootstrapped mean Nf (cycles)")
    plt.ylabel("Count")
    plt.title("Bootstrap distribution of global mean fatigue life")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task3_5A_global_bootstrap_mean_nf_85.png", dpi=300)
    plt.close()

    ci_log = np.percentile(boot_log, [2.5, 50, 97.5])
    plt.figure(figsize=(8, 5))
    plt.hist(boot_log, bins=40)
    plt.axvline(ci_log[0], linestyle="--")
    plt.axvline(ci_log[1], linestyle="-")
    plt.axvline(ci_log[2], linestyle="--")
    plt.xlabel("Bootstrapped mean log10(Nf)")
    plt.ylabel("Count")
    plt.title("Bootstrap distribution of global mean log10 fatigue life")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "task3_5A_global_bootstrap_mean_log10nf_85.png", dpi=300)
    plt.close()


def save_ci_plot(ci_df: pd.DataFrame, group_label: str, out_name: str, title: str) -> None:
    """Save horizontal error-bar plot for mean Nf CI by group."""
    plot_df = ci_df[
        (ci_df["metric"] == "cycles_to_failure") &
        (ci_df["statistic"] == "mean") &
        (ci_df["group_label"] == group_label)
    ].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("observed", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    x = plot_df["observed"].to_numpy(dtype=float)
    left = x - plot_df["ci_2p5"].to_numpy(dtype=float)
    right = plot_df["ci_97p5"].to_numpy(dtype=float) - x

    plt.figure(figsize=(9, max(4, 0.35 * len(plot_df) + 1.5)))
    plt.errorbar(x, y, xerr=[left, right], fmt="o", capsize=3)
    plt.yticks(y, plot_df["group_value"])
    plt.xlabel("Mean fatigue life with 95% bootstrap CI (cycles)")
    plt.ylabel(group_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / out_name, dpi=300)
    plt.close()


def main() -> None:
    print("=== START task3.5A_bootstrap_CI_85 ===")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    raw = pd.read_csv(INPUT_FILE)
    print("Raw columns:", raw.columns.tolist())
    df = normalise_columns(raw)
    print("Normalized columns:", df.columns.tolist())

    require_columns(df, ["specimen_id", "route_id", "route_family", "cycles_to_failure"])

    df["specimen_id"] = df["specimen_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["route_family"] = df["route_family"].astype(str).str.strip()
    df["cycles_to_failure"] = safe_numeric(df["cycles_to_failure"], "cycles_to_failure")

    if "log10_nf" not in df.columns:
        df["log10_nf"] = np.log10(df["cycles_to_failure"])
    else:
        df["log10_nf"] = pd.to_numeric(df["log10_nf"], errors="coerce")
        missing_log = df["log10_nf"].isna()
        if missing_log.any():
            df.loc[missing_log, "log10_nf"] = np.log10(df.loc[missing_log, "cycles_to_failure"])

    validation = make_validation(df)
    validation_path = STATS_DIR / "task3_5A_bootstrap_validation_overview_85.csv"
    validation.to_csv(validation_path, index=False)
    print("\nValidation overview:")
    print(validation.to_string(index=False))

    if (validation["status"] == "FAIL").any():
        raise ValueError("Validation failed. Please review task3_5A_bootstrap_validation_overview_85.csv")

    rng = np.random.default_rng(RANDOM_SEED)

    # Global bootstrap
    global_df = bootstrap_group(df, "global", "all_samples", rng)
    global_path = STATS_DIR / "task3_5A_global_bootstrap_CI_85.csv"
    global_df.to_csv(global_path, index=False)

    # Route-family bootstrap
    family_results = []
    for family, group in df.groupby("route_family", sort=True):
        family_results.append(bootstrap_group(group, "route_family", family, rng))
    family_df = pd.concat(family_results, ignore_index=True)
    family_path = STATS_DIR / "task3_5A_route_family_bootstrap_CI_85.csv"
    family_df.to_csv(family_path, index=False)

    # Route-wise bootstrap
    route_results = []
    for route, group in df.groupby("route_id", sort=True):
        route_results.append(bootstrap_group(group, "route_id", route, rng))
    route_df = pd.concat(route_results, ignore_index=True)
    route_path = STATS_DIR / "task3_5A_routewise_bootstrap_CI_85.csv"
    route_df.to_csv(route_path, index=False)

    # Compact route ranking table for direct manuscript/workflow interpretation
    route_mean = route_df[
        (route_df["metric"] == "cycles_to_failure") &
        (route_df["statistic"] == "mean")
    ].copy()
    route_mean = route_mean.sort_values("observed", ascending=False)
    route_mean["rank_by_mean_nf"] = np.arange(1, len(route_mean) + 1)
    route_ranking_path = STATS_DIR / "task3_5A_routewise_mean_nf_CI_ranking_85.csv"
    route_mean.to_csv(route_ranking_path, index=False)

    # Figures
    fig_rng = np.random.default_rng(RANDOM_SEED + 100)
    save_global_figures(df, fig_rng)
    save_ci_plot(
        route_df,
        "route_id",
        "task3_5A_routewise_mean_nf_CI_85.png",
        "Route-wise mean fatigue life with 95% bootstrap CI",
    )
    save_ci_plot(
        family_df,
        "route_family",
        "task3_5A_route_family_mean_nf_CI_85.png",
        "Route-family mean fatigue life with 95% bootstrap CI",
    )

    # Report
    global_mean_nf = global_df[(global_df["metric"] == "cycles_to_failure") & (global_df["statistic"] == "mean")].iloc[0]
    global_mean_log = global_df[(global_df["metric"] == "log10_nf") & (global_df["statistic"] == "mean")].iloc[0]

    report_path = STATS_DIR / "task3_5A_bootstrap_CI_report_85.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("task3.5A_bootstrap_CI_85 report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        f.write(f"Bootstrap resamples B: {BOOTSTRAP_B}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("Validation overview\n")
        f.write("-" * 70 + "\n")
        f.write(validation.to_string(index=False))
        f.write("\n\n")

        f.write("Global bootstrap confidence intervals\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Mean Nf observed = {global_mean_nf['observed']:.4f}, "
            f"95% CI = [{global_mean_nf['ci_2p5']:.4f}, {global_mean_nf['ci_97p5']:.4f}], "
            f"median bootstrap = {global_mean_nf['ci_50']:.4f}\n"
        )
        f.write(
            f"Mean log10(Nf) observed = {global_mean_log['observed']:.6f}, "
            f"95% CI = [{global_mean_log['ci_2p5']:.6f}, {global_mean_log['ci_97p5']:.6f}], "
            f"median bootstrap = {global_mean_log['ci_50']:.6f}\n\n"
        )

        f.write("Route-family mean Nf bootstrap CI\n")
        f.write("-" * 70 + "\n")
        family_mean = family_df[(family_df["metric"] == "cycles_to_failure") & (family_df["statistic"] == "mean")]
        f.write(family_mean.sort_values("observed", ascending=False).to_string(index=False))
        f.write("\n\n")

        f.write("Top route-wise mean Nf bootstrap CI ranking\n")
        f.write("-" * 70 + "\n")
        f.write(route_mean.head(17).to_string(index=False))
        f.write("\n\n")

        f.write("Generated files\n")
        f.write("-" * 70 + "\n")
        for p in [
            global_path,
            family_path,
            route_path,
            route_ranking_path,
            validation_path,
            FIG_DIR / "task3_5A_global_bootstrap_mean_nf_85.png",
            FIG_DIR / "task3_5A_global_bootstrap_mean_log10nf_85.png",
            FIG_DIR / "task3_5A_routewise_mean_nf_CI_85.png",
            FIG_DIR / "task3_5A_route_family_mean_nf_CI_85.png",
        ]:
            f.write(f"  {p}\n")

        f.write("\nInterpretation note:\n")
        f.write(
            "Bootstrap resampling was performed at the sample level, not the cycle-row level. "
            "Route-wise confidence intervals are based on five samples per route and should be "
            "interpreted as uncertainty estimates for the expanded route-level sample set.\n"
        )
        f.write("\nFinal status: PASS\n")

    print("\nGlobal bootstrap summary:")
    print(global_df.to_string(index=False))
    print(f"\nSaved global CI to: {global_path}")
    print(f"Saved route-family CI to: {family_path}")
    print(f"Saved route-wise CI to: {route_path}")
    print(f"Saved report to: {report_path}")
    print("\n✅ Done task3.5A_bootstrap_CI_85. Status: PASS")


if __name__ == "__main__":
    main()
