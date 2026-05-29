"""
task3.5B_weibull_reliability_85.py
------------------------------------------------------------
Version: v2 fixed route-family plotting scale column name
Purpose
    Reliability-oriented Weibull analysis for the 85-sample fatigue dataset.

Input
    Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Outputs
    Fatigue_85_augmented_dataset/04_statistics_outputs/
        task3_5B_weibull_global_parameters_85.csv
        task3_5B_weibull_survival_probabilities_85.csv
        task3_5B_weibull_life_quantiles_85.csv
        task3_5B_route_family_weibull_parameters_85.csv
        task3_5B_route_family_survival_probabilities_85.csv
        task3_5B_routewise_weibull_parameters_85.csv
        task3_5B_routewise_life_quantiles_85.csv
        task3_5B_weibull_bootstrap_parameter_CI_85.csv
        task3_5B_weibull_validation_overview_85.csv
        task3_5B_weibull_reliability_report_85.txt

    Fatigue_85_augmented_dataset/06_figures/
        task3_5B_global_weibull_reliability_curve_85.png
        task3_5B_global_weibull_probability_plot_85.png
        task3_5B_route_family_weibull_reliability_curves_85.png
        task3_5B_routewise_b10_median_b90_85.png

Notes
    1. This script fits a two-parameter Weibull distribution with loc fixed at 0.
    2. Global Weibull results should be interpreted as cross-route fatigue-life distribution.
    3. Route-wise Weibull fitting is based on n = 5 samples per route and should be treated
       as exploratory route-level reliability metadata, not final industrial reliability.
    4. ML must still use the 85-row sample-level dataset, not the cycle-level rows.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# ---------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------
try:
    # Preferred when script is placed inside db_scripts_85 with db_config_85.py
    from db_config_85 import CLEANED_DIR, STATS_OUTPUT_DIR, FIGURES_DIR  # type: ignore
except Exception:
    # Fallback for standalone execution from project root/db_scripts_85
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "db_scripts_85" else SCRIPT_DIR
    DATASET_DIR = PROJECT_DIR / "Fatigue_85_augmented_dataset"
    CLEANED_DIR = DATASET_DIR / "02_cleaned"
    STATS_OUTPUT_DIR = DATASET_DIR / "04_statistics_outputs"
    FIGURES_DIR = DATASET_DIR / "06_figures"

INPUT_FILE = Path(CLEANED_DIR) / "sample_level_features_85.csv"
STATS_OUTPUT_DIR = Path(STATS_OUTPUT_DIR)
FIGURES_DIR = Path(FIGURES_DIR)

EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5
BOOTSTRAP_B = 3000
RANDOM_SEED = 42
SURVIVAL_CYCLES = np.array([250, 500, 1000, 1500, 2000, 3000, 5000, 7000], dtype=float)
FAILURE_PROBABILITIES = np.array([0.01, 0.05, 0.10, 0.50, 0.90], dtype=float)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
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
        "log10nf": "log10_nf",
        "log_nf": "log10_nf",
        "log10_n_f": "log10_nf",
    }
    return df.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}\nAvailable columns: {df.columns.tolist()}")


def fit_weibull_2p(values: np.ndarray) -> Tuple[float, float]:
    """Return shape beta and scale eta for a 2-parameter Weibull fit."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 3:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beta, loc, eta = weibull_min.fit(values, floc=0)
    return float(beta), float(eta)


def survival_probability(cycles: np.ndarray, beta: float, eta: float) -> np.ndarray:
    cycles = np.asarray(cycles, dtype=float)
    if not np.isfinite(beta) or not np.isfinite(eta) or beta <= 0 or eta <= 0:
        return np.full_like(cycles, np.nan, dtype=float)
    return np.exp(-((cycles / eta) ** beta))


def weibull_quantile(p_failure: np.ndarray, beta: float, eta: float) -> np.ndarray:
    p_failure = np.asarray(p_failure, dtype=float)
    if not np.isfinite(beta) or not np.isfinite(eta) or beta <= 0 or eta <= 0:
        return np.full_like(p_failure, np.nan, dtype=float)
    return eta * (-np.log(1.0 - p_failure)) ** (1.0 / beta)


def median_rank(n: int) -> np.ndarray:
    ranks = np.arange(1, n + 1)
    return (ranks - 0.3) / (n + 0.4)


def safe_cv(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    mean = values.mean()
    if mean == 0:
        return np.nan
    return float(values.std(ddof=1) / mean)


def bootstrap_weibull_ci(values: np.ndarray, B: int, seed: int) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    rng = np.random.default_rng(seed)
    beta_boot: List[float] = []
    eta_boot: List[float] = []
    b10_boot: List[float] = []
    b50_boot: List[float] = []
    b90_boot: List[float] = []

    n = len(values)
    for _ in range(B):
        sample = values[rng.integers(0, n, size=n)]
        beta, eta = fit_weibull_2p(sample)
        if np.isfinite(beta) and np.isfinite(eta):
            beta_boot.append(beta)
            eta_boot.append(eta)
            q = weibull_quantile(np.array([0.10, 0.50, 0.90]), beta, eta)
            b10_boot.append(float(q[0]))
            b50_boot.append(float(q[1]))
            b90_boot.append(float(q[2]))

    rows = []
    for metric, arr in [
        ("shape_beta", beta_boot),
        ("scale_eta", eta_boot),
        ("B10_life_cycles", b10_boot),
        ("B50_median_life_cycles", b50_boot),
        ("B90_life_cycles", b90_boot),
    ]:
        arr_np = np.asarray(arr, dtype=float)
        rows.append({
            "metric": metric,
            "ci_2p5": float(np.percentile(arr_np, 2.5)) if len(arr_np) else np.nan,
            "ci_50": float(np.percentile(arr_np, 50)) if len(arr_np) else np.nan,
            "ci_97p5": float(np.percentile(arr_np, 97.5)) if len(arr_np) else np.nan,
            "n_boot_success": int(len(arr_np)),
            "bootstrap_B_requested": int(B),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_global_reliability(nf: np.ndarray, beta: float, eta: float, out_path: Path) -> None:
    x_max = max(float(np.nanmax(nf) * 1.15), 1.0)
    x = np.linspace(1, x_max, 400)
    r = survival_probability(x, beta, eta)

    plt.figure(figsize=(9, 5.5))
    plt.plot(x, r, linewidth=2)
    plt.scatter(SURVIVAL_CYCLES, survival_probability(SURVIVAL_CYCLES, beta, eta), zorder=3)
    plt.xlabel("Cycles, N")
    plt.ylabel("Survival probability, R(N)")
    plt.title("Global Weibull reliability curve for 85-sample fatigue life")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_weibull_probability(nf: np.ndarray, beta: float, eta: float, out_path: Path) -> None:
    nf_sorted = np.sort(np.asarray(nf, dtype=float))
    F = median_rank(len(nf_sorted))
    x = np.log(nf_sorted)
    y = np.log(-np.log(1 - F))

    x_line = np.linspace(x.min(), x.max(), 200)
    # Weibull probability line: y = beta * ln(N) - beta * ln(eta)
    y_line = beta * x_line - beta * np.log(eta)

    plt.figure(figsize=(9, 5.5))
    plt.scatter(x, y, label="Median-rank data")
    plt.plot(x_line, y_line, label=f"2P Weibull fit: β={beta:.3f}, η={eta:.1f}")
    plt.xlabel("ln(Nf)")
    plt.ylabel("ln(-ln(1-F))")
    plt.title("Global Weibull probability plot for 85-sample fatigue life")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_route_family_reliability(family_params: pd.DataFrame, nf_max: float, out_path: Path) -> None:
    """Plot route-family reliability curves.

    The parameter table may use either `scale_eta` or the clearer
    `scale_eta_cycles` column name. This helper accepts both so the
    plotting stage does not fail after CSV/report-friendly naming.
    """
    scale_col = "scale_eta" if "scale_eta" in family_params.columns else "scale_eta_cycles"
    required = ["shape_beta", scale_col]

    x = np.linspace(1, nf_max * 1.15, 400)
    plt.figure(figsize=(9, 5.5))
    for _, row in family_params.dropna(subset=required).iterrows():
        r = survival_probability(x, row["shape_beta"], row[scale_col])
        plt.plot(x, r, linewidth=2, label=str(row["route_family"]))
    plt.xlabel("Cycles, N")
    plt.ylabel("Survival probability, R(N)")
    plt.title("Route-family Weibull reliability curves")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_routewise_quantiles(route_quantiles: pd.DataFrame, out_path: Path) -> None:
    pivot = route_quantiles.pivot(index="route_id", columns="failure_probability", values="life_cycles_np")
    pivot = pivot.rename(columns={0.10: "B10", 0.50: "B50", 0.90: "B90"})
    pivot = pivot.sort_values("B50", ascending=True)

    y = np.arange(len(pivot))
    plt.figure(figsize=(9, max(5.5, len(pivot) * 0.35)))
    plt.hlines(y, pivot["B10"], pivot["B90"], linewidth=2)
    plt.scatter(pivot["B50"], y, zorder=3, label="B50 median")
    plt.scatter(pivot["B10"], y, marker="|", s=80, label="B10")
    plt.scatter(pivot["B90"], y, marker="|", s=80, label="B90")
    plt.yticks(y, pivot.index)
    plt.xlabel("Weibull life quantile, cycles")
    plt.ylabel("Processing route")
    plt.title("Route-wise Weibull B10-B50-B90 life estimates")
    plt.grid(True, axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    print("=== START task3.5B_weibull_reliability_85 ===")
    STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input sample-level feature file not found: {INPUT_FILE}")

    df = normalize_columns(pd.read_csv(INPUT_FILE))
    require_columns(df, ["specimen_id", "route_id", "cycles_to_failure"], "sample-level feature")
    if "route_family" not in df.columns:
        df["route_family"] = "Unknown"

    df["specimen_id"] = df["specimen_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()
    df["route_family"] = df["route_family"].astype(str).str.strip()
    df["cycles_to_failure"] = pd.to_numeric(df["cycles_to_failure"], errors="coerce")
    df = df[np.isfinite(df["cycles_to_failure"]) & (df["cycles_to_failure"] > 0)].copy()
    df["log10_nf"] = np.log10(df["cycles_to_failure"])

    nf = df["cycles_to_failure"].to_numpy(dtype=float)

    # Validation
    route_counts = df.groupby("route_id")["specimen_id"].nunique()
    validation_rows = [
        {"check": "Total sample rows", "expected": EXPECTED_SAMPLES, "observed": int(len(df)), "status": "PASS" if len(df) == EXPECTED_SAMPLES else "FAIL"},
        {"check": "Unique specimen_id count", "expected": EXPECTED_SAMPLES, "observed": int(df["specimen_id"].nunique()), "status": "PASS" if df["specimen_id"].nunique() == EXPECTED_SAMPLES else "FAIL"},
        {"check": "Unique route_id count", "expected": EXPECTED_ROUTES, "observed": int(df["route_id"].nunique()), "status": "PASS" if df["route_id"].nunique() == EXPECTED_ROUTES else "FAIL"},
        {"check": "Routes with exactly five samples", "expected": EXPECTED_ROUTES, "observed": int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()), "status": "PASS" if int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES else "FAIL"},
        {"check": "Duplicate specimen_id rows", "expected": 0, "observed": int(df.duplicated("specimen_id").sum()), "status": "PASS" if int(df.duplicated("specimen_id").sum()) == 0 else "FAIL"},
        {"check": "Non-positive fatigue life", "expected": 0, "observed": int((df["cycles_to_failure"] <= 0).sum()), "status": "PASS" if int((df["cycles_to_failure"] <= 0).sum()) == 0 else "FAIL"},
    ]
    validation_df = pd.DataFrame(validation_rows)
    validation_path = STATS_OUTPUT_DIR / "task3_5B_weibull_validation_overview_85.csv"
    validation_df.to_csv(validation_path, index=False)

    # Global Weibull
    beta, eta = fit_weibull_2p(nf)
    global_params = pd.DataFrame([{
        "analysis_level": "global",
        "n_samples": int(len(nf)),
        "nf_min": float(np.min(nf)),
        "nf_max": float(np.max(nf)),
        "nf_mean": float(np.mean(nf)),
        "nf_sd": float(np.std(nf, ddof=1)),
        "nf_cov": safe_cv(nf),
        "shape_beta": beta,
        "scale_eta_cycles": eta,
        "interpretation": "Global cross-route two-parameter Weibull descriptor",
    }])

    global_params_path = STATS_OUTPUT_DIR / "task3_5B_weibull_global_parameters_85.csv"
    global_params.to_csv(global_params_path, index=False)

    # Survival probabilities and quantiles
    surv_global = pd.DataFrame({
        "analysis_level": "global",
        "cycles_N": SURVIVAL_CYCLES.astype(int),
        "survival_probability_R": survival_probability(SURVIVAL_CYCLES, beta, eta),
        "failure_probability_F": 1.0 - survival_probability(SURVIVAL_CYCLES, beta, eta),
    })
    surv_path = STATS_OUTPUT_DIR / "task3_5B_weibull_survival_probabilities_85.csv"
    surv_global.to_csv(surv_path, index=False)

    q_global = pd.DataFrame({
        "analysis_level": "global",
        "failure_probability": FAILURE_PROBABILITIES,
        "life_cycles_np": weibull_quantile(FAILURE_PROBABILITIES, beta, eta),
    })
    q_global["quantile_label"] = q_global["failure_probability"].map(lambda p: f"B{int(round(p * 100))}")
    quant_path = STATS_OUTPUT_DIR / "task3_5B_weibull_life_quantiles_85.csv"
    q_global.to_csv(quant_path, index=False)

    # Bootstrap CIs for global Weibull parameters and quantiles
    boot_ci = bootstrap_weibull_ci(nf, BOOTSTRAP_B, RANDOM_SEED)
    boot_ci_path = STATS_OUTPUT_DIR / "task3_5B_weibull_bootstrap_parameter_CI_85.csv"
    boot_ci.to_csv(boot_ci_path, index=False)

    # Route-family Weibull
    family_rows = []
    family_surv_rows = []
    for family, g in df.groupby("route_family", sort=True):
        vals = g["cycles_to_failure"].to_numpy(dtype=float)
        fb, fe = fit_weibull_2p(vals)
        family_rows.append({
            "route_family": family,
            "n_samples": int(len(vals)),
            "nf_mean": float(np.mean(vals)),
            "nf_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "nf_cov": safe_cv(vals),
            "shape_beta": fb,
            "scale_eta_cycles": fe,
            "note": "Exploratory family-level Weibull fit",
        })
        for c, r in zip(SURVIVAL_CYCLES, survival_probability(SURVIVAL_CYCLES, fb, fe)):
            family_surv_rows.append({
                "route_family": family,
                "cycles_N": int(c),
                "survival_probability_R": float(r),
                "failure_probability_F": float(1 - r) if np.isfinite(r) else np.nan,
            })

    family_params = pd.DataFrame(family_rows).sort_values("nf_mean", ascending=False)
    family_surv = pd.DataFrame(family_surv_rows)
    family_params_path = STATS_OUTPUT_DIR / "task3_5B_route_family_weibull_parameters_85.csv"
    family_surv_path = STATS_OUTPUT_DIR / "task3_5B_route_family_survival_probabilities_85.csv"
    family_params.to_csv(family_params_path, index=False)
    family_surv.to_csv(family_surv_path, index=False)

    # Route-wise Weibull parameters and quantiles
    route_rows = []
    route_quant_rows = []
    for route, g in df.groupby("route_id", sort=True):
        vals = g["cycles_to_failure"].to_numpy(dtype=float)
        rb, re = fit_weibull_2p(vals)
        route_rows.append({
            "route_id": route,
            "route_family": str(g["route_family"].iloc[0]),
            "n_samples": int(len(vals)),
            "nf_mean": float(np.mean(vals)),
            "nf_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "nf_cov": safe_cv(vals),
            "shape_beta": rb,
            "scale_eta_cycles": re,
            "note": "Exploratory route-wise Weibull fit based on n=5 samples",
        })
        q_vals = weibull_quantile(FAILURE_PROBABILITIES, rb, re)
        for p, q in zip(FAILURE_PROBABILITIES, q_vals):
            route_quant_rows.append({
                "route_id": route,
                "route_family": str(g["route_family"].iloc[0]),
                "failure_probability": float(p),
                "quantile_label": f"B{int(round(p * 100))}",
                "life_cycles_np": float(q),
                "n_samples": int(len(vals)),
            })

    route_params = pd.DataFrame(route_rows).sort_values("nf_mean", ascending=False)
    route_quant = pd.DataFrame(route_quant_rows)
    route_params_path = STATS_OUTPUT_DIR / "task3_5B_routewise_weibull_parameters_85.csv"
    route_quant_path = STATS_OUTPUT_DIR / "task3_5B_routewise_life_quantiles_85.csv"
    route_params.to_csv(route_params_path, index=False)
    route_quant.to_csv(route_quant_path, index=False)

    # Figures
    fig_global_curve = FIGURES_DIR / "task3_5B_global_weibull_reliability_curve_85.png"
    fig_global_prob = FIGURES_DIR / "task3_5B_global_weibull_probability_plot_85.png"
    fig_family_curve = FIGURES_DIR / "task3_5B_route_family_weibull_reliability_curves_85.png"
    fig_route_quant = FIGURES_DIR / "task3_5B_routewise_b10_median_b90_85.png"

    plot_global_reliability(nf, beta, eta, fig_global_curve)
    plot_weibull_probability(nf, beta, eta, fig_global_prob)
    plot_route_family_reliability(family_params, float(np.max(nf)), fig_family_curve)
    plot_routewise_quantiles(route_quant, fig_route_quant)

    # Console preview
    print("\nGlobal Weibull parameters")
    print(global_params.to_string(index=False))
    print("\nGlobal survival probabilities")
    print(surv_global.to_string(index=False))
    print("\nGlobal life quantiles")
    print(q_global.to_string(index=False))
    print("\nRoute-family Weibull parameters")
    print(family_params.to_string(index=False))
    print("\nTop route-wise Weibull parameters by mean Nf")
    print(route_params.head(10).to_string(index=False))
    print("\nValidation overview")
    print(validation_df.to_string(index=False))

    final_status = "PASS" if (validation_df["status"] == "PASS").all() else "CHECK"

    # Report
    report_path = STATS_OUTPUT_DIR / "task3_5B_weibull_reliability_report_85.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("task3.5B_weibull_reliability_85 report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        f.write(f"Bootstrap resamples for global Weibull CI: {BOOTSTRAP_B}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("Validation overview\n")
        f.write("-" * 70 + "\n")
        f.write(validation_df.to_string(index=False))
        f.write("\n\n")

        f.write("Global Weibull parameters\n")
        f.write("-" * 70 + "\n")
        f.write(global_params.to_string(index=False))
        f.write("\n\n")

        f.write("Global survival probabilities\n")
        f.write("-" * 70 + "\n")
        f.write(surv_global.to_string(index=False))
        f.write("\n\n")

        f.write("Global Weibull life quantiles\n")
        f.write("-" * 70 + "\n")
        f.write(q_global.to_string(index=False))
        f.write("\n\n")

        f.write("Bootstrap CI for global Weibull parameters and quantiles\n")
        f.write("-" * 70 + "\n")
        f.write(boot_ci.to_string(index=False))
        f.write("\n\n")

        f.write("Route-family Weibull parameters\n")
        f.write("-" * 70 + "\n")
        f.write(family_params.to_string(index=False))
        f.write("\n\n")

        f.write("Top route-wise Weibull parameters by mean Nf\n")
        f.write("-" * 70 + "\n")
        f.write(route_params.head(17).to_string(index=False))
        f.write("\n\n")

        f.write("Generated files\n")
        f.write("-" * 70 + "\n")
        for p in [
            global_params_path, surv_path, quant_path, boot_ci_path,
            family_params_path, family_surv_path, route_params_path, route_quant_path,
            validation_path, fig_global_curve, fig_global_prob, fig_family_curve, fig_route_quant,
        ]:
            f.write(f"  {p}\n")
        f.write("\n")

        f.write("Interpretation note:\n")
        f.write("The global Weibull fit represents the fatigue-life distribution across the expanded processing-route sample set. "
                "Route-family and route-wise Weibull fits provide reliability metadata, but route-wise results are based on "
                "five samples per route and should be interpreted cautiously.\n\n")
        f.write(f"Final status: {final_status}\n")

    print(f"\nSaved report to: {report_path}")
    print(f"✅ Done task3.5B_weibull_reliability_85. Status: {final_status}")


if __name__ == "__main__":
    main()
