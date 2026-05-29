"""
task2.3_grain_sizeVsfatigue_life_85.py

Purpose
-------
Validate the Structure -> Performance relationship in the 85-sample fatigue
workflow by correlating grain size / inverse square-root grain size with fatigue
life, expressed as log10(Nf).

This script produces both sample-level and route-mean analyses:

    grain_size_um  -> log10(Nf)     expected negative relationship
    d_inv_sqrt     -> log10(Nf)     expected positive relationship

For manuscript interpretation, the route-mean analysis is generally the more
conservative descriptor because the dataset contains five samples per
processing route.

Expected input
--------------
Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Expected outputs
----------------
Fatigue_85_augmented_dataset/04_statistics_outputs/
    task2_3_grain_size_vs_fatigue_life_report_85.txt
    task2_3_grain_fatigue_correlation_summary_85.csv
    task2_3_grain_fatigue_model_comparison_85.csv
    task2_3_sample_level_grain_fatigue_data_85.csv
    task2_3_route_mean_grain_fatigue_data_85.csv
    task2_3_grain_size_vs_log10nf_curve_sample_level_85.csv
    task2_3_grain_size_vs_log10nf_curve_route_mean_85.csv
    task2_3_d_inv_sqrt_vs_log10nf_curve_sample_level_85.csv
    task2_3_d_inv_sqrt_vs_log10nf_curve_route_mean_85.csv
    task2_3_grain_fatigue_validation_overview_85.csv

Fatigue_85_augmented_dataset/06_figures/
    task2_3_grain_size_vs_log10nf_sample_level_85.png
    task2_3_grain_size_vs_log10nf_route_mean_85.png
    task2_3_d_inv_sqrt_vs_log10nf_route_mean_85.png
    task2_3_grain_fatigue_residuals_route_mean_85.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5
BOOTSTRAP_B = 5000
RANDOM_SEED = 42


def resolve_dataset_dir() -> Path:
    """Resolve dataset directory from the db_scripts_85 location or fallback path."""
    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir.parent / "Fatigue_85_augmented_dataset",
        script_dir / "Fatigue_85_augmented_dataset",
        Path(r"data"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


DATASET_DIR = resolve_dataset_dir()
CLEANED_DIR = DATASET_DIR / "02_cleaned"
STATS_DIR = DATASET_DIR / "04_statistics_outputs"
FIG_DIR = DATASET_DIR / "06_figures"
INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and map common variants to the current workflow names."""
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
        "grain_size": "grain_size_um",
        "grain_size_µm": "grain_size_um",
        "grain_size_micrometre": "grain_size_um",
        "grain_size_micrometer": "grain_size_um",
        "d^_1_2": "d_inv_sqrt",
        "d_inv_sqrt_um": "d_inv_sqrt",
        "nf": "cycles_to_failure",
        "n_f": "cycles_to_failure",
        "lognf": "log10_nf",
        "log_nf": "log10_nf",
        "log10nf": "log10_nf",
        "log10_nf": "log10_nf",
        "route": "route_id",
    }
    return df.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {df.columns.tolist()}")


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    route_sample_counts = df.groupby("route_id")["specimen_id"].nunique() if "route_id" in df.columns else pd.Series(dtype=int)

    checks = [
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
            "check": "Missing grain size values",
            "expected": 0,
            "observed": int(df["grain_size_um"].isna().sum()),
            "status": "PASS" if int(df["grain_size_um"].isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Missing fatigue-life values",
            "expected": 0,
            "observed": int(df["cycles_to_failure"].isna().sum()),
            "status": "PASS" if int(df["cycles_to_failure"].isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Missing log10_nf values",
            "expected": 0,
            "observed": int(df["log10_nf"].isna().sum()),
            "status": "PASS" if int(df["log10_nf"].isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Non-positive grain size values",
            "expected": 0,
            "observed": int((df["grain_size_um"] <= 0).sum()),
            "status": "PASS" if int((df["grain_size_um"] <= 0).sum()) == 0 else "FAIL",
        },
        {
            "check": "Non-positive fatigue life values",
            "expected": 0,
            "observed": int((df["cycles_to_failure"] <= 0).sum()),
            "status": "PASS" if int((df["cycles_to_failure"] <= 0).sum()) == 0 else "FAIL",
        },
        {
            "check": "Route mean rows",
            "expected": EXPECTED_ROUTES,
            "observed": int(route_sample_counts.shape[0]),
            "status": "PASS" if int(route_sample_counts.shape[0]) == EXPECTED_ROUTES else "FAIL",
        },
    ]
    return pd.DataFrame(checks)


def regression_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "n_observations": len(x),
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "slope_ci_2p5": np.nan,
            "slope_ci_97p5": np.nan,
            "intercept_ci_2p5": np.nan,
            "intercept_ci_97p5": np.nan,
        }

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    lr = stats.linregress(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    rng = np.random.default_rng(RANDOM_SEED)
    boot_slopes = []
    boot_intercepts = []
    n = len(x)
    for _ in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        # Avoid fully repeated x values causing degenerate regression.
        if np.unique(x[idx]).size < 2:
            continue
        m = LinearRegression()
        m.fit(x[idx].reshape(-1, 1), y[idx])
        boot_slopes.append(float(m.coef_[0]))
        boot_intercepts.append(float(m.intercept_))

    if boot_slopes:
        slope_ci_2p5, slope_ci_97p5 = np.percentile(boot_slopes, [2.5, 97.5])
        intercept_ci_2p5, intercept_ci_97p5 = np.percentile(boot_intercepts, [2.5, 97.5])
    else:
        slope_ci_2p5 = slope_ci_97p5 = intercept_ci_2p5 = intercept_ci_97p5 = np.nan

    return {
        "n_observations": int(len(x)),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "slope_ci_2p5": float(slope_ci_2p5),
        "slope_ci_97p5": float(slope_ci_97p5),
        "intercept_ci_2p5": float(intercept_ci_2p5),
        "intercept_ci_97p5": float(intercept_ci_97p5),
    }


def fit_curve(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str = "log10_nf") -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    x_range = np.linspace(x.min(), x.max(), 200)
    y_pred = model.predict(x_range.reshape(-1, 1))
    return pd.DataFrame({x_label: x_range, f"predicted_{y_label}": y_pred})


def add_predictions(df: pd.DataFrame, x_col: str, y_col: str, pred_col: str, residual_col: str) -> pd.DataFrame:
    out = df.copy()
    model = LinearRegression().fit(out[[x_col]], out[y_col])
    out[pred_col] = model.predict(out[[x_col]])
    out[residual_col] = out[y_col] - out[pred_col]
    return out


def route_mean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["route_id"]
    agg_dict = {
        "specimen_id": "nunique",
        "grain_size_um": ["mean", "std"],
        "d_inv_sqrt": ["mean", "std"],
        "cycles_to_failure": ["mean", "std"],
        "log10_nf": ["mean", "std"],
    }
    if "route_family" in df.columns:
        agg_dict["route_family"] = "first"
    if "process_subtype" in df.columns:
        agg_dict["process_subtype"] = "first"

    route = df.groupby(group_cols).agg(agg_dict)
    route.columns = ["_".join([str(c) for c in col if c]) for col in route.columns.to_flat_index()]
    route = route.reset_index()

    rename_map = {
        "specimen_id_nunique": "n_samples",
        "route_family_first": "route_family",
        "process_subtype_first": "process_subtype",
        "grain_size_um_mean": "grain_size_um",
        "grain_size_um_std": "grain_size_um_std",
        "d_inv_sqrt_mean": "d_inv_sqrt",
        "d_inv_sqrt_std": "d_inv_sqrt_std",
        "cycles_to_failure_mean": "cycles_to_failure_mean",
        "cycles_to_failure_std": "cycles_to_failure_std",
        "log10_nf_mean": "log10_nf",
        "log10_nf_std": "log10_nf_std",
    }
    route = route.rename(columns=rename_map)

    # Use d^-1/2 recalculated from route-mean grain size for clearer physical interpretation.
    route["d_inv_sqrt_from_mean_grain"] = 1.0 / np.sqrt(route["grain_size_um"])
    return route


def plot_scatter_with_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    annotate_routes: bool = False,
    invert_x: bool = False,
    use_family_colors: bool = True,
    text_box: str | None = None,
) -> None:
    plt.figure(figsize=(9, 6))

    if use_family_colors and "route_family" in df.columns:
        families = sorted(df["route_family"].dropna().unique())
        for fam in families:
            sub = df[df["route_family"] == fam]
            plt.scatter(sub[x_col], sub[y_col], s=70, label=fam)
    else:
        plt.scatter(df[x_col], df[y_col], s=60)

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    x_range = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_pred = model.predict(x_range.reshape(-1, 1))
    plt.plot(x_range, y_pred, linewidth=2, label="Linear fit")

    if annotate_routes and "route_id" in df.columns:
        for _, row in df.iterrows():
            plt.annotate(str(row["route_id"]), (row[x_col], row[y_col]), xytext=(5, 4), textcoords="offset points", fontsize=8)

    if text_box:
        plt.text(0.03, 0.97, text_box, transform=plt.gca().transAxes, va="top", ha="left", bbox=dict(boxstyle="round", alpha=0.15))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if invert_x:
        plt.gca().invert_xaxis()
    if use_family_colors and "route_family" in df.columns:
        plt.legend(title="Route family")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_route_residuals(route_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.axhline(0, linewidth=1)
    plt.scatter(route_df["predicted_log10_nf_from_grain_size"], route_df["residual_log10_nf_from_grain_size"], s=70)
    for _, row in route_df.iterrows():
        plt.annotate(str(row["route_id"]), (row["predicted_log10_nf_from_grain_size"], row["residual_log10_nf_from_grain_size"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("Predicted log10(Nf)")
    plt.ylabel("Residual, measured - predicted log10(Nf)")
    plt.title("Route-mean residuals: grain size vs fatigue life")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    print("=== START task2.3_grain_sizeVsfatigue_life_85 ===")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = normalize_columns(pd.read_csv(INPUT_FILE))
    require_columns(df, ["specimen_id", "route_id", "grain_size_um", "cycles_to_failure"])

    df = safe_numeric(df, ["grain_size_um", "cycles_to_failure"])
    df = df[(df["grain_size_um"] > 0) & (df["cycles_to_failure"] > 0)].copy()

    if "log10_nf" not in df.columns:
        df["log10_nf"] = np.log10(df["cycles_to_failure"])
    else:
        df["log10_nf"] = pd.to_numeric(df["log10_nf"], errors="coerce")
        missing_log = df["log10_nf"].isna()
        df.loc[missing_log, "log10_nf"] = np.log10(df.loc[missing_log, "cycles_to_failure"])

    if "d_inv_sqrt" not in df.columns:
        df["d_inv_sqrt"] = 1.0 / np.sqrt(df["grain_size_um"])
    else:
        df["d_inv_sqrt"] = pd.to_numeric(df["d_inv_sqrt"], errors="coerce")
        missing_d = df["d_inv_sqrt"].isna()
        df.loc[missing_d, "d_inv_sqrt"] = 1.0 / np.sqrt(df.loc[missing_d, "grain_size_um"])

    df = df.dropna(subset=["specimen_id", "route_id", "grain_size_um", "d_inv_sqrt", "cycles_to_failure", "log10_nf"]).copy()

    validation = validate_dataset(df)
    validation_path = STATS_DIR / "task2_3_grain_fatigue_validation_overview_85.csv"
    validation.to_csv(validation_path, index=False)

    sample_cols = [
        "specimen_id", "route_id", "grain_size_um", "d_inv_sqrt", "cycles_to_failure", "log10_nf"
    ]
    for optional_col in ["route_family", "process_subtype", "ys_mpa", "hardness_hv"]:
        if optional_col in df.columns:
            sample_cols.insert(2, optional_col)

    sample_df = df[sample_cols].copy()
    sample_df = add_predictions(sample_df, "grain_size_um", "log10_nf", "predicted_log10_nf_from_grain_size", "residual_log10_nf_from_grain_size")
    sample_df = add_predictions(sample_df, "d_inv_sqrt", "log10_nf", "predicted_log10_nf_from_d_inv_sqrt", "residual_log10_nf_from_d_inv_sqrt")

    route_df = route_mean_dataframe(df)
    route_df = add_predictions(route_df, "grain_size_um", "log10_nf", "predicted_log10_nf_from_grain_size", "residual_log10_nf_from_grain_size")
    # Use d_inv_sqrt_from_mean_grain for route-mean physical interpretation.
    route_df = add_predictions(route_df, "d_inv_sqrt_from_mean_grain", "log10_nf", "predicted_log10_nf_from_d_inv_sqrt", "residual_log10_nf_from_d_inv_sqrt")

    # Model summaries.
    model_rows = []
    analyses = [
        ("sample_level_85", "grain_size_um", sample_df["grain_size_um"].to_numpy(), sample_df["log10_nf"].to_numpy()),
        ("sample_level_85", "d_inv_sqrt", sample_df["d_inv_sqrt"].to_numpy(), sample_df["log10_nf"].to_numpy()),
        ("route_mean_17", "grain_size_um", route_df["grain_size_um"].to_numpy(), route_df["log10_nf"].to_numpy()),
        ("route_mean_17", "d_inv_sqrt_from_mean_grain", route_df["d_inv_sqrt_from_mean_grain"].to_numpy(), route_df["log10_nf"].to_numpy()),
    ]

    for level, feature, x, y in analyses:
        metrics = regression_metrics(x, y)
        model_rows.append({
            "analysis_level": level,
            "feature": feature,
            "target": "log10_nf",
            **metrics,
        })

    summary_df = pd.DataFrame(model_rows)
    correlation_summary_path = STATS_DIR / "task2_3_grain_fatigue_correlation_summary_85.csv"
    summary_df.to_csv(correlation_summary_path, index=False)

    model_comparison = summary_df[[
        "analysis_level", "feature", "n_observations", "slope", "intercept", "r2", "rmse", "mae", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"
    ]].copy()
    model_comparison_path = STATS_DIR / "task2_3_grain_fatigue_model_comparison_85.csv"
    model_comparison.to_csv(model_comparison_path, index=False)

    sample_data_path = STATS_DIR / "task2_3_sample_level_grain_fatigue_data_85.csv"
    route_data_path = STATS_DIR / "task2_3_route_mean_grain_fatigue_data_85.csv"
    sample_df.to_csv(sample_data_path, index=False)
    route_df.to_csv(route_data_path, index=False)

    # Curves for Origin/manual plotting.
    curve_sample_grain = fit_curve(sample_df["grain_size_um"].to_numpy(), sample_df["log10_nf"].to_numpy(), "grain_size_um")
    curve_route_grain = fit_curve(route_df["grain_size_um"].to_numpy(), route_df["log10_nf"].to_numpy(), "grain_size_um")
    curve_sample_d = fit_curve(sample_df["d_inv_sqrt"].to_numpy(), sample_df["log10_nf"].to_numpy(), "d_inv_sqrt")
    curve_route_d = fit_curve(route_df["d_inv_sqrt_from_mean_grain"].to_numpy(), route_df["log10_nf"].to_numpy(), "d_inv_sqrt_from_mean_grain")

    curve_sample_grain_path = STATS_DIR / "task2_3_grain_size_vs_log10nf_curve_sample_level_85.csv"
    curve_route_grain_path = STATS_DIR / "task2_3_grain_size_vs_log10nf_curve_route_mean_85.csv"
    curve_sample_d_path = STATS_DIR / "task2_3_d_inv_sqrt_vs_log10nf_curve_sample_level_85.csv"
    curve_route_d_path = STATS_DIR / "task2_3_d_inv_sqrt_vs_log10nf_curve_route_mean_85.csv"
    curve_sample_grain.to_csv(curve_sample_grain_path, index=False)
    curve_route_grain.to_csv(curve_route_grain_path, index=False)
    curve_sample_d.to_csv(curve_sample_d_path, index=False)
    curve_route_d.to_csv(curve_route_d_path, index=False)

    # Figures.
    route_grain_metrics = summary_df[(summary_df["analysis_level"] == "route_mean_17") & (summary_df["feature"] == "grain_size_um")].iloc[0]
    route_d_metrics = summary_df[(summary_df["analysis_level"] == "route_mean_17") & (summary_df["feature"] == "d_inv_sqrt_from_mean_grain")].iloc[0]

    text_grain = (
        f"log10(Nf) = {route_grain_metrics['intercept']:.3f} "
        f"{route_grain_metrics['slope']:+.4f} grain size\n"
        f"R² = {route_grain_metrics['r2']:.3f}, p = {route_grain_metrics['pearson_p']:.2e}"
    )
    text_d = (
        f"log10(Nf) = {route_d_metrics['intercept']:.3f} "
        f"{route_d_metrics['slope']:+.3f} d⁻¹/²\n"
        f"R² = {route_d_metrics['r2']:.3f}, p = {route_d_metrics['pearson_p']:.2e}"
    )

    fig_sample_grain = FIG_DIR / "task2_3_grain_size_vs_log10nf_sample_level_85.png"
    fig_route_grain = FIG_DIR / "task2_3_grain_size_vs_log10nf_route_mean_85.png"
    fig_route_d = FIG_DIR / "task2_3_d_inv_sqrt_vs_log10nf_route_mean_85.png"
    fig_resid = FIG_DIR / "task2_3_grain_fatigue_residuals_route_mean_85.png"

    plot_scatter_with_fit(
        sample_df,
        "grain_size_um",
        "log10_nf",
        "Sample-level grain size vs fatigue life for 85-sample dataset",
        "Grain size (µm)",
        "log10(Nf)",
        fig_sample_grain,
        annotate_routes=False,
        invert_x=True,
        use_family_colors=True,
    )

    plot_scatter_with_fit(
        route_df,
        "grain_size_um",
        "log10_nf",
        "Route-mean grain size vs fatigue life for 85-sample dataset",
        "Mean grain size (µm)",
        "Mean log10(Nf)",
        fig_route_grain,
        annotate_routes=True,
        invert_x=True,
        use_family_colors=True,
        text_box=text_grain,
    )

    plot_scatter_with_fit(
        route_df,
        "d_inv_sqrt_from_mean_grain",
        "log10_nf",
        "Route-mean inverse square-root grain size vs fatigue life",
        r"$d^{-1/2}$ (µm$^{-1/2}$)",
        "Mean log10(Nf)",
        fig_route_d,
        annotate_routes=True,
        invert_x=False,
        use_family_colors=True,
        text_box=text_d,
    )

    plot_route_residuals(route_df, fig_resid)

    # Report.
    report_path = STATS_DIR / "task2_3_grain_size_vs_fatigue_life_report_85.txt"
    final_status = "PASS" if (validation["status"] == "PASS").all() else "CHECK_REQUIRED"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("task2.3_grain_sizeVsfatigue_life_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows used after cleaning: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        if "route_family" in df.columns:
            f.write(f"Route families: {', '.join(sorted(df['route_family'].dropna().unique()))}\n")
        f.write("\n")

        f.write("Validation overview\n")
        f.write("-" * 78 + "\n")
        f.write(validation.to_string(index=False))
        f.write("\n\n")

        f.write("Correlation and regression summary\n")
        f.write("-" * 78 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("Route-mean grain-fatigue data preview\n")
        f.write("-" * 78 + "\n")
        preview_cols = [
            "route_id", "route_family", "n_samples", "grain_size_um", "d_inv_sqrt_from_mean_grain", "cycles_to_failure_mean", "log10_nf", "predicted_log10_nf_from_grain_size", "residual_log10_nf_from_grain_size"
        ]
        preview_cols = [c for c in preview_cols if c in route_df.columns]
        f.write(route_df[preview_cols].sort_values("log10_nf", ascending=False).to_string(index=False))
        f.write("\n\n")

        f.write("Generated files\n")
        f.write("-" * 78 + "\n")
        for path in [
            correlation_summary_path,
            model_comparison_path,
            sample_data_path,
            route_data_path,
            curve_sample_grain_path,
            curve_route_grain_path,
            curve_sample_d_path,
            curve_route_d_path,
            validation_path,
            fig_sample_grain,
            fig_route_grain,
            fig_route_d,
            fig_resid,
        ]:
            f.write(f"  {path}\n")
        f.write("\n")

        f.write("Interpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "This script evaluates the Structure -> Performance relationship in the PSPP workflow. "
            "A negative correlation between grain size and log10(Nf), or a positive correlation between "
            "d^-1/2 and log10(Nf), supports the interpretation that grain refinement improves fatigue "
            "durability. For manuscript interpretation, the route-mean regression should be treated as "
            "the primary descriptor because it avoids over-emphasising the five repeated samples per route.\n"
        )
        f.write("\n")
        f.write(f"Final status: {final_status}\n")

    print("\nCorrelation and regression summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved report to: {report_path}")
    print(f"Saved route-mean data to: {route_data_path}")
    print(f"\n✅ Done task2.3_grain_sizeVsfatigue_life_85. Status: {final_status}")


if __name__ == "__main__":
    main()

