"""
task2.2_Hall_Petch_Validation_85.py

Purpose
-------
Validate the Structure -> Property relationship in the 85-sample fatigue dataset
using the Hall-Petch form:

    YS_MPa = sigma0 + k * d^(-1/2)

where d is grain size in micrometres and d^(-1/2) has units µm^(-1/2).

This script produces both sample-level and route-mean Hall-Petch regressions.
For journal interpretation, the route-mean regression is generally the more
conservative descriptor because the 85-row dataset contains five samples for
each processing route.

Expected input
--------------
Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Expected outputs
----------------
Fatigue_85_augmented_dataset/04_statistics_outputs/
    task2_2_hall_petch_report_85.txt
    task2_2_hall_petch_regression_summary_85.csv
    task2_2_hall_petch_model_comparison_85.csv
    task2_2_hall_petch_sample_data_85.csv
    task2_2_hall_petch_route_mean_data_85.csv
    task2_2_hall_petch_curve_sample_level_85.csv
    task2_2_hall_petch_curve_route_mean_85.csv
    task2_2_hall_petch_validation_overview_85.csv

Fatigue_85_augmented_dataset/06_figures/
    task2_2_hall_petch_sample_level_85.png
    task2_2_hall_petch_route_mean_85.png
    task2_2_hall_petch_residuals_route_mean_85.png
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

    # Return preferred path even if it does not exist, so the error message is clear.
    return candidates[0]


DATASET_DIR = resolve_dataset_dir()
CLEANED_DIR = DATASET_DIR / "02_cleaned"
STATS_DIR = DATASET_DIR / "04_statistics_outputs"
FIG_DIR = DATASET_DIR / "06_figures"
INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and map common variants to current workflow names."""
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
        "ys": "ys_mpa",
        "ys_mpa": "ys_mpa",
        "yield_strength_mpa": "ys_mpa",
        "yield_strength": "ys_mpa",
        "grain_size": "grain_size_um",
        "grain_size_µm": "grain_size_um",
        "grain_size_micrometre": "grain_size_um",
        "grain_size_micrometer": "grain_size_um",
        "d^_1_2": "d_inv_sqrt",
        "d_inv_sqrt_um": "d_inv_sqrt",
        "route": "route_id",
    }
    return df.rename(columns=rename_map)


def require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\nAvailable columns: {df.columns.tolist()}"
        )


def clean_input(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Keep required columns, enforce numeric values, and compute d_inv_sqrt."""
    required = ["specimen_id", "route_id", "route_family", "ys_mpa", "grain_size_um"]
    require_columns(df, required)

    keep_cols = [
        "specimen_id",
        "route_id",
        "route_family",
        "process_subtype",
        "soak_hours",
        "ecap_angle_deg",
        "ys_mpa",
        "uts_mpa",
        "hardness_hv",
        "grain_size_um",
        "d_inv_sqrt",
        "cycles_to_failure",
        "log10_nf",
        "log_nf_db",
        "is_synthetic",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    out = df[keep_cols].copy()

    out["specimen_id"] = out["specimen_id"].astype(str).str.strip()
    out["route_id"] = out["route_id"].astype(str).str.strip()
    out["route_family"] = out["route_family"].astype(str).str.strip()

    numeric_cols = [
        col for col in [
            "ys_mpa",
            "uts_mpa",
            "hardness_hv",
            "grain_size_um",
            "d_inv_sqrt",
            "cycles_to_failure",
            "log10_nf",
            "log_nf_db",
            "soak_hours",
            "ecap_angle_deg",
        ] if col in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if (out["grain_size_um"] <= 0).any():
        bad = out.loc[out["grain_size_um"] <= 0, ["specimen_id", "route_id", "grain_size_um"]]
        raise ValueError(f"Non-positive grain size values found:\n{bad.to_string(index=False)}")

    out["d_inv_sqrt_calc"] = 1.0 / np.sqrt(out["grain_size_um"])

    diagnostics: Dict[str, float] = {}
    if "d_inv_sqrt" in out.columns:
        diff = (out["d_inv_sqrt"] - out["d_inv_sqrt_calc"]).abs()
        diagnostics["max_abs_existing_vs_calculated_d_inv_sqrt_diff"] = float(diff.max(skipna=True))
    else:
        diagnostics["max_abs_existing_vs_calculated_d_inv_sqrt_diff"] = np.nan

    # Use calculated value for regression to keep the relationship physically traceable.
    out["d_inv_sqrt"] = out["d_inv_sqrt_calc"]

    out = out.dropna(subset=["specimen_id", "route_id", "route_family", "ys_mpa", "grain_size_um", "d_inv_sqrt"]).copy()
    return out, diagnostics


def fit_hall_petch(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit YS = sigma0 + k*d^-1/2 and return summary, fitted data, curve."""
    work = df[["d_inv_sqrt", "ys_mpa"]].dropna().copy()
    x = work["d_inv_sqrt"].to_numpy(dtype=float)
    y = work["ys_mpa"].to_numpy(dtype=float)

    if len(work) < 3:
        raise ValueError(f"Need at least 3 observations for Hall-Petch regression. {label} has {len(work)}.")

    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(r2_score(y, pred))
    rmse = float(math.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))

    lin = stats.linregress(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    alpha = 0.05
    dof = len(work) - 2
    tcrit = stats.t.ppf(1 - alpha / 2, dof) if dof > 0 else np.nan
    slope_ci_low = slope - tcrit * lin.stderr if dof > 0 else np.nan
    slope_ci_high = slope + tcrit * lin.stderr if dof > 0 else np.nan
    intercept_stderr = getattr(lin, "intercept_stderr", np.nan)
    intercept_ci_low = intercept - tcrit * intercept_stderr if dof > 0 and not np.isnan(intercept_stderr) else np.nan
    intercept_ci_high = intercept + tcrit * intercept_stderr if dof > 0 and not np.isnan(intercept_stderr) else np.nan

    summary = pd.DataFrame([{
        "analysis_level": label,
        "n_observations": len(work),
        "hall_petch_slope_k_mpa_um_half": slope,
        "sigma0_intercept_mpa": intercept,
        "r2": r2,
        "rmse_mpa": rmse,
        "mae_mpa": mae,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "slope_ci_2p5": float(slope_ci_low),
        "slope_ci_97p5": float(slope_ci_high),
        "intercept_ci_2p5": float(intercept_ci_low),
        "intercept_ci_97p5": float(intercept_ci_high),
    }])

    fitted = df.copy()
    fitted["ys_predicted_mpa"] = model.predict(fitted[["d_inv_sqrt"]])
    fitted["ys_residual_mpa"] = fitted["ys_mpa"] - fitted["ys_predicted_mpa"]

    x_range = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_range = model.predict(x_range.reshape(-1, 1))
    curve = pd.DataFrame({
        "analysis_level": label,
        "d_inv_sqrt": x_range,
        "ys_predicted_mpa": y_range,
    })

    return summary, fitted, curve


def make_route_mean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sample data to route level for conservative Hall-Petch interpretation."""
    agg_dict = {
        "route_family": ("route_family", "first"),
        "n_samples": ("specimen_id", "nunique"),
        "ys_mpa": ("ys_mpa", "mean"),
        "ys_mpa_std": ("ys_mpa", "std"),
        "grain_size_um": ("grain_size_um", "mean"),
        "grain_size_um_std": ("grain_size_um", "std"),
        "d_inv_sqrt": ("d_inv_sqrt", "mean"),
        "d_inv_sqrt_std": ("d_inv_sqrt", "std"),
    }
    for optional in ["uts_mpa", "hardness_hv", "cycles_to_failure"]:
        if optional in df.columns:
            agg_dict[optional] = (optional, "mean")

    route_df = df.groupby("route_id", as_index=False).agg(**agg_dict)
    return route_df.sort_values("d_inv_sqrt").reset_index(drop=True)


def build_validation(df: pd.DataFrame, route_df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("route_id")["specimen_id"].nunique()
    validation = pd.DataFrame([
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
            "observed": int((counts == EXPECTED_SAMPLES_PER_ROUTE).sum()),
            "status": "PASS" if int((counts == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES else "FAIL",
        },
        {
            "check": "Missing YS values",
            "expected": 0,
            "observed": int(df["ys_mpa"].isna().sum()),
            "status": "PASS" if int(df["ys_mpa"].isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Missing grain size values",
            "expected": 0,
            "observed": int(df["grain_size_um"].isna().sum()),
            "status": "PASS" if int(df["grain_size_um"].isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Non-positive grain size values",
            "expected": 0,
            "observed": int((df["grain_size_um"] <= 0).sum()),
            "status": "PASS" if int((df["grain_size_um"] <= 0).sum()) == 0 else "FAIL",
        },
        {
            "check": "Route mean rows",
            "expected": EXPECTED_ROUTES,
            "observed": len(route_df),
            "status": "PASS" if len(route_df) == EXPECTED_ROUTES else "FAIL",
        },
    ])
    return validation


def plot_hall_petch_sample(sample_df: pd.DataFrame, curve_df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for family, sub in sample_df.groupby("route_family"):
        ax.scatter(sub["d_inv_sqrt"], sub["ys_mpa"], s=42, alpha=0.75, label=family)

    ax.plot(curve_df["d_inv_sqrt"], curve_df["ys_predicted_mpa"], linewidth=2, label="Hall-Petch fit")
    r2 = summary.loc[0, "r2"]
    k = summary.loc[0, "hall_petch_slope_k_mpa_um_half"]
    sigma0 = summary.loc[0, "sigma0_intercept_mpa"]
    ax.text(
        0.04,
        0.96,
        f"YS = {sigma0:.2f} + {k:.2f} d$^{{-1/2}}$\nR$^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    ax.set_xlabel(r"$d^{-1/2}$ ($\mu$m$^{-1/2}$)")
    ax.set_ylabel("Yield strength, YS (MPa)")
    ax.set_title("Sample-level Hall-Petch validation for 85-sample dataset")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Route family")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_hall_petch_route(route_df: pd.DataFrame, curve_df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for family, sub in route_df.groupby("route_family"):
        ax.scatter(sub["d_inv_sqrt"], sub["ys_mpa"], s=72, label=family)

    ax.plot(curve_df["d_inv_sqrt"], curve_df["ys_predicted_mpa"], linewidth=2, label="Hall-Petch fit")

    for _, row in route_df.iterrows():
        ax.annotate(
            row["route_id"],
            (row["d_inv_sqrt"], row["ys_mpa"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    r2 = summary.loc[0, "r2"]
    k = summary.loc[0, "hall_petch_slope_k_mpa_um_half"]
    sigma0 = summary.loc[0, "sigma0_intercept_mpa"]
    pearson_p = summary.loc[0, "pearson_p"]
    ax.text(
        0.04,
        0.96,
        f"YS = {sigma0:.2f} + {k:.2f} d$^{{-1/2}}$\nR$^2$ = {r2:.3f}, p = {pearson_p:.2e}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    ax.set_xlabel(r"$d^{-1/2}$ ($\mu$m$^{-1/2}$)")
    ax.set_ylabel("Yield strength, YS (MPa)")
    ax.set_title("Route-mean Hall-Petch validation for 85-sample dataset")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Route family")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_route_residuals(route_df: pd.DataFrame, path: Path) -> None:
    plot_df = route_df.sort_values("ys_predicted_mpa").copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axhline(0, linewidth=1)
    ax.scatter(plot_df["ys_predicted_mpa"], plot_df["ys_residual_mpa"], s=70)
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["route_id"],
            (row["ys_predicted_mpa"], row["ys_residual_mpa"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel("Predicted yield strength (MPa)")
    ax.set_ylabel("Residual, measured - predicted YS (MPa)")
    ax.set_title("Hall-Petch residuals using route-mean data")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_report(
    path: Path,
    input_file: Path,
    sample_df: pd.DataFrame,
    route_df: pd.DataFrame,
    validation: pd.DataFrame,
    sample_summary: pd.DataFrame,
    route_summary: pd.DataFrame,
    diagnostics: Dict[str, float],
    generated_files: list[Path],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("task2.2_Hall_Petch_Validation_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Rows used after cleaning: {len(sample_df)}\n")
        f.write(f"Unique samples: {sample_df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {sample_df['route_id'].nunique()}\n")
        f.write(f"Route families: {', '.join(sorted(sample_df['route_family'].dropna().unique()))}\n")
        if not math.isnan(diagnostics.get("max_abs_existing_vs_calculated_d_inv_sqrt_diff", np.nan)):
            f.write(
                "Max absolute difference between existing and calculated d_inv_sqrt: "
                f"{diagnostics['max_abs_existing_vs_calculated_d_inv_sqrt_diff']:.8e}\n"
            )
        f.write("\nValidation overview\n")
        f.write("-" * 78 + "\n")
        f.write(validation.to_string(index=False))
        f.write("\n\nSample-level Hall-Petch regression\n")
        f.write("-" * 78 + "\n")
        f.write(sample_summary.to_string(index=False))
        f.write("\n\nRoute-mean Hall-Petch regression\n")
        f.write("-" * 78 + "\n")
        f.write(route_summary.to_string(index=False))
        f.write("\n\nRoute-mean data preview\n")
        f.write("-" * 78 + "\n")
        preview_cols = [
            "route_id",
            "route_family",
            "n_samples",
            "grain_size_um",
            "d_inv_sqrt",
            "ys_mpa",
            "ys_predicted_mpa",
            "ys_residual_mpa",
        ]
        f.write(route_df[preview_cols].to_string(index=False))
        f.write("\n\nGenerated files\n")
        f.write("-" * 78 + "\n")
        for item in generated_files:
            f.write(f"  {item}\n")
        f.write("\nInterpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "The sample-level regression uses all 85 rows, whereas the route-mean regression "
            "uses 17 processing-route means. For manuscript interpretation, the route-mean "
            "regression should be treated as the primary Hall-Petch validation because it avoids "
            "over-emphasising the five repeated samples per route. A moderate R2 is acceptable in "
            "Al 6063 because yield strength is influenced not only by grain refinement but also by "
            "precipitation state, dislocation density, residual stresses and processing history.\n"
        )
        final_status = "PASS" if (validation["status"] == "PASS").all() else "FAIL"
        f.write(f"\nFinal status: {final_status}\n")


def main() -> None:
    print("=== START task2.2_Hall_Petch_Validation_85 ===")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    raw = pd.read_csv(INPUT_FILE)
    print(f"Loaded rows: {len(raw)}")
    print("Raw columns:", raw.columns.tolist())

    df = normalize_columns(raw)
    sample_df, diagnostics = clean_input(df)
    route_df = make_route_mean_data(sample_df)

    validation = build_validation(sample_df, route_df)
    print("\nValidation overview:")
    print(validation)

    sample_summary, sample_fitted, sample_curve = fit_hall_petch(sample_df, "sample_level_85")
    route_summary, route_fitted, route_curve = fit_hall_petch(route_df, "route_mean_17")

    # Model comparison is a compact table for reporting.
    model_comparison = pd.concat([sample_summary, route_summary], ignore_index=True)

    # Save CSV outputs.
    out_report = STATS_DIR / "task2_2_hall_petch_report_85.txt"
    out_reg_summary = STATS_DIR / "task2_2_hall_petch_regression_summary_85.csv"
    out_model_comp = STATS_DIR / "task2_2_hall_petch_model_comparison_85.csv"
    out_sample_data = STATS_DIR / "task2_2_hall_petch_sample_data_85.csv"
    out_route_data = STATS_DIR / "task2_2_hall_petch_route_mean_data_85.csv"
    out_sample_curve = STATS_DIR / "task2_2_hall_petch_curve_sample_level_85.csv"
    out_route_curve = STATS_DIR / "task2_2_hall_petch_curve_route_mean_85.csv"
    out_validation = STATS_DIR / "task2_2_hall_petch_validation_overview_85.csv"

    out_fig_sample = FIG_DIR / "task2_2_hall_petch_sample_level_85.png"
    out_fig_route = FIG_DIR / "task2_2_hall_petch_route_mean_85.png"
    out_fig_resid = FIG_DIR / "task2_2_hall_petch_residuals_route_mean_85.png"

    model_comparison.to_csv(out_reg_summary, index=False)
    model_comparison.to_csv(out_model_comp, index=False)
    sample_fitted.to_csv(out_sample_data, index=False)
    route_fitted.to_csv(out_route_data, index=False)
    sample_curve.to_csv(out_sample_curve, index=False)
    route_curve.to_csv(out_route_curve, index=False)
    validation.to_csv(out_validation, index=False)

    plot_hall_petch_sample(sample_fitted, sample_curve, sample_summary, out_fig_sample)
    plot_hall_petch_route(route_fitted, route_curve, route_summary, out_fig_route)
    plot_route_residuals(route_fitted, out_fig_resid)

    generated_files = [
        out_reg_summary,
        out_model_comp,
        out_sample_data,
        out_route_data,
        out_sample_curve,
        out_route_curve,
        out_validation,
        out_fig_sample,
        out_fig_route,
        out_fig_resid,
    ]

    write_report(
        out_report,
        INPUT_FILE,
        sample_fitted,
        route_fitted,
        validation,
        sample_summary,
        route_summary,
        diagnostics,
        generated_files,
    )

    print("\nSample-level regression:")
    print(sample_summary.to_string(index=False))
    print("\nRoute-mean regression:")
    print(route_summary.to_string(index=False))
    print(f"\nSaved report to: {out_report}")

    final_status = "PASS" if (validation["status"] == "PASS").all() else "FAIL"
    print(f"\n✅ Done task2.2_Hall_Petch_Validation_85. Status: {final_status}")


if __name__ == "__main__":
    main()

