"""
task3.1_statistical_characterization_85.py

Purpose
-------
Statistically characterise the 85-sample fatigue dataset after cycle-level
aggregation. This script reads the ML-ready sample-level feature table and
produces descriptive statistics, route-wise/family-wise summaries, normality
checks, feature-variability rankings, and quick diagnostic figures.

Expected input
--------------
Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Expected outputs
----------------
Fatigue_85_augmented_dataset/04_statistics_outputs/
    task3_1_global_descriptive_stats_85.csv
    task3_1_routewise_descriptive_stats_85.csv
    task3_1_route_family_stats_85.csv
    task3_1_feature_variability_ranking_85.csv
    task3_1_normality_tests_85.csv
    task3_1_feature_target_correlation_snapshot_85.csv
    task3_1_validation_overview_85.csv
    task3_1_statistical_characterization_report_85.txt

Fatigue_85_augmented_dataset/06_figures/
    task3_1_hist_cycles_to_failure_85.png
    task3_1_hist_log10_nf_85.png
    task3_1_boxplot_route_family_log10_nf_85.png
    task3_1_feature_variability_top20_85.png
    task3_1_selected_feature_boxplots_85.png

How to run
----------
Place this file inside db_scripts_85 and run:
    python task3.1_statistical_characterization_85.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -------------------------------------------------------------------------
# Path configuration
# -------------------------------------------------------------------------
def resolve_project_paths() -> tuple[Path, Path, Path, Path]:
    """Resolve dataset folders assuming the script is placed in db_scripts_85."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_dir = project_root / "Fatigue_85_augmented_dataset"

    cleaned_dir = dataset_dir / "02_cleaned"
    stats_dir = dataset_dir / "04_statistics_outputs"
    figures_dir = dataset_dir / "06_figures"

    return dataset_dir, cleaned_dir, stats_dir, figures_dir


DATASET_DIR, CLEANED_DIR, STATS_DIR, FIGURES_DIR = resolve_project_paths()
INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"

EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for safer downstream processing."""
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
        "nf_cycles": "cycles_to_failure",
        "n_f": "cycles_to_failure",
        "nf": "cycles_to_failure",
        "cyclces_to_failure": "cycles_to_failure",
        "cyclces_to_fatilure": "cycles_to_failure",
        "log_nf": "log10_nf",
        "log10nf": "log10_nf",
        "log10_nf_db": "log10_nf",
        "ys": "ys_mpa",
        "yield_strength_mpa": "ys_mpa",
        "uts": "uts_mpa",
        "ultimate_tensile_strength_mpa": "uts_mpa",
        "hardness": "hardness_hv",
        "grain_size": "grain_size_um",
        "stress_amp_stable_mean": "stress_amplitude_stable_mean",
        "stress_amp_mean": "stress_amplitude_mean",
        "stress_amp_mpa": "stress_amplitude_mpa",
    }
    out = out.rename(columns=rename_map)

    # If duplicate normalized columns appear, keep the first non-null value row-wise.
    if out.columns.duplicated().any():
        repaired = pd.DataFrame(index=out.index)
        for col in pd.unique(out.columns):
            same = out.loc[:, out.columns == col]
            if same.shape[1] == 1:
                repaired[col] = same.iloc[:, 0]
            else:
                repaired[col] = same.bfill(axis=1).iloc[:, 0]
        out = repaired

    return out


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def coefficient_of_variation(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return np.nan
    mean_val = values.mean()
    if np.isclose(mean_val, 0.0):
        return np.nan
    return values.std(ddof=1) / abs(mean_val)


def build_descriptive_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        x_clean = x.dropna()
        row = {
            "feature": col,
            "n_total": len(x),
            "n_valid": int(x_clean.shape[0]),
            "n_missing": int(x.isna().sum()),
            "missing_percent": float(x.isna().mean() * 100),
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "cv_abs": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "max": np.nan,
            "iqr": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
        }
        if len(x_clean) > 0:
            row.update({
                "mean": float(x_clean.mean()),
                "median": float(x_clean.median()),
                "std": float(x_clean.std(ddof=1)) if len(x_clean) > 1 else 0.0,
                "cv_abs": coefficient_of_variation(x_clean),
                "min": float(x_clean.min()),
                "q25": float(x_clean.quantile(0.25)),
                "q75": float(x_clean.quantile(0.75)),
                "max": float(x_clean.max()),
                "iqr": float(x_clean.quantile(0.75) - x_clean.quantile(0.25)),
                "skewness": float(x_clean.skew()) if len(x_clean) > 2 else np.nan,
                "kurtosis": float(x_clean.kurt()) if len(x_clean) > 3 else np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def shapiro_test_for_features(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        row = {
            "feature": col,
            "n_valid": int(len(x)),
            "shapiro_w": np.nan,
            "shapiro_p": np.nan,
            "normality_decision_alpha_0_05": "not_tested",
            "note": "",
        }
        if len(x) < 3:
            row["note"] = "Shapiro-Wilk requires at least 3 valid values."
        elif len(x) > 5000:
            row["note"] = "Shapiro-Wilk not applied because n > 5000."
        elif np.isclose(x.std(ddof=1), 0.0):
            row["note"] = "Not tested because the feature is nearly constant."
        else:
            try:
                w, p = stats.shapiro(x)
                row["shapiro_w"] = float(w)
                row["shapiro_p"] = float(p)
                row["normality_decision_alpha_0_05"] = (
                    "no_evidence_against_normality" if p >= 0.05 else "non_normal"
                )
            except Exception as exc:  # pragma: no cover - diagnostic safety
                row["note"] = f"Shapiro-Wilk failed: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def flatten_columns(columns: pd.MultiIndex | pd.Index) -> list[str]:
    out = []
    for col in columns:
        if isinstance(col, tuple):
            clean = [str(part) for part in col if str(part) not in {"", "None"}]
            out.append("_".join(clean))
        else:
            out.append(str(col))
    return out


def grouped_summary(df: pd.DataFrame, group_col: str, features: list[str]) -> pd.DataFrame:
    available = [f for f in features if f in df.columns]
    grouped = df.groupby(group_col, dropna=False)[available].agg(["count", "mean", "std", "min", "median", "max"])
    grouped.columns = flatten_columns(grouped.columns)
    grouped = grouped.reset_index()

    # Add CoV columns for key features.
    for f in available:
        mean_col = f"{f}_mean"
        std_col = f"{f}_std"
        if mean_col in grouped.columns and std_col in grouped.columns:
            grouped[f"{f}_cv_abs"] = np.where(
                np.isclose(grouped[mean_col], 0.0),
                np.nan,
                grouped[std_col] / grouped[mean_col].abs(),
            )
    return grouped


def target_correlation_snapshot(df: pd.DataFrame, numeric_cols: list[str], target: str) -> pd.DataFrame:
    rows = []
    if target not in df.columns:
        return pd.DataFrame(rows)

    y = pd.to_numeric(df[target], errors="coerce")
    for col in numeric_cols:
        if col == target:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 3 or np.isclose(pair["x"].std(ddof=1), 0.0):
            pearson_r = pearson_p = spearman_r = spearman_p = np.nan
        else:
            pearson_r, pearson_p = stats.pearsonr(pair["x"], pair["y"])
            spearman_r, spearman_p = stats.spearmanr(pair["x"], pair["y"])
        rows.append({
            "feature": col,
            "target": target,
            "n_pairwise": int(len(pair)),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
            "spearman_p": spearman_p,
            "abs_pearson_r": abs(pearson_r) if pd.notna(pearson_r) else np.nan,
            "abs_spearman_rho": abs(spearman_r) if pd.notna(spearman_r) else np.nan,
        })
    corr = pd.DataFrame(rows)
    if not corr.empty:
        corr = corr.sort_values("abs_spearman_rho", ascending=False, na_position="last")
    return corr


def save_histogram(df: pd.DataFrame, col: str, path: Path, title: str, xlabel: str, bins: int = 12) -> None:
    if col not in df.columns:
        return
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if x.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Sample count")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_boxplot_by_group(df: pd.DataFrame, group_col: str, value_col: str, path: Path, title: str) -> None:
    if group_col not in df.columns or value_col not in df.columns:
        return
    sub = df[[group_col, value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return
    groups = [g for g in sorted(sub[group_col].astype(str).unique())]
    data = [sub.loc[sub[group_col].astype(str) == g, value_col].values for g in groups]
    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=groups, showmeans=True)
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_variability_bar(variability_df: pd.DataFrame, path: Path) -> None:
    if variability_df.empty or "cv_abs" not in variability_df.columns:
        return
    top = variability_df.dropna(subset=["cv_abs"]).head(20).copy()
    if top.empty:
        return
    plt.figure(figsize=(11, 6))
    plt.barh(top["feature"].iloc[::-1], top["cv_abs"].iloc[::-1])
    plt.title("Top 20 variable features by absolute coefficient of variation")
    plt.xlabel("Absolute CoV")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_selected_feature_boxplots(df: pd.DataFrame, path: Path) -> None:
    selected = [
        "ys_mpa", "uts_mpa", "hardness_hv", "grain_size_um",
        "psa_stable_mean", "mean_stress_stable_mean",
        "stress_amplitude_stable_mean", "unloading_modulus_stable_mean",
        "energy_proxy_stable_mean",
    ]
    selected = [c for c in selected if c in df.columns]
    if not selected:
        return
    data = []
    labels = []
    for col in selected:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if not x.empty:
            data.append(x.values)
            labels.append(col)
    if not data:
        return
    plt.figure(figsize=(13, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title("Distribution of selected PSPP and cyclic descriptors")
    plt.ylabel("Feature value")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=== START task3.1_statistical_characterization_85 ===")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    raw = pd.read_csv(INPUT_FILE)
    print(f"Input file: {INPUT_FILE}")
    print(f"Raw shape: {raw.shape}")

    df = normalize_columns(raw)
    print(f"Normalized shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    required = ["specimen_id", "route_id", "cycles_to_failure"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df["specimen_id"] = df["specimen_id"].astype(str).str.strip()
    df["route_id"] = df["route_id"].astype(str).str.strip()

    df["cycles_to_failure"] = safe_numeric(df["cycles_to_failure"])
    if "log10_nf" not in df.columns:
        df["log10_nf"] = np.log10(df["cycles_to_failure"])
    else:
        df["log10_nf"] = safe_numeric(df["log10_nf"])
        missing_log = df["log10_nf"].isna() & df["cycles_to_failure"].notna()
        df.loc[missing_log, "log10_nf"] = np.log10(df.loc[missing_log, "cycles_to_failure"])

    # Convert all likely numeric columns.
    id_like = {
        "specimen_id", "route_id", "route_family", "process_subtype", "source_seed_route",
        "source_sample_id", "generation_method", "source_group", "source_file",
    }
    for col in df.columns:
        if col not in id_like:
            converted = pd.to_numeric(df[col], errors="ignore")
            df[col] = converted

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Descriptive statistics for all numeric columns.
    descriptive = build_descriptive_stats(df, numeric_cols)
    descriptive_path = STATS_DIR / "task3_1_global_descriptive_stats_85.csv"
    descriptive.to_csv(descriptive_path, index=False)

    # Route-wise and route-family descriptive summaries.
    summary_features = [
        "cycles_to_failure", "log10_nf", "ys_mpa", "uts_mpa", "hardness_hv", "grain_size_um",
        "psa_stable_mean", "mean_stress_stable_mean", "stress_amplitude_stable_mean",
        "unloading_modulus_stable_mean", "energy_proxy_stable_mean",
    ]
    summary_features = [f for f in summary_features if f in df.columns]

    routewise = grouped_summary(df, "route_id", summary_features)
    routewise_path = STATS_DIR / "task3_1_routewise_descriptive_stats_85.csv"
    routewise.to_csv(routewise_path, index=False)

    if "route_family" in df.columns:
        df["route_family"] = df["route_family"].astype(str).str.strip()
        family_stats = grouped_summary(df, "route_family", summary_features)
    else:
        family_stats = pd.DataFrame()
    family_path = STATS_DIR / "task3_1_route_family_stats_85.csv"
    family_stats.to_csv(family_path, index=False)

    # Variability ranking.
    variability = descriptive.copy()
    variability["near_zero_mean_flag"] = variability["mean"].abs() < 1e-12
    variability = variability.sort_values("cv_abs", ascending=False, na_position="last")
    variability_path = STATS_DIR / "task3_1_feature_variability_ranking_85.csv"
    variability.to_csv(variability_path, index=False)

    # Normality checks for selected important variables.
    normality_features = [
        "cycles_to_failure", "log10_nf", "ys_mpa", "uts_mpa", "elongation_percent",
        "hardness_hv", "grain_size_um", "d_inv_sqrt", "strength_ratio",
        "fatigue_efficiency_lognf_per_mpa", "fatigue_efficiency",
        "psa_stable_mean", "psa_stable_std", "mean_stress_stable_mean",
        "stress_amplitude_stable_mean", "unloading_modulus_stable_mean",
        "energy_proxy_stable_mean",
    ]
    normality_features = [f for f in normality_features if f in df.columns]
    normality = shapiro_test_for_features(df, normality_features)
    normality_path = STATS_DIR / "task3_1_normality_tests_85.csv"
    normality.to_csv(normality_path, index=False)

    # Quick correlation snapshot with log10 fatigue life.
    corr = target_correlation_snapshot(df, numeric_cols, "log10_nf")
    corr_path = STATS_DIR / "task3_1_feature_target_correlation_snapshot_85.csv"
    corr.to_csv(corr_path, index=False)

    # Validation overview.
    route_counts = df.groupby("route_id")["specimen_id"].nunique()
    validations = pd.DataFrame([
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
        {
            "check": "Infinite numeric values",
            "expected": 0,
            "observed": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()),
            "status": "PASS" if int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()) == 0 else "FAIL",
        },
    ])
    validation_path = STATS_DIR / "task3_1_validation_overview_85.csv"
    validations.to_csv(validation_path, index=False)

    # Figures.
    save_histogram(
        df, "cycles_to_failure", FIGURES_DIR / "task3_1_hist_cycles_to_failure_85.png",
        "Histogram of fatigue life across 85 samples", "Cycles to failure, Nf", bins=12,
    )
    save_histogram(
        df, "log10_nf", FIGURES_DIR / "task3_1_hist_log10_nf_85.png",
        "Histogram of log10(Nf) across 85 samples", "log10(Nf)", bins=12,
    )
    if "route_family" in df.columns:
        save_boxplot_by_group(
            df, "route_family", "log10_nf", FIGURES_DIR / "task3_1_boxplot_route_family_log10_nf_85.png",
            "Route-family-wise log10(Nf) distribution",
        )
    save_variability_bar(variability, FIGURES_DIR / "task3_1_feature_variability_top20_85.png")
    save_selected_feature_boxplots(df, FIGURES_DIR / "task3_1_selected_feature_boxplots_85.png")

    # Report.
    nf = df["cycles_to_failure"].dropna()
    log_nf = df["log10_nf"].dropna()
    log_norm = normality.loc[normality["feature"] == "log10_nf"]
    log_shapiro_w = log_norm["shapiro_w"].iloc[0] if not log_norm.empty else np.nan
    log_shapiro_p = log_norm["shapiro_p"].iloc[0] if not log_norm.empty else np.nan

    top_var = variability.dropna(subset=["cv_abs"]).head(10)
    top_corr = corr.dropna(subset=["spearman_rho"]).head(10) if not corr.empty else pd.DataFrame()

    report_path = STATS_DIR / "task3_1_statistical_characterization_report_85.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("task3.1_statistical_characterization_85 report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        if "route_family" in df.columns:
            f.write(f"Route families: {', '.join(sorted(df['route_family'].astype(str).unique()))}\n")

        f.write("\nFatigue-life statistics\n")
        f.write("-" * 70 + "\n")
        f.write(f"Nf min: {nf.min():.4f}\n")
        f.write(f"Nf max: {nf.max():.4f}\n")
        f.write(f"Nf mean: {nf.mean():.4f}\n")
        f.write(f"Nf SD: {nf.std(ddof=1):.4f}\n")
        f.write(f"Nf CoV: {coefficient_of_variation(nf):.6f}\n")
        f.write(f"log10(Nf) mean: {log_nf.mean():.6f}\n")
        f.write(f"log10(Nf) SD: {log_nf.std(ddof=1):.6f}\n")
        f.write(f"Shapiro-Wilk W on log10(Nf): {log_shapiro_w:.6f}\n")
        f.write(f"Shapiro-Wilk p on log10(Nf): {log_shapiro_p:.6f}\n")

        f.write("\nRoute-wise fatigue-life summary preview\n")
        f.write("-" * 70 + "\n")
        if "cycles_to_failure_mean" in routewise.columns:
            route_preview_cols = [
                "route_id", "cycles_to_failure_count", "cycles_to_failure_mean",
                "cycles_to_failure_std", "cycles_to_failure_cv_abs", "log10_nf_mean",
            ]
            route_preview_cols = [c for c in route_preview_cols if c in routewise.columns]
            f.write(routewise[route_preview_cols].to_string(index=False))
            f.write("\n")

        if not family_stats.empty:
            f.write("\nRoute-family fatigue-life summary\n")
            f.write("-" * 70 + "\n")
            fam_preview_cols = [
                "route_family", "cycles_to_failure_count", "cycles_to_failure_mean",
                "cycles_to_failure_std", "cycles_to_failure_cv_abs", "log10_nf_mean",
            ]
            fam_preview_cols = [c for c in fam_preview_cols if c in family_stats.columns]
            f.write(family_stats[fam_preview_cols].to_string(index=False))
            f.write("\n")

        f.write("\nTop 10 variable numeric features by absolute CoV\n")
        f.write("-" * 70 + "\n")
        if not top_var.empty:
            f.write(top_var[["feature", "mean", "std", "cv_abs", "n_missing"]].to_string(index=False))
            f.write("\n")

        f.write("\nTop 10 features associated with log10(Nf) by Spearman ranking\n")
        f.write("-" * 70 + "\n")
        if not top_corr.empty:
            f.write(top_corr[["feature", "n_pairwise", "spearman_rho", "spearman_p", "pearson_r", "pearson_p"]].to_string(index=False))
            f.write("\n")

        f.write("\nValidation overview\n")
        f.write("-" * 70 + "\n")
        f.write(validations.to_string(index=False))
        f.write("\n\n")

        overall_status = "PASS" if (validations["status"] == "PASS").all() else "CHECK_REQUIRED"
        f.write(f"Final status: {overall_status}\n")
        f.write("\nInterpretation note:\n")
        f.write(
            "This script characterises the 85-row sample-level dataset. "
            "Cycle-level rows are not treated as independent ML samples. "
            "If log10(Nf) is non-normal, this should be interpreted in relation "
            "to processing-route clustering rather than as a data-loading failure.\n"
        )

    print("\nSaved outputs:")
    for path in [
        descriptive_path, routewise_path, family_path, variability_path,
        normality_path, corr_path, validation_path, report_path,
    ]:
        print(f"  {path}")

    print("\nValidation overview:")
    print(validations.to_string(index=False))

    final_status = "PASS" if (validations["status"] == "PASS").all() else "CHECK_REQUIRED"
    print(f"\n✅ Done task3.1_statistical_characterization_85. Status: {final_status}")


if __name__ == "__main__":
    main()
