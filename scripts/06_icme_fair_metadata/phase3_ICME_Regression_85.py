"""
phase3_ICME_Regression_85.py
================================

Purpose
-------
Create a manuscript-ready ICME/PSPP integration layer for the 85-sample fatigue
workflow. The script consolidates route-wise PSPP descriptors, forward and inverse
ICME screening summaries, and interpretable regression validation using the
sample-level feature matrix.

Input
-----
Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Outputs
-------
04_statistics_outputs:
    phase3_ICME_routewise_pspp_matrix_85.csv
    phase3_ICME_route_family_pspp_summary_85.csv
    phase3_ICME_forward_mapping_summary_85.csv
    phase3_ICME_inverse_screening_summary_85.csv
    phase3_ICME_evidence_chain_85.csv
    phase3_ICME_feature_layer_mapping_85.csv
    phase3_ICME_validation_overview_85.csv
    phase3_ICME_report_85.txt
    phase3_ICME_metadata_85.json

05_ml_outputs:
    phase3_ICME_regression_model_summary_85.csv
    phase3_ICME_best_model_loro_predictions_85.csv
    phase3_ICME_best_model_routewise_error_85.csv
    phase3_ICME_best_model_coefficients_85.csv

06_figures:
    phase3_ICME_routewise_fatigue_ranking_85.png
    phase3_ICME_route_family_summary_85.png
    phase3_ICME_structure_performance_map_85.png
    phase3_ICME_best_model_loro_predicted_vs_actual_85.png
    phase3_ICME_best_model_coefficients_85.png
    phase3_ICME_routewise_prediction_error_85.png

Author note
-----------
This script intentionally uses sample-level rows for model validation and
route-mean rows for manuscript-level ICME interpretation. Route-aware validation
is used to avoid artificially optimistic ML results from the five samples per route.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r".")
DATASET_DIR = BASE_DIR / "Fatigue_85_augmented_dataset"
INPUT_PATH = DATASET_DIR / "02_cleaned" / "sample_level_features_85.csv"

STATS_DIR = DATASET_DIR / "04_statistics_outputs"
ML_DIR = DATASET_DIR / "05_ml_outputs"
FIG_DIR = DATASET_DIR / "06_figures"

for _d in [STATS_DIR, ML_DIR, FIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "log10_nf"
EXPECTED_ROWS = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5


# =============================================================================
# Utility functions
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake-like names."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percent", regex=False)
        .str.lower()
    )

    rename_map = {
        "sample_id": "specimen_id",
        "ys_mpa": "ys_mpa",
        "yield_strength_mpa": "ys_mpa",
        "uts_mpa": "uts_mpa",
        "hardness_hv": "hardness_hv",
        "hardness_hv_": "hardness_hv",
        "log10nf": "log10_nf",
        "lognf": "log10_nf",
        "log_nf": "log10_nf",
        "log_nf_db": "log10_nf",
        "log10_nf_db": "log10_nf",
        "cycles": "cycle_no",
        "cyclces_to_failure": "cycles_to_failure",
        "cyclces_to_fatilure": "cycles_to_failure",
        "nf": "cycles_to_failure",
        "nf_cycles": "cycles_to_failure",
        "d_inv_sqrt_db": "d_inv_sqrt",
    }
    out = out.rename(columns={c: rename_map.get(c, c) for c in out.columns})
    # Drop duplicated columns after renaming by keeping first non-null value across duplicates.
    if out.columns.duplicated().any():
        new_cols = []
        for col in pd.unique(out.columns):
            same = out.loc[:, out.columns == col]
            if same.shape[1] == 1:
                new_cols.append(same.iloc[:, 0].rename(col))
            else:
                new_cols.append(same.bfill(axis=1).iloc[:, 0].rename(col))
        out = pd.concat(new_cols, axis=1)
    return out


def require_columns(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {label}: {missing}\nAvailable: {df.columns.tolist()}")


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def leave_one_route_out_predictions(
    df: pd.DataFrame,
    features: List[str],
    model,
    target: str = TARGET,
    group_col: str = "route_id",
) -> pd.DataFrame:
    rows = []
    usable = df[[group_col, "specimen_id", target] + features].dropna().copy()
    for route in sorted(usable[group_col].dropna().unique()):
        train = usable[usable[group_col] != route]
        test = usable[usable[group_col] == route]
        if len(train) < 5 or len(test) == 0:
            continue
        estimator = make_pipeline(clone(model))
        estimator.fit(train[features], train[target])
        pred = estimator.predict(test[features])
        for i, (_, row) in enumerate(test.iterrows()):
            rows.append({
                "heldout_route_id": route,
                "specimen_id": row.get("specimen_id"),
                "actual_log10_nf": row[target],
                "predicted_log10_nf": pred[i],
                "residual_log10_nf": row[target] - pred[i],
                "abs_error_log10_nf": abs(row[target] - pred[i]),
                "features_used": ";".join(features),
            })
    return pd.DataFrame(rows)


def route_mean_loocv_predictions(route_df: pd.DataFrame, features: List[str], model, target: str = TARGET) -> pd.DataFrame:
    usable = route_df[["route_id", "route_family", target] + features].dropna().copy()
    rows = []
    for idx in usable.index:
        train = usable.drop(index=idx)
        test = usable.loc[[idx]]
        if len(train) < 5:
            continue
        estimator = make_pipeline(clone(model))
        estimator.fit(train[features], train[target])
        pred = estimator.predict(test[features])[0]
        rows.append({
            "heldout_route_id": test.iloc[0]["route_id"],
            "route_family": test.iloc[0]["route_family"],
            "actual_log10_nf": test.iloc[0][target],
            "predicted_log10_nf": pred,
            "residual_log10_nf": test.iloc[0][target] - pred,
            "abs_error_log10_nf": abs(test.iloc[0][target] - pred),
            "features_used": ";".join(features),
        })
    return pd.DataFrame(rows)


def model_coefficients(df: pd.DataFrame, features: List[str], model, target: str = TARGET) -> pd.DataFrame:
    usable = df[[target] + features].dropna().copy()
    estimator = make_pipeline(clone(model))
    estimator.fit(usable[features], usable[target])
    fitted_model = estimator.named_steps["model"]
    if hasattr(fitted_model, "coef_"):
        coefs = np.ravel(fitted_model.coef_)
    elif hasattr(fitted_model, "feature_importances_"):
        coefs = np.ravel(fitted_model.feature_importances_)
    else:
        coefs = np.full(len(features), np.nan)
    return pd.DataFrame({
        "feature": features,
        "coefficient_or_importance": coefs,
        "abs_value": np.abs(coefs),
    }).sort_values("abs_value", ascending=False).reset_index(drop=True)


def plot_predicted_vs_actual(pred_df: pd.DataFrame, title: str, out_path: Path) -> None:
    if pred_df.empty:
        return
    actual = pred_df["actual_log10_nf"].to_numpy()
    predicted = pred_df["predicted_log10_nf"].to_numpy()
    lo = float(np.nanmin([actual.min(), predicted.min()])) - 0.05
    hi = float(np.nanmax([actual.max(), predicted.max()])) + 0.05
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.75)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.xlabel("Actual log10(Nf)")
    plt.ylabel("Predicted log10(Nf)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_barh(df: pd.DataFrame, y_col: str, x_col: str, title: str, xlabel: str, out_path: Path) -> None:
    if df.empty:
        return
    plot_df = df.copy().sort_values(x_col, ascending=True)
    plt.figure(figsize=(9, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df[y_col].astype(str), plot_df[x_col])
    plt.xlabel(xlabel)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main() -> None:
    print("=== START phase3_ICME_Regression_85 ===")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    raw = pd.read_csv(INPUT_PATH)
    df = normalize_columns(raw)

    # Essential identifiers and target construction
    if "specimen_id" not in df.columns:
        if "sample_id" in df.columns:
            df = df.rename(columns={"sample_id": "specimen_id"})
        else:
            df["specimen_id"] = np.arange(1, len(df) + 1).astype(str)

    require_columns(df, ["route_id", "route_family"], "ICME identifiers")

    if TARGET not in df.columns:
        require_columns(df, ["cycles_to_failure"], "target generation")
        df[TARGET] = np.log10(pd.to_numeric(df["cycles_to_failure"], errors="coerce"))

    if "d_inv_sqrt" not in df.columns:
        require_columns(df, ["grain_size_um"], "Hall-Petch descriptor")
        df["d_inv_sqrt"] = 1 / np.sqrt(pd.to_numeric(df["grain_size_um"], errors="coerce"))

    # Process context fields
    df["is_ecap"] = (df["route_family"].astype(str).str.upper() == "ECAP").astype(int)
    df["is_dct"] = (df["route_family"].astype(str).str.upper() == "DCT").astype(int)
    if "soak_hours" not in df.columns:
        df["soak_hours"] = np.nan
    if "ecap_angle_deg" not in df.columns:
        df["ecap_angle_deg"] = np.nan

    # Numeric columns used in ICME matrix
    preferred_numeric = [
        "cycles_to_failure", TARGET,
        "is_ecap", "is_dct", "soak_hours", "ecap_angle_deg",
        "grain_size_um", "d_inv_sqrt",
        "ys_mpa", "uts_mpa", "hardness_hv", "strength_ratio", "elongation_percent",
        "psa_stable_mean", "mean_stress_stable_mean", "stress_amplitude_stable_mean",
        "stress_range_stable_mean", "unloading_modulus_stable_mean", "energy_proxy_stable_mean",
        "max_stress_peak", "min_stress_minimum",
    ]
    numeric_cols = [c for c in preferred_numeric if c in df.columns]
    df = safe_numeric(df, numeric_cols)

    require_columns(df, [TARGET, "grain_size_um", "d_inv_sqrt"], "ICME regression")

    # -------------------------------------------------------------------------
    # Route-wise PSPP matrix and family summaries
    # -------------------------------------------------------------------------
    aggregation_cols = [c for c in numeric_cols if c not in ["is_ecap", "is_dct"]]
    route_agg = {c: "mean" for c in aggregation_cols}
    route_agg.update({"specimen_id": "count"})
    route_df = (
        df.groupby(["route_id", "route_family"], dropna=False)
        .agg(route_agg)
        .rename(columns={"specimen_id": "n_samples"})
        .reset_index()
    )
    # Add route-family flags after aggregation for clarity.
    route_df["is_ecap"] = (route_df["route_family"].astype(str).str.upper() == "ECAP").astype(int)
    route_df["is_dct"] = (route_df["route_family"].astype(str).str.upper() == "DCT").astype(int)
    route_df["rank_by_log10_nf"] = route_df[TARGET].rank(ascending=False, method="min").astype(int)
    route_df = route_df.sort_values("rank_by_log10_nf").reset_index(drop=True)

    family_df = (
        route_df.groupby("route_family", dropna=False)
        .agg(
            n_routes=("route_id", "count"),
            n_samples=("n_samples", "sum"),
            mean_cycles_to_failure=("cycles_to_failure", "mean") if "cycles_to_failure" in route_df.columns else (TARGET, "count"),
            mean_log10_nf=(TARGET, "mean"),
            sd_log10_nf=(TARGET, "std"),
            mean_grain_size_um=("grain_size_um", "mean"),
            mean_d_inv_sqrt=("d_inv_sqrt", "mean"),
            mean_hardness_hv=("hardness_hv", "mean") if "hardness_hv" in route_df.columns else (TARGET, "count"),
            mean_psa_stable=("psa_stable_mean", "mean") if "psa_stable_mean" in route_df.columns else (TARGET, "count"),
            mean_mean_stress_stable=("mean_stress_stable_mean", "mean") if "mean_stress_stable_mean" in route_df.columns else (TARGET, "count"),
        )
        .reset_index()
        .sort_values("mean_log10_nf", ascending=False)
        .reset_index(drop=True)
    )
    family_df["rank_by_mean_log10_nf"] = family_df["mean_log10_nf"].rank(ascending=False, method="min").astype(int)

    # -------------------------------------------------------------------------
    # ICME forward and inverse summaries
    # -------------------------------------------------------------------------
    forward_cols = [
        "route_id", "route_family", "n_samples", "rank_by_log10_nf",
        "soak_hours", "ecap_angle_deg", "grain_size_um", "d_inv_sqrt",
        "hardness_hv", "ys_mpa", "uts_mpa", "psa_stable_mean",
        "mean_stress_stable_mean", "unloading_modulus_stable_mean",
        "energy_proxy_stable_mean", "cycles_to_failure", TARGET,
    ]
    forward_cols = [c for c in forward_cols if c in route_df.columns]
    forward_df = route_df[forward_cols].copy()

    threshold_log = float(route_df[TARGET].quantile(0.75))
    inverse_df = route_df.copy()
    inverse_df["screening_rule"] = np.where(
        inverse_df[TARGET] >= threshold_log,
        "High-fatigue candidate: top quartile log10(Nf)",
        "Below top-quartile fatigue-life threshold",
    )
    inverse_df = inverse_df.sort_values(TARGET, ascending=False).reset_index(drop=True)
    inverse_cols = [
        "route_id", "route_family", "n_samples", "rank_by_log10_nf", "screening_rule",
        "cycles_to_failure", TARGET, "grain_size_um", "d_inv_sqrt",
        "hardness_hv", "psa_stable_mean", "mean_stress_stable_mean",
        "unloading_modulus_stable_mean", "energy_proxy_stable_mean",
    ]
    inverse_cols = [c for c in inverse_cols if c in inverse_df.columns]
    inverse_df = inverse_df[inverse_cols]

    # Evidence-chain correlations at route-mean level
    evidence_rows = []
    correlation_specs = [
        ("Process -> Structure", "is_ecap", "grain_size_um", "ECAP membership and grain refinement"),
        ("Process -> Structure", "soak_hours", "grain_size_um", "DCT soaking duration and grain-size response"),
        ("Structure -> Property", "d_inv_sqrt", "ys_mpa", "Hall-Petch strengthening descriptor"),
        ("Structure -> Performance", "grain_size_um", TARGET, "Grain size and fatigue-life response"),
        ("Structure -> Performance", "d_inv_sqrt", TARGET, "Inverse square-root grain size and fatigue-life response"),
        ("Property -> Performance", "hardness_hv", TARGET, "Hardness contribution to fatigue-life response"),
        ("Cyclic response -> Performance", "psa_stable_mean", TARGET, "Stabilized plastic strain amplitude and fatigue damage"),
        ("Cyclic response -> Performance", "mean_stress_stable_mean", TARGET, "Stabilized mean stress and fatigue response"),
        ("Cyclic response -> Performance", "unloading_modulus_stable_mean", TARGET, "Cyclic stiffness/degradation descriptor"),
        ("Cyclic response -> Performance", "energy_proxy_stable_mean", TARGET, "Energy-proxy fatigue damage descriptor"),
    ]
    for layer_link, x_col, y_col, note in correlation_specs:
        if x_col in route_df.columns and y_col in route_df.columns:
            tmp = route_df[[x_col, y_col]].dropna()
            if len(tmp) >= 3 and tmp[x_col].nunique() > 1 and tmp[y_col].nunique() > 1:
                pearson = tmp[x_col].corr(tmp[y_col], method="pearson")
                spearman = tmp[x_col].corr(tmp[y_col], method="spearman")
                evidence_rows.append({
                    "icme_link": layer_link,
                    "x_descriptor": x_col,
                    "y_descriptor": y_col,
                    "n_route_means": len(tmp),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "abs_spearman": abs(spearman),
                    "interpretation_note": note,
                })
    evidence_df = pd.DataFrame(evidence_rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)

    feature_layer_rows = [
        ("is_ecap", "Process", "Binary ECAP route-family indicator"),
        ("is_dct", "Process", "Binary DCT route-family indicator"),
        ("soak_hours", "Process", "DCT soaking duration metadata"),
        ("ecap_angle_deg", "Process", "ECAP die channel angle metadata"),
        ("grain_size_um", "Structure", "Mean grain size"),
        ("d_inv_sqrt", "Structure", "Hall-Petch inverse square-root grain-size descriptor"),
        ("hardness_hv", "Property", "Microhardness"),
        ("ys_mpa", "Property", "Yield strength"),
        ("uts_mpa", "Property", "Ultimate tensile strength"),
        ("strength_ratio", "Derived property", "Yield-to-UTS strength ratio"),
        ("psa_stable_mean", "Performance/Cyclic", "Stabilized plastic strain amplitude"),
        ("mean_stress_stable_mean", "Performance/Cyclic", "Stabilized mean stress"),
        ("stress_amplitude_stable_mean", "Performance/Cyclic", "Stabilized stress amplitude"),
        ("unloading_modulus_stable_mean", "Performance/Cyclic", "Stabilized unloading modulus"),
        ("energy_proxy_stable_mean", "Performance/Cyclic", "Approximate cyclic energy proxy"),
        (TARGET, "Performance target", "Log-transformed cycles to failure"),
    ]
    feature_layer_df = pd.DataFrame(feature_layer_rows, columns=["feature", "pspp_layer", "description"])
    feature_layer_df = feature_layer_df[feature_layer_df["feature"].isin(df.columns) | feature_layer_df["feature"].isin(route_df.columns)]

    # -------------------------------------------------------------------------
    # Regression model comparison for ICME integrated demonstration
    # -------------------------------------------------------------------------
    feature_sets: Dict[str, List[str]] = {
        "structure_only": ["d_inv_sqrt"],
        "structure_property": ["d_inv_sqrt", "hardness_hv", "ys_mpa"],
        "cyclic_damage": ["psa_stable_mean", "mean_stress_stable_mean", "unloading_modulus_stable_mean", "energy_proxy_stable_mean"],
        "compact_pspp_4": ["d_inv_sqrt", "hardness_hv", "psa_stable_mean", "mean_stress_stable_mean"],
        "integrated_pspp_6": ["d_inv_sqrt", "hardness_hv", "ys_mpa", "psa_stable_mean", "mean_stress_stable_mean", "unloading_modulus_stable_mean"],
        "integrated_pspp_7_energy": ["d_inv_sqrt", "hardness_hv", "ys_mpa", "psa_stable_mean", "mean_stress_stable_mean", "unloading_modulus_stable_mean", "energy_proxy_stable_mean"],
    }
    feature_sets = {name: [f for f in feats if f in df.columns] for name, feats in feature_sets.items()}
    feature_sets = {name: feats for name, feats in feature_sets.items() if len(feats) >= 1}

    models = {
        "linear_regression": LinearRegression(),
        "ridge_alpha_1": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "ridge_alpha_10": Ridge(alpha=10.0, random_state=RANDOM_STATE),
    }

    model_rows = []
    all_loro_predictions = {}
    for fs_name, feats in feature_sets.items():
        usable_sample = df[["route_id", "specimen_id", TARGET] + feats].dropna().copy()
        usable_route = route_df[["route_id", "route_family", TARGET] + feats].dropna().copy()
        if len(usable_sample) < 20 or usable_sample["route_id"].nunique() < 5:
            continue
        for model_name, model in models.items():
            estimator = make_pipeline(clone(model))
            estimator.fit(usable_sample[feats], usable_sample[TARGET])
            train_pred = estimator.predict(usable_sample[feats])

            loro_df = leave_one_route_out_predictions(usable_sample, feats, model)
            route_loocv_df = route_mean_loocv_predictions(usable_route, feats, model)
            key = f"{model_name}__{fs_name}"
            all_loro_predictions[key] = loro_df

            row = {
                "model_name": model_name,
                "feature_set": fs_name,
                "features": ";".join(feats),
                "n_features": len(feats),
                "sample_rows_used": len(usable_sample),
                "routes_used": usable_sample["route_id"].nunique(),
                "training_r2": safe_r2(usable_sample[TARGET].to_numpy(), train_pred),
                "training_rmse": rmse(usable_sample[TARGET].to_numpy(), train_pred),
                "training_mae": float(mean_absolute_error(usable_sample[TARGET], train_pred)),
            }
            if not loro_df.empty:
                row.update({
                    "loro_r2": safe_r2(loro_df["actual_log10_nf"].to_numpy(), loro_df["predicted_log10_nf"].to_numpy()),
                    "loro_rmse": rmse(loro_df["actual_log10_nf"].to_numpy(), loro_df["predicted_log10_nf"].to_numpy()),
                    "loro_mae": float(mean_absolute_error(loro_df["actual_log10_nf"], loro_df["predicted_log10_nf"])),
                    "loro_error_factor": 10 ** rmse(loro_df["actual_log10_nf"].to_numpy(), loro_df["predicted_log10_nf"].to_numpy()),
                })
            else:
                row.update({"loro_r2": np.nan, "loro_rmse": np.nan, "loro_mae": np.nan, "loro_error_factor": np.nan})

            if not route_loocv_df.empty:
                row.update({
                    "route_mean_loocv_r2": safe_r2(route_loocv_df["actual_log10_nf"].to_numpy(), route_loocv_df["predicted_log10_nf"].to_numpy()),
                    "route_mean_loocv_rmse": rmse(route_loocv_df["actual_log10_nf"].to_numpy(), route_loocv_df["predicted_log10_nf"].to_numpy()),
                    "route_mean_loocv_mae": float(mean_absolute_error(route_loocv_df["actual_log10_nf"], route_loocv_df["predicted_log10_nf"])),
                })
            else:
                row.update({"route_mean_loocv_r2": np.nan, "route_mean_loocv_rmse": np.nan, "route_mean_loocv_mae": np.nan})
            model_rows.append(row)

    model_summary_df = pd.DataFrame(model_rows).sort_values(["loro_rmse", "route_mean_loocv_rmse"], ascending=True).reset_index(drop=True)
    if model_summary_df.empty:
        raise RuntimeError("No usable ICME regression models were fitted. Check feature availability.")

    best_row = model_summary_df.iloc[0]
    best_key = f"{best_row['model_name']}__{best_row['feature_set']}"
    best_features = str(best_row["features"]).split(";")
    best_model_name = best_row["model_name"]
    best_model = models[best_model_name]
    best_loro_df = all_loro_predictions[best_key].copy()
    best_loro_df["model_name"] = best_model_name
    best_loro_df["feature_set"] = best_row["feature_set"]

    best_route_error = (
        best_loro_df.groupby("heldout_route_id", dropna=False)
        .agg(
            n_samples=("specimen_id", "count"),
            actual_log10_nf_mean=("actual_log10_nf", "mean"),
            predicted_log10_nf_mean=("predicted_log10_nf", "mean"),
            mean_abs_error_log10_nf=("abs_error_log10_nf", "mean"),
            rmse_log10_nf=("residual_log10_nf", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        )
        .reset_index()
        .sort_values("rmse_log10_nf", ascending=False)
    )

    best_coef_df = model_coefficients(df, best_features, best_model)
    best_coef_df["model_name"] = best_model_name
    best_coef_df["feature_set"] = best_row["feature_set"]

    # -------------------------------------------------------------------------
    # Validation overview
    # -------------------------------------------------------------------------
    validation_rows = [
        {"check": "Total sample rows", "expected": EXPECTED_ROWS, "observed": len(df), "status": "PASS" if len(df) == EXPECTED_ROWS else "CHECK"},
        {"check": "Unique specimen_id count", "expected": EXPECTED_ROWS, "observed": df["specimen_id"].nunique(), "status": "PASS" if df["specimen_id"].nunique() == EXPECTED_ROWS else "CHECK"},
        {"check": "Unique route_id count", "expected": EXPECTED_ROUTES, "observed": df["route_id"].nunique(), "status": "PASS" if df["route_id"].nunique() == EXPECTED_ROUTES else "CHECK"},
        {"check": "Routes with exactly five samples", "expected": EXPECTED_ROUTES, "observed": int((df.groupby("route_id")["specimen_id"].nunique() == EXPECTED_SAMPLES_PER_ROUTE).sum()), "status": "PASS" if int((df.groupby("route_id")["specimen_id"].nunique() == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES else "CHECK"},
        {"check": "Missing target log10_nf", "expected": 0, "observed": int(df[TARGET].isna().sum()), "status": "PASS" if int(df[TARGET].isna().sum()) == 0 else "FAIL"},
        {"check": "Route-mean PSPP rows", "expected": EXPECTED_ROUTES, "observed": len(route_df), "status": "PASS" if len(route_df) == EXPECTED_ROUTES else "CHECK"},
        {"check": "ICME evidence-chain rows", "expected": ">=5", "observed": len(evidence_df), "status": "PASS" if len(evidence_df) >= 5 else "CHECK"},
        {"check": "Successful regression model rows", "expected": ">=3", "observed": len(model_summary_df), "status": "PASS" if len(model_summary_df) >= 3 else "CHECK"},
        {"check": "Best model LORO predictions", "expected": EXPECTED_ROWS, "observed": len(best_loro_df), "status": "PASS" if len(best_loro_df) == EXPECTED_ROWS else "CHECK"},
    ]
    validation_df = pd.DataFrame(validation_rows)
    final_status = "PASS" if (validation_df["status"].isin(["FAIL"]).sum() == 0 and validation_df["status"].isin(["CHECK"]).sum() == 0) else "CHECK"

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------
    route_df.to_csv(STATS_DIR / "phase3_ICME_routewise_pspp_matrix_85.csv", index=False)
    family_df.to_csv(STATS_DIR / "phase3_ICME_route_family_pspp_summary_85.csv", index=False)
    forward_df.to_csv(STATS_DIR / "phase3_ICME_forward_mapping_summary_85.csv", index=False)
    inverse_df.to_csv(STATS_DIR / "phase3_ICME_inverse_screening_summary_85.csv", index=False)
    evidence_df.to_csv(STATS_DIR / "phase3_ICME_evidence_chain_85.csv", index=False)
    feature_layer_df.to_csv(STATS_DIR / "phase3_ICME_feature_layer_mapping_85.csv", index=False)
    validation_df.to_csv(STATS_DIR / "phase3_ICME_validation_overview_85.csv", index=False)

    model_summary_df.to_csv(ML_DIR / "phase3_ICME_regression_model_summary_85.csv", index=False)
    best_loro_df.to_csv(ML_DIR / "phase3_ICME_best_model_loro_predictions_85.csv", index=False)
    best_route_error.to_csv(ML_DIR / "phase3_ICME_best_model_routewise_error_85.csv", index=False)
    best_coef_df.to_csv(ML_DIR / "phase3_ICME_best_model_coefficients_85.csv", index=False)

    metadata = {
        "script": "phase3_ICME_Regression_85.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(INPUT_PATH),
        "n_rows": int(len(df)),
        "n_routes": int(df["route_id"].nunique()),
        "target": TARGET,
        "best_model_name": str(best_row["model_name"]),
        "best_feature_set": str(best_row["feature_set"]),
        "best_features": best_features,
        "best_loro_r2": None if pd.isna(best_row["loro_r2"]) else float(best_row["loro_r2"]),
        "best_loro_rmse": None if pd.isna(best_row["loro_rmse"]) else float(best_row["loro_rmse"]),
        "final_status": final_status,
    }
    with open(STATS_DIR / "phase3_ICME_metadata_85.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    plot_barh(
        route_df.sort_values(TARGET, ascending=True),
        y_col="route_id",
        x_col=TARGET,
        title="ICME route-wise fatigue-life ranking across 85 samples",
        xlabel="Mean log10(Nf)",
        out_path=FIG_DIR / "phase3_ICME_routewise_fatigue_ranking_85.png",
    )

    plot_barh(
        family_df.sort_values("mean_log10_nf", ascending=True),
        y_col="route_family",
        x_col="mean_log10_nf",
        title="ICME route-family fatigue-life summary",
        xlabel="Mean log10(Nf)",
        out_path=FIG_DIR / "phase3_ICME_route_family_summary_85.png",
    )

    if {"grain_size_um", TARGET}.issubset(route_df.columns):
        plt.figure(figsize=(8, 6))
        for fam, sub in route_df.groupby("route_family"):
            plt.scatter(sub["grain_size_um"], sub[TARGET], label=str(fam), s=70, alpha=0.85)
            for _, r in sub.iterrows():
                plt.annotate(str(r["route_id"]), (r["grain_size_um"], r[TARGET]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        plt.gca().invert_xaxis()
        plt.xlabel("Mean grain size (µm)")
        plt.ylabel("Mean log10(Nf)")
        plt.title("ICME structure-performance map: grain size vs fatigue life")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Route family")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "phase3_ICME_structure_performance_map_85.png", dpi=300)
        plt.close()

    plot_predicted_vs_actual(
        best_loro_df,
        title=f"ICME regression: leave-one-route-out prediction ({best_row['model_name']} / {best_row['feature_set']})",
        out_path=FIG_DIR / "phase3_ICME_best_model_loro_predicted_vs_actual_85.png",
    )

    if not best_coef_df.empty:
        coef_plot = best_coef_df.sort_values("coefficient_or_importance", ascending=True)
        plt.figure(figsize=(8, max(4, 0.5 * len(coef_plot))))
        plt.barh(coef_plot["feature"], coef_plot["coefficient_or_importance"])
        plt.xlabel("Standardized coefficient / importance")
        plt.ylabel("Feature")
        plt.title(f"ICME best-model feature contribution: {best_row['model_name']} / {best_row['feature_set']}")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "phase3_ICME_best_model_coefficients_85.png", dpi=300)
        plt.close()

    plot_barh(
        best_route_error,
        y_col="heldout_route_id",
        x_col="rmse_log10_nf",
        title="ICME regression route-wise prediction error",
        xlabel="Leave-one-route-out RMSE in log10(Nf)",
        out_path=FIG_DIR / "phase3_ICME_routewise_prediction_error_85.png",
    )

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    report_path = STATS_DIR / "phase3_ICME_report_85.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("phase3_ICME_Regression_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {INPUT_PATH}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        f.write(f"Route families: {', '.join(sorted(df['route_family'].astype(str).unique()))}\n")
        f.write(f"Analysis timestamp: {metadata['timestamp']}\n\n")

        f.write("Validation overview\n")
        f.write("-" * 78 + "\n")
        f.write(validation_df.to_string(index=False))
        f.write("\n\n")

        f.write("Route-family PSPP summary\n")
        f.write("-" * 78 + "\n")
        f.write(family_df.to_string(index=False))
        f.write("\n\n")

        f.write("Top route-wise ICME forward mapping records\n")
        f.write("-" * 78 + "\n")
        f.write(forward_df.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("ICME evidence-chain correlations using route-mean data\n")
        f.write("-" * 78 + "\n")
        if not evidence_df.empty:
            f.write(evidence_df.to_string(index=False))
        else:
            f.write("No evidence-chain correlations computed.\n")
        f.write("\n\n")

        f.write("Regression model summary ranked by leave-one-route-out RMSE\n")
        f.write("-" * 78 + "\n")
        f.write(model_summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("Best ICME regression model\n")
        f.write("-" * 78 + "\n")
        f.write(f"Model: {best_row['model_name']}\n")
        f.write(f"Feature set: {best_row['feature_set']}\n")
        f.write(f"Features: {best_row['features']}\n")
        f.write(f"Training R2: {best_row['training_r2']:.4f}\n")
        f.write(f"Training RMSE log10(Nf): {best_row['training_rmse']:.4f}\n")
        f.write(f"Leave-one-route-out R2: {best_row['loro_r2']:.4f}\n")
        f.write(f"Leave-one-route-out RMSE log10(Nf): {best_row['loro_rmse']:.4f}\n")
        f.write(f"Leave-one-route-out MAE log10(Nf): {best_row['loro_mae']:.4f}\n")
        f.write(f"Approximate fatigue-life error factor: {best_row['loro_error_factor']:.3f}x\n\n")

        f.write("Best-model coefficient / importance table\n")
        f.write("-" * 78 + "\n")
        f.write(best_coef_df.to_string(index=False))
        f.write("\n\n")

        f.write("Highest route-wise prediction errors for best model\n")
        f.write("-" * 78 + "\n")
        f.write(best_route_error.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("Manuscript interpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "This script consolidates the validated PSPP workflow into an ICME-ready "
            "interpretation layer. Route-wise and route-family tables provide forward "
            "navigation from processing route to structure, property, cyclic response and "
            "fatigue performance. Inverse screening identifies high-fatigue candidates "
            "from the same traceable feature matrix. The regression model is intended as "
            "a physically interpretable demonstration of the database-enabled ICME layer, "
            "not as a final deployable fatigue-life predictor. Leave-one-route-out metrics "
            "should be prioritized over training performance when reporting predictive "
            "capability.\n\n"
        )
        f.write(f"Final status: {final_status}\n")

    print("Saved report:", report_path)
    print("Best model:", best_row["model_name"], "/", best_row["feature_set"])
    print("Best LORO R2:", best_row["loro_r2"])
    print("Best LORO RMSE:", best_row["loro_rmse"])
    print(f"Done phase3_ICME_Regression_85. Status: {final_status}")


if __name__ == "__main__":
    main()

