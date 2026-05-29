"""
task3.4_ml_validation_4feature_85.py
------------------------------------------------------------
Publication-ready 4-feature ML validation for the 85-sample fatigue DBMS workflow.

Purpose
-------
Freeze and validate a compact, leakage-safe, physically interpretable model for
predicting log10(Nf) using four PSPP-aligned descriptors:
    1. d_inv_sqrt
    2. hardness_hv
    3. psa_stable_mean
    4. mean_stress_stable_mean

The script uses sample-level data only, not cycle-level rows.
Main manuscript-relevant validation: Leave-One-Route-Out (LORO).

Expected location
-----------------
Place this script inside:
    <project_root>/db_scripts_85/

Expected input
--------------
    <project_root>/Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Outputs
-------
    <project_root>/Fatigue_85_augmented_dataset/05_ml_outputs/
    <project_root>/Fatigue_85_augmented_dataset/06_figures/
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, LeaveOneOut, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Configuration
# =============================================================================
RANDOM_SEED = 42
PRIMARY_MODEL_NAME = "ridge_alpha_1"
PRIMARY_FEATURE_SET_NAME = "core_4_dinv_hv_psa_ms"

FEATURES = [
    "d_inv_sqrt",
    "hardness_hv",
    "psa_stable_mean",
    "mean_stress_stable_mean",
]

ID_COLS = ["specimen_id", "sample_id", "route_id", "route_family"]
TARGET_CANDIDATES = ["log10_nf", "log10nf", "log_nf", "log_nf_db", "log10n_f"]
CYCLES_CANDIDATES = ["cycles_to_failure", "cyclces_to_failure", "nf", "n_f"]


# =============================================================================
# Path helpers
# =============================================================================
def get_project_paths() -> Dict[str, Path]:
    """Resolve project and dataset folders when this script is run from db_scripts_85."""
    script_dir = Path(__file__).resolve().parent

    # Usual case: <project_root>/db_scripts_85/this_script.py
    project_root = script_dir.parent if script_dir.name.lower() == "db_scripts_85" else script_dir

    dataset_dir = project_root / "Fatigue_85_augmented_dataset"
    input_path = dataset_dir / "02_cleaned" / "sample_level_features_85.csv"
    stats_dir = dataset_dir / "04_statistics_outputs"
    ml_dir = dataset_dir / "05_ml_outputs"
    fig_dir = dataset_dir / "06_figures"

    ml_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    return {
        "script_dir": script_dir,
        "project_root": project_root,
        "dataset_dir": dataset_dir,
        "input_path": input_path,
        "stats_dir": stats_dir,
        "ml_dir": ml_dir,
        "fig_dir": fig_dir,
    }


# =============================================================================
# Data preparation
# =============================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower snake-like labels."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "percent", regex=False)
        .str.lower()
    )

    alias_map = {
        "ys_mpa": "ys_mpa",
        "yield_strength_mpa": "ys_mpa",
        "hardness_hv": "hardness_hv",
        "hardness_h_v": "hardness_hv",
        "hardness": "hardness_hv",
        "d_inv_sqrt_db": "d_inv_sqrt",
        "d_inverse_sqrt": "d_inv_sqrt",
        "psa_mean": "psa_stable_mean",
        "psa_stable": "psa_stable_mean",
        "mean_stress_mean": "mean_stress_stable_mean",
        "ms_mean": "mean_stress_stable_mean",
        "mean_stress_stable": "mean_stress_stable_mean",
        "log10nf": "log10_nf",
        "log10n_f": "log10_nf",
        "log_nf": "log10_nf",
        "log_nf_db": "log10_nf",
        "sampleid": "sample_id",
        "specimenid": "specimen_id",
        "routeid": "route_id",
    }
    df = df.rename(columns={c: alias_map.get(c, c) for c in df.columns})

    # If duplicate columns appear after aliasing, coalesce left-to-right.
    if df.columns.duplicated().any():
        new_cols = []
        for col in pd.unique(df.columns):
            same = df.loc[:, df.columns == col]
            if same.shape[1] == 1:
                new_cols.append(same.iloc[:, 0].rename(col))
            else:
                new_cols.append(same.bfill(axis=1).iloc[:, 0].rename(col))
        df = pd.concat(new_cols, axis=1)

    return df


def ensure_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Ensure a target column named log10_nf exists."""
    df = df.copy()
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            df["log10_nf"] = pd.to_numeric(df[col], errors="coerce")
            return df, "log10_nf"

    for col in CYCLES_CANDIDATES:
        if col in df.columns:
            cycles = pd.to_numeric(df[col], errors="coerce")
            df["log10_nf"] = np.where(cycles > 0, np.log10(cycles), np.nan)
            return df, "log10_nf"

    raise ValueError(
        "Target not found. Expected one of log10_nf/log10nf/log_nf/log_nf_db "
        "or cycles_to_failure to compute log10_nf."
    )


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}\nAvailable columns: {df.columns.tolist()}")


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    """Return cleaned modelling dataframe, X, y, and route groups."""
    require_columns(df, ["route_id"] + FEATURES + ["log10_nf"], "model")

    keep_cols = [c for c in ID_COLS if c in df.columns] + FEATURES + ["log10_nf"]
    data = df[keep_cols].copy()

    for col in FEATURES + ["log10_nf"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES + ["log10_nf", "route_id"]).reset_index(drop=True)

    X = data[FEATURES].copy()
    y = data["log10_nf"].copy()
    groups = data["route_id"].astype(str).values
    return data, X, y, groups


# =============================================================================
# Models and metrics
# =============================================================================
def make_models() -> Dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ridge_alpha_0_1": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=0.1, random_state=RANDOM_SEED)),
        ]),
        "ridge_alpha_1": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]),
        "ridge_alpha_10": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        ]),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        f"{prefix}r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}mean_abs_error_factor": float(np.mean(10 ** np.abs(y_true - y_pred))),
        f"{prefix}median_abs_error_factor": float(np.median(10 ** np.abs(y_true - y_pred))),
    }
    return out


def cross_val_predict_manual(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    groups: np.ndarray | None = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Manual CV prediction to keep fold metrics and avoid version-dependent helpers."""
    preds = np.full(len(y), np.nan, dtype=float)
    fold_rows = []

    split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = fitted.predict(X.iloc[test_idx])
        preds[test_idx] = fold_pred

        row = {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        if groups is not None:
            row["test_routes"] = ";".join(sorted(pd.unique(groups[test_idx]).astype(str)))
        row.update(regression_metrics(y.iloc[test_idx].values, fold_pred, prefix="fold_"))
        fold_rows.append(row)

    return preds, pd.DataFrame(fold_rows)


def fit_route_mean_model(data: pd.DataFrame, model: Pipeline) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Aggregate to 17 route means and perform route-mean LOOCV."""
    agg_spec = {f: "mean" for f in FEATURES}
    agg_spec.update({"log10_nf": "mean"})
    if "route_family" in data.columns:
        agg_spec["route_family"] = "first"

    route_mean = data.groupby("route_id", as_index=False).agg(agg_spec)
    Xr = route_mean[FEATURES]
    yr = route_mean["log10_nf"]

    loo = LeaveOneOut()
    preds, fold_metrics = cross_val_predict_manual(model, Xr, yr, loo)
    route_mean["route_mean_loocv_pred_log10_nf"] = preds
    route_mean["route_mean_loocv_residual"] = route_mean["log10_nf"] - route_mean["route_mean_loocv_pred_log10_nf"]

    metrics = regression_metrics(yr.values, preds, prefix="route_mean_loocv_")
    metrics["route_mean_n_routes"] = len(route_mean)
    metrics["route_mean_fold_count"] = len(fold_metrics)
    return route_mean, metrics


def standardized_coefficients(model: Pipeline, features: List[str]) -> pd.DataFrame:
    """Extract standardized coefficients from a fitted scaler+linear/ridge pipeline."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise ValueError("Model does not expose coefficients.")
    coefs = np.asarray(estimator.coef_, dtype=float).ravel()
    df = pd.DataFrame({
        "feature": features,
        "standardized_coefficient": coefs,
        "abs_standardized_coefficient": np.abs(coefs),
    }).sort_values("abs_standardized_coefficient", ascending=False)
    df["rank_abs_coefficient"] = np.arange(1, len(df) + 1)
    return df


# =============================================================================
# Plotting
# =============================================================================
def plot_predicted_vs_actual(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    plt.xlabel("Actual log10(Nf)")
    plt.ylabel("Predicted log10(Nf)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_coefficients(coef_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = coef_df.sort_values("standardized_coefficient")
    plt.figure(figsize=(7, 4.5))
    plt.barh(plot_df["feature"], plot_df["standardized_coefficient"])
    plt.xlabel("Standardized coefficient")
    plt.ylabel("Feature")
    plt.title("Four-feature Ridge model coefficients")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_routewise_error(route_error: pd.DataFrame, out_path: Path) -> None:
    plot_df = route_error.sort_values("loro_rmse_log10_nf", ascending=True)
    plt.figure(figsize=(8, 7))
    plt.barh(plot_df["route_id"].astype(str), plot_df["loro_rmse_log10_nf"])
    plt.xlabel("Leave-one-route-out RMSE in log10(Nf)")
    plt.ylabel("Processing route")
    plt.title("Route-wise prediction error for four-feature model")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metric_comparison(metrics_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = metrics_df.sort_values("leave_one_route_out_rmse", ascending=False)
    labels = plot_df["model_name"].astype(str)
    plt.figure(figsize=(8, 4.8))
    plt.barh(labels, plot_df["leave_one_route_out_rmse"])
    plt.xlabel("Leave-one-route-out RMSE in log10(Nf)")
    plt.ylabel("Model")
    plt.title("Four-feature model comparison")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Report generation
# =============================================================================
def make_validation_overview(df_raw: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, expected, observed, status: str | None = None):
        if status is None:
            status = "PASS" if expected == observed else "FAIL"
        rows.append({"check": check, "expected": expected, "observed": observed, "status": status})

    add("Total raw sample rows", 85, len(df_raw))
    if "specimen_id" in df_raw.columns:
        add("Unique specimen_id count", 85, df_raw["specimen_id"].nunique())
    elif "sample_id" in df_raw.columns:
        add("Unique sample_id count", 85, df_raw["sample_id"].nunique())

    add("Unique route_id count", 17, df_raw["route_id"].nunique() if "route_id" in df_raw.columns else np.nan)
    if "route_id" in df_raw.columns:
        route_counts = df_raw.groupby("route_id").size()
        add("Routes with exactly five samples", 17, int((route_counts == 5).sum()))

    add("Rows used after four-feature cleaning", 85, len(data), "PASS" if len(data) == 85 else "CHECK")
    add("Missing log10_nf after cleaning", 0, int(data["log10_nf"].isna().sum()))
    for f in FEATURES:
        add(f"Missing {f} after cleaning", 0, int(data[f].isna().sum()))

    return pd.DataFrame(rows)


def generate_report(
    paths: Dict[str, Path],
    data: pd.DataFrame,
    validation: pd.DataFrame,
    metrics_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    route_error: pd.DataFrame,
    primary_metrics: Dict[str, float],
    out_files: Dict[str, Path],
) -> str:
    best = metrics_df.sort_values("leave_one_route_out_rmse", ascending=True).iloc[0]

    lines = []
    lines.append("task3.4_ml_validation_4feature_85 report")
    lines.append("=" * 78)
    lines.append(f"Input file: {paths['input_path']}")
    lines.append(f"Rows used after cleaning: {len(data)}")
    lines.append(f"Unique routes: {data['route_id'].nunique()}")
    lines.append(f"Feature set: {PRIMARY_FEATURE_SET_NAME}")
    lines.append("Features: " + ", ".join(FEATURES))
    lines.append("Target: log10(Nf)")
    lines.append(f"Random seed: {RANDOM_SEED}")
    lines.append(f"Analysis timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("Validation overview")
    lines.append("-" * 78)
    lines.append(validation.to_string(index=False))
    lines.append("")

    lines.append("Model performance summary")
    lines.append("-" * 78)
    display_cols = [
        "model_name", "training_r2", "training_rmse", "loocv_r2", "loocv_rmse",
        "groupkfold_r2", "groupkfold_rmse", "leave_one_route_out_r2",
        "leave_one_route_out_rmse", "leave_one_route_out_mae",
        "route_mean_loocv_r2", "route_mean_loocv_rmse",
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    lines.append(metrics_df[display_cols].to_string(index=False))
    lines.append("")

    lines.append("Primary model")
    lines.append("-" * 78)
    lines.append(f"Primary model selected: {PRIMARY_MODEL_NAME}")
    lines.append(f"Training R2 = {primary_metrics['training_r2']:.4f}")
    lines.append(f"Training RMSE = {primary_metrics['training_rmse']:.4f}")
    lines.append(f"LOOCV R2 = {primary_metrics['loocv_r2']:.4f}")
    lines.append(f"LOOCV RMSE = {primary_metrics['loocv_rmse']:.4f}")
    lines.append(f"GroupKFold R2 = {primary_metrics['groupkfold_r2']:.4f}")
    lines.append(f"GroupKFold RMSE = {primary_metrics['groupkfold_rmse']:.4f}")
    lines.append(f"Leave-one-route-out R2 = {primary_metrics['leave_one_route_out_r2']:.4f}")
    lines.append(f"Leave-one-route-out RMSE = {primary_metrics['leave_one_route_out_rmse']:.4f}")
    lines.append(f"Leave-one-route-out MAE = {primary_metrics['leave_one_route_out_mae']:.4f}")
    lines.append(
        "Approximate mean absolute prediction factor from LORO = "
        f"{primary_metrics['leave_one_route_out_mean_abs_error_factor']:.3f}x"
    )
    lines.append("")

    lines.append("Primary standardized coefficients")
    lines.append("-" * 78)
    lines.append(coef_df.to_string(index=False))
    lines.append("")

    lines.append("Route-wise error summary preview")
    lines.append("-" * 78)
    preview_cols = [
        "route_id", "route_family", "n_samples", "actual_mean_log10_nf", "predicted_mean_log10_nf",
        "mean_residual_log10_nf", "loro_rmse_log10_nf", "loro_mae_log10_nf",
    ]
    preview_cols = [c for c in preview_cols if c in route_error.columns]
    lines.append(route_error.sort_values("loro_rmse_log10_nf", ascending=False)[preview_cols].head(17).to_string(index=False))
    lines.append("")

    lines.append("Generated files")
    lines.append("-" * 78)
    for p in out_files.values():
        lines.append(f"  {p}")
    lines.append("")

    lines.append("Interpretation note")
    lines.append("-" * 78)
    lines.append(
        "This script freezes the compact four-feature Ridge regression model identified in the broader "
        "ML validation stage. It uses only leakage-safe PSPP descriptors and validates predictions using "
        "leave-one-route-out testing, which is more appropriate than random cycle-level splitting for the "
        "85-sample fatigue dataset. The model should be described as a manuscript-ready, interpretable "
        "demonstration of PSPP-guided fatigue-life prediction, not as a fully qualified industrial model."
    )
    lines.append("")
    lines.append("Final status: PASS" if (validation["status"].eq("FAIL").sum() == 0) else "Final status: CHECK")

    return "\n".join(lines)


# =============================================================================
# Main workflow
# =============================================================================
def main() -> None:
    paths = get_project_paths()
    input_path = paths["input_path"]
    ml_dir = paths["ml_dir"]
    fig_dir = paths["fig_dir"]

    print("=== START task3.4_ml_validation_4feature_85 ===")
    print(f"Input: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Confirm that task3.3_cycle_aggregated_85.py has generated sample_level_features_85.csv."
        )

    df_raw = pd.read_csv(input_path)
    df = normalize_columns(df_raw)
    df, target_col = ensure_target(df)

    data, X, y, groups = prepare_model_data(df)
    validation = make_validation_overview(df, data)

    models = make_models()
    metrics_rows = []
    all_fold_rows = []
    primary_objects = {}

    # Splitter configuration
    n_groups = len(pd.unique(groups))
    group_splits = min(5, n_groups)

    for model_name, model in models.items():
        print(f"Running model: {model_name}")

        # Training fit
        fitted = clone(model)
        fitted.fit(X, y)
        train_pred = fitted.predict(X)
        row = {
            "model_name": model_name,
            "feature_set": PRIMARY_FEATURE_SET_NAME,
            "n_samples": len(y),
            "n_features": len(FEATURES),
        }
        row.update(regression_metrics(y.values, train_pred, prefix="training_"))

        # Random 80/20 diagnostic only
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_SEED
        )
        split_model = clone(model)
        split_model.fit(X_train, y_train)
        split_pred = split_model.predict(X_test)
        row.update(regression_metrics(y_test.values, split_pred, prefix="random_80_20_test_"))

        # Sample-level LOOCV
        loocv = LeaveOneOut()
        loocv_pred, loocv_folds = cross_val_predict_manual(clone(model), X, y, loocv)
        row.update(regression_metrics(y.values, loocv_pred, prefix="loocv_"))

        # GroupKFold by route
        gkf = GroupKFold(n_splits=group_splits)
        gkf_pred, gkf_folds = cross_val_predict_manual(clone(model), X, y, gkf, groups=groups)
        row.update(regression_metrics(y.values, gkf_pred, prefix="groupkfold_"))

        # Leave-one-route-out
        logo = LeaveOneGroupOut()
        loro_pred, loro_folds = cross_val_predict_manual(clone(model), X, y, logo, groups=groups)
        row.update(regression_metrics(y.values, loro_pred, prefix="leave_one_route_out_"))

        # Route-mean LOOCV
        route_mean_df, route_mean_metrics = fit_route_mean_model(data, clone(model))
        row.update(route_mean_metrics)

        metrics_rows.append(row)

        gkf_folds.insert(0, "cv_strategy", "GroupKFold")
        gkf_folds.insert(1, "model_name", model_name)
        loro_folds.insert(0, "cv_strategy", "LeaveOneRouteOut")
        loro_folds.insert(1, "model_name", model_name)
        all_fold_rows.append(gkf_folds)
        all_fold_rows.append(loro_folds)

        if model_name == PRIMARY_MODEL_NAME:
            primary_objects = {
                "model": fitted,
                "train_pred": train_pred,
                "loocv_pred": loocv_pred,
                "groupkfold_pred": gkf_pred,
                "loro_pred": loro_pred,
                "route_mean_df": route_mean_df,
                "metrics": row,
            }

    metrics_df = pd.DataFrame(metrics_rows).sort_values("leave_one_route_out_rmse", ascending=True)
    fold_metrics_df = pd.concat(all_fold_rows, ignore_index=True) if all_fold_rows else pd.DataFrame()

    primary_model = primary_objects["model"]
    primary_metrics = primary_objects["metrics"]
    coef_df = standardized_coefficients(primary_model, FEATURES)

    # Prediction tables for the primary model
    pred_train = data[[c for c in ID_COLS if c in data.columns]].copy()
    pred_train["actual_log10_nf"] = y.values
    pred_train["predicted_log10_nf_training"] = primary_objects["train_pred"]
    pred_train["residual_training"] = pred_train["actual_log10_nf"] - pred_train["predicted_log10_nf_training"]
    pred_train["abs_error_factor_training"] = 10 ** pred_train["residual_training"].abs()

    pred_loro = data[[c for c in ID_COLS if c in data.columns]].copy()
    pred_loro["actual_log10_nf"] = y.values
    pred_loro["predicted_log10_nf_loro"] = primary_objects["loro_pred"]
    pred_loro["residual_loro"] = pred_loro["actual_log10_nf"] - pred_loro["predicted_log10_nf_loro"]
    pred_loro["abs_error_factor_loro"] = 10 ** pred_loro["residual_loro"].abs()

    pred_loocv = data[[c for c in ID_COLS if c in data.columns]].copy()
    pred_loocv["actual_log10_nf"] = y.values
    pred_loocv["predicted_log10_nf_loocv"] = primary_objects["loocv_pred"]
    pred_loocv["residual_loocv"] = pred_loocv["actual_log10_nf"] - pred_loocv["predicted_log10_nf_loocv"]
    pred_loocv["abs_error_factor_loocv"] = 10 ** pred_loocv["residual_loocv"].abs()

    # Route-wise error summary from LORO predictions
    route_error = pred_loro.groupby("route_id", as_index=False).agg(
        n_samples=("actual_log10_nf", "size"),
        actual_mean_log10_nf=("actual_log10_nf", "mean"),
        predicted_mean_log10_nf=("predicted_log10_nf_loro", "mean"),
        mean_residual_log10_nf=("residual_loro", "mean"),
        loro_rmse_log10_nf=("residual_loro", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        loro_mae_log10_nf=("residual_loro", lambda x: float(np.mean(np.abs(x)))),
        mean_abs_error_factor_loro=("abs_error_factor_loro", "mean"),
    )
    if "route_family" in pred_loro.columns:
        fam = pred_loro.groupby("route_id", as_index=False)["route_family"].first()
        route_error = route_error.merge(fam, on="route_id", how="left")
        cols = ["route_id", "route_family"] + [c for c in route_error.columns if c not in {"route_id", "route_family"}]
        route_error = route_error[cols]

    route_mean_df = primary_objects["route_mean_df"]

    # Output paths
    out_files = {
        "report": ml_dir / "task3_4_4feature_model_report_85.txt",
        "metrics": ml_dir / "task3_4_4feature_model_metrics_85.csv",
        "coefficients": ml_dir / "task3_4_4feature_coefficients_85.csv",
        "pred_training": ml_dir / "task3_4_4feature_predictions_training_85.csv",
        "pred_loro": ml_dir / "task3_4_4feature_predictions_loro_85.csv",
        "pred_loocv": ml_dir / "task3_4_4feature_predictions_loocv_85.csv",
        "route_error": ml_dir / "task3_4_4feature_routewise_error_summary_85.csv",
        "route_mean": ml_dir / "task3_4_4feature_route_mean_loocv_dataset_85.csv",
        "fold_metrics": ml_dir / "task3_4_4feature_fold_metrics_85.csv",
        "validation": ml_dir / "task3_4_4feature_validation_overview_85.csv",
        "metadata": ml_dir / "task3_4_4feature_metadata_85.json",
        "fig_training": fig_dir / "task3_4_4feature_training_predicted_vs_actual_85.png",
        "fig_loro": fig_dir / "task3_4_4feature_loro_predicted_vs_actual_85.png",
        "fig_coefficients": fig_dir / "task3_4_4feature_coefficients_85.png",
        "fig_route_error": fig_dir / "task3_4_4feature_routewise_error_85.png",
        "fig_metric_comparison": fig_dir / "task3_4_4feature_model_comparison_85.png",
    }

    # Save tables
    metrics_df.to_csv(out_files["metrics"], index=False)
    coef_df.to_csv(out_files["coefficients"], index=False)
    pred_train.to_csv(out_files["pred_training"], index=False)
    pred_loro.to_csv(out_files["pred_loro"], index=False)
    pred_loocv.to_csv(out_files["pred_loocv"], index=False)
    route_error.to_csv(out_files["route_error"], index=False)
    route_mean_df.to_csv(out_files["route_mean"], index=False)
    fold_metrics_df.to_csv(out_files["fold_metrics"], index=False)
    validation.to_csv(out_files["validation"], index=False)

    metadata = {
        "script": "task3.4_ml_validation_4feature_85.py",
        "input_file": str(input_path),
        "analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
        "target": "log10_nf",
        "primary_model": PRIMARY_MODEL_NAME,
        "feature_set": PRIMARY_FEATURE_SET_NAME,
        "features": FEATURES,
        "n_samples_used": int(len(data)),
        "n_routes": int(data["route_id"].nunique()),
        "random_seed": RANDOM_SEED,
        "validation_methods": [
            "training_fit",
            "random_80_20_diagnostic",
            "sample_level_LOOCV",
            "GroupKFold_by_route",
            "LeaveOneRouteOut",
            "route_mean_LOOCV",
        ],
        "interpretation": (
            "Four-feature Ridge model is intended as a manuscript-ready, interpretable, "
            "route-aware demonstration of PSPP-guided fatigue-life prediction."
        ),
    }
    with open(out_files["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save figures
    plot_predicted_vs_actual(
        pred_train["actual_log10_nf"],
        pred_train["predicted_log10_nf_training"],
        "Training prediction: four-feature Ridge model",
        out_files["fig_training"],
    )
    plot_predicted_vs_actual(
        pred_loro["actual_log10_nf"],
        pred_loro["predicted_log10_nf_loro"],
        "Leave-one-route-out prediction: four-feature Ridge model",
        out_files["fig_loro"],
    )
    plot_coefficients(coef_df, out_files["fig_coefficients"])
    plot_routewise_error(route_error, out_files["fig_route_error"])
    plot_metric_comparison(metrics_df, out_files["fig_metric_comparison"])

    # Report last so it includes all generated files
    report = generate_report(
        paths=paths,
        data=data,
        validation=validation,
        metrics_df=metrics_df,
        coef_df=coef_df,
        route_error=route_error,
        primary_metrics=primary_metrics,
        out_files=out_files,
    )
    with open(out_files["report"], "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("\n=== DONE task3.4_ml_validation_4feature_85 ===")


if __name__ == "__main__":
    main()
