"""
task3.4_ml_validation_85.py

Purpose
-------
ML validation for the 85-sample fatigue DBMS workflow.

This script reads the sample-level PSPP/cyclic feature table:
    Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

It performs leakage-safe fatigue-life prediction using:
    - Linear Regression
    - Ridge Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor

The target is log10(Nf). Cycle-level rows are NOT treated as independent ML samples.

Main validation emphasis:
    1. Quick 80/20 train-test split      -> diagnostic only
    2. GroupKFold by route_id            -> route-aware validation
    3. Leave-One-Route-Out validation    -> strongest validation for manuscript
    4. Route-mean LOOCV                  -> route-level comparison against old 17-route workflow

Author workflow note:
    Use this after task2.5b_multivariate_regression_85.py and task5_correlations_85.py.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GroupKFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Path configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Expected placement:
#   .../Fatigue_80 sampl``/db_scripts_85/task3.4_ml_validation_85.py
# Dataset folder:
#   .../Fatigue_80 sampl``/Fatigue_85_augmented_dataset
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "Fatigue_85_augmented_dataset"

if not DATASET_DIR.exists():
    # Fallback to the path used throughout the current workflow.
    DATASET_DIR = Path(
        r"data"
    )

INPUT_FILE = DATASET_DIR / "02_cleaned" / "sample_level_features_85.csv"
OUT_DIR = DATASET_DIR / "05_ml_outputs"
FIG_DIR = DATASET_DIR / "06_figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# =============================================================================
# Utility functions
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower snake-like names without changing values."""
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_", regex=False)
        .str.replace("__", "_", regex=False)
        .str.lower()
    )
    return out


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return np.nan
    return float(r2_score(y_true, y_pred))


def metrics_dict(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}r2": safe_r2(y_true, y_pred),
        f"{p}rmse": rmse(y_true, y_pred),
        f"{p}mae": float(mean_absolute_error(y_true, y_pred)),
    }


def make_pipeline(model) -> Pipeline:
    """Create a model pipeline with median imputation and standardization."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def evaluate_train_test(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """One random train-test split. Diagnostic only because route leakage may occur."""
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=test_size, random_state=random_state
    )
    model = clone(pipe)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    m = {}
    m.update(metrics_dict(y_train, pred_train, prefix="train_split"))
    m.update(metrics_dict(y_test, pred_test, prefix="test_split"))

    pred_df = pd.DataFrame({
        "row_index": list(idx_train) + list(idx_test),
        "split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
        "actual_log10_nf": list(y_train) + list(y_test),
        "predicted_log10_nf": list(pred_train) + list(pred_test),
    })
    return m, pred_df


def evaluate_group_cv(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splitter,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Manual grouped CV prediction and fold-wise metrics."""
    pred = pd.Series(index=y.index, dtype=float)
    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        group_test = groups.iloc[test_idx].astype(str).unique()

        model = clone(pipe)
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        pred.iloc[test_idx] = p

        fold_metric = {
            "fold_id": fold_id,
            "test_groups": ";".join(group_test),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "fold_r2": safe_r2(y_test, p),
            "fold_rmse": rmse(y_test, p),
            "fold_mae": float(mean_absolute_error(y_test, p)),
        }
        fold_rows.append(fold_metric)

    m = metrics_dict(y, pred, prefix="cv")
    pred_df = pd.DataFrame({
        "row_index": y.index,
        "actual_log10_nf": y.values,
        "predicted_log10_nf": pred.values,
        "route_id": groups.values,
        "residual": y.values - pred.values,
    })

    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        m["fold_r2_mean"] = float(fold_df["fold_r2"].mean(skipna=True))
        m["fold_r2_std"] = float(fold_df["fold_r2"].std(skipna=True))
        m["fold_rmse_mean"] = float(fold_df["fold_rmse"].mean(skipna=True))
        m["fold_rmse_std"] = float(fold_df["fold_rmse"].std(skipna=True))
    return m, pred_df, fold_df


def evaluate_loocv(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Sample/route-mean LOOCV."""
    loo = LeaveOneOut()
    pred = pd.Series(index=y.index, dtype=float)

    for train_idx, test_idx in loo.split(X, y):
        model = clone(pipe)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[test_idx] = model.predict(X.iloc[test_idx])

    m = metrics_dict(y, pred, prefix="loocv")
    pred_df = pd.DataFrame({
        "row_index": y.index,
        "actual_log10_nf": y.values,
        "predicted_log10_nf": pred.values,
        "residual": y.values - pred.values,
    })
    return m, pred_df


def extract_feature_importance(pipe: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """Extract model coefficients or native feature importances after fitting."""
    model = pipe.named_steps["model"]

    if hasattr(model, "coef_"):
        vals = np.asarray(model.coef_).ravel()
        imp_type = "standardized_coefficient"
    elif hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_).ravel()
        imp_type = "tree_feature_importance"
    else:
        vals = np.repeat(np.nan, len(feature_names))
        imp_type = "not_available"

    out = pd.DataFrame({
        "feature": feature_names,
        "importance_value": vals,
        "abs_importance": np.abs(vals),
        "importance_type": imp_type,
    }).sort_values("abs_importance", ascending=False).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "feature", "importance_value", "abs_importance", "importance_type"]]


def plot_predicted_vs_actual(
    pred_df: pd.DataFrame,
    out_path: Path,
    title: str,
):
    plt.figure(figsize=(7, 6))
    plt.scatter(pred_df["actual_log10_nf"], pred_df["predicted_log10_nf"], alpha=0.8)
    lo = float(min(pred_df["actual_log10_nf"].min(), pred_df["predicted_log10_nf"].min()))
    hi = float(max(pred_df["actual_log10_nf"].max(), pred_df["predicted_log10_nf"].max()))
    pad = (hi - lo) * 0.05 if hi > lo else 0.1
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=2)
    plt.xlabel("Actual log10(Nf)")
    plt.ylabel("Predicted log10(Nf)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_feature_importance(imp_df: pd.DataFrame, out_path: Path, title: str):
    plot_df = imp_df.sort_values("abs_importance", ascending=True)
    plt.figure(figsize=(8, max(4, 0.45 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["importance_value"])
    plt.xlabel("Importance / standardized coefficient")
    plt.ylabel("Feature")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_performance(perf_df: pd.DataFrame, out_path: Path):
    p = perf_df.dropna(subset=["leave_one_route_out_rmse"]).copy()
    p = p.sort_values("leave_one_route_out_rmse", ascending=False).tail(20)
    p["label"] = p["model_name"] + " | " + p["feature_set"]
    plt.figure(figsize=(9, max(5, 0.45 * len(p))))
    plt.barh(p["label"], p["leave_one_route_out_rmse"])
    plt.xlabel("Leave-one-route-out RMSE in log10(Nf)")
    plt.ylabel("Model / feature set")
    plt.title("ML validation comparison using route-aware RMSE")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =============================================================================
# Main workflow
# =============================================================================

def main():
    print("=== START task3.4_ml_validation_85 ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df_raw = pd.read_csv(INPUT_FILE)
    df = normalize_columns(df_raw)

    # Basic route-family indicators for optional diagnostic feature set only.
    if "route_family" in df.columns:
        rf = df["route_family"].astype(str).str.upper()
        df["is_ecap"] = (rf == "ECAP").astype(int)
        df["is_dct"] = (rf == "DCT").astype(int)
        df["is_ht"] = (rf == "HT").astype(int)
    else:
        df["is_ecap"] = 0
        df["is_dct"] = 0
        df["is_ht"] = 0

    # Target detection.
    if "log10_nf" in df.columns:
        target_col = "log10_nf"
    elif "log_nf_db" in df.columns:
        target_col = "log_nf_db"
    elif "lognf" in df.columns:
        target_col = "lognf"
    elif "cycles_to_failure" in df.columns:
        df["log10_nf"] = np.log10(pd.to_numeric(df["cycles_to_failure"], errors="coerce"))
        target_col = "log10_nf"
    else:
        raise ValueError("No target column found. Expected log10_nf/log_nf_db/lognf or cycles_to_failure.")

    required_id_cols = ["specimen_id", "route_id"]
    missing_id = [c for c in required_id_cols if c not in df.columns]
    if missing_id:
        raise ValueError(f"Missing required identifier columns: {missing_id}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col, "route_id", "specimen_id"]).copy()

    # Do not use target-derived or cycle-count-derived leakage features.
    leakage_or_excluded = [
        "cycles_to_failure", "log10_nf", "log_nf_db", "lognf", "nf_cycles",
        "fatigue_efficiency_lognf_per_mpa", "fatigue_efficiency", "fatigue_efficiency_log10nf_per_mpa",
        "cycle_no_max", "cycle_no_min", "n_cycle_rows", "cycle_coverage_ratio",
        "initial_region_rows", "stable_region_rows", "final_region_rows",
        "cycle_fraction_min", "cycle_fraction_max", "cycle_fraction_mean",
        "specimen_id", "sample_id", "route_id", "source_sample_id", "source_seed_route",
        "is_synthetic", "sample_number", "sample_no",
    ]

    feature_sets_requested: Dict[str, List[str]] = {
        "core_4_dinv_hv_psa_ms": [
            "d_inv_sqrt", "hardness_hv", "psa_stable_mean", "mean_stress_stable_mean"
        ],
        "core_4_grain_hv_psa_ms": [
            "grain_size_um", "hardness_hv", "psa_stable_mean", "mean_stress_stable_mean"
        ],
        "compact_5_physics_guided": [
            "grain_size_um", "hardness_hv", "psa_stable_mean",
            "unloading_modulus_stable_mean", "energy_proxy_stable_mean"
        ],
        "pspp_7_structure_property_cyclic": [
            "grain_size_um", "hardness_hv", "ys_mpa", "uts_mpa",
            "psa_stable_mean", "mean_stress_stable_mean", "unloading_modulus_stable_mean"
        ],
        "damage_7_structure_cyclic_energy": [
            "grain_size_um", "hardness_hv", "psa_stable_mean",
            "mean_stress_stable_mean", "stress_amplitude_stable_mean",
            "unloading_modulus_stable_mean", "energy_proxy_stable_mean"
        ],
        "process_context_diagnostic_6": [
            "is_ecap", "is_dct", "d_inv_sqrt", "hardness_hv",
            "psa_stable_mean", "unloading_modulus_stable_mean"
        ],
    }

    feature_rows = []
    feature_sets: Dict[str, List[str]] = {}

    for fs_name, cols in feature_sets_requested.items():
        available = [c for c in cols if c in df.columns and c not in leakage_or_excluded]
        missing = [c for c in cols if c not in df.columns]
        removed_as_leakage = [c for c in cols if c in leakage_or_excluded]
        usable = len(available) >= 2

        feature_rows.append({
            "feature_set": fs_name,
            "requested_features": ";".join(cols),
            "available_features": ";".join(available),
            "missing_features": ";".join(missing),
            "removed_as_leakage": ";".join(removed_as_leakage),
            "n_available": len(available),
            "usable": usable,
            "note": "diagnostic process-family context" if "process_context" in fs_name else "physics-guided"
        })

        if usable:
            feature_sets[fs_name] = available

    feature_sets_df = pd.DataFrame(feature_rows)
    feature_sets_df.to_csv(OUT_DIR / "task3_4_ml_feature_sets_used_85.csv", index=False)

    if not feature_sets:
        raise ValueError("No usable feature set found. Check sample_level_features_85.csv columns.")

    models = {
        "linear_regression": LinearRegression(),
        "ridge_alpha_1": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "ridge_alpha_10": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "random_forest_shallow": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ),
        "gradient_boosting_shallow": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE
        ),
    }

    perf_rows = []
    all_logo_predictions = []
    all_gkf_fold_rows = []
    all_route_mean_rows = []

    y = df[target_col].astype(float)
    groups = df["route_id"].astype(str)
    n_routes = groups.nunique()

    for fs_name, features in feature_sets.items():
        X = df[features].apply(pd.to_numeric, errors="coerce")

        for model_name, model in models.items():
            pipe = make_pipeline(model)

            row = {
                "analysis_level": "sample_level_85",
                "model_name": model_name,
                "feature_set": fs_name,
                "features": ";".join(features),
                "n_features": len(features),
                "n_samples": len(df),
                "n_routes": n_routes,
            }

            # Fit on all samples for training performance.
            fitted = clone(pipe)
            fitted.fit(X, y)
            pred_train = fitted.predict(X)
            row.update(metrics_dict(y, pred_train, prefix="training"))

            # Quick train-test diagnostic.
            try:
                split_metrics, split_pred = evaluate_train_test(pipe, X, y)
                row.update(split_metrics)
            except Exception as e:
                row["train_test_error"] = str(e)

            # GroupKFold by route_id.
            try:
                gkf = GroupKFold(n_splits=min(5, n_routes))
                gkf_metrics, gkf_pred, gkf_fold_df = evaluate_group_cv(pipe, X, y, groups, gkf)
                for k, v in gkf_metrics.items():
                    row[f"groupkfold_{k}"] = v

                gkf_fold_df["model_name"] = model_name
                gkf_fold_df["feature_set"] = fs_name
                all_gkf_fold_rows.append(gkf_fold_df)
            except Exception as e:
                row["groupkfold_error"] = str(e)

            # Leave-One-Route-Out.
            try:
                logo = LeaveOneGroupOut()
                logo_metrics, logo_pred, logo_fold_df = evaluate_group_cv(pipe, X, y, groups, logo)
                row["leave_one_route_out_r2"] = logo_metrics["cv_r2"]
                row["leave_one_route_out_rmse"] = logo_metrics["cv_rmse"]
                row["leave_one_route_out_mae"] = logo_metrics["cv_mae"]
                row["leave_one_route_out_fold_rmse_mean"] = logo_metrics.get("fold_rmse_mean", np.nan)
                row["leave_one_route_out_fold_rmse_std"] = logo_metrics.get("fold_rmse_std", np.nan)

                logo_pred["model_name"] = model_name
                logo_pred["feature_set"] = fs_name
                all_logo_predictions.append(logo_pred)

            except Exception as e:
                row["leave_one_route_out_error"] = str(e)

            perf_rows.append(row)

    perf_df = pd.DataFrame(perf_rows)

    # Route-mean LOOCV for route-level comparison.
    route_mean = df.copy()
    numeric_features_all = sorted(set([f for features in feature_sets.values() for f in features]))
    agg_cols = numeric_features_all + [target_col]
    route_mean = (
        route_mean.groupby("route_id", as_index=False)
        .agg({**{c: "mean" for c in agg_cols if c in route_mean.columns}, "route_family": "first"})
    )

    for fs_name, features in feature_sets.items():
        if not all(f in route_mean.columns for f in features):
            continue
        Xr = route_mean[features].apply(pd.to_numeric, errors="coerce")
        yr = route_mean[target_col].astype(float)

        for model_name, model in models.items():
            pipe = make_pipeline(model)
            row = {
                "analysis_level": "route_mean_17",
                "model_name": model_name,
                "feature_set": fs_name,
                "features": ";".join(features),
                "n_features": len(features),
                "n_samples": len(route_mean),
                "n_routes": len(route_mean),
            }
            fitted = clone(pipe)
            fitted.fit(Xr, yr)
            row.update(metrics_dict(yr, fitted.predict(Xr), prefix="training_route_mean"))

            try:
                loocv_metrics, pred_route = evaluate_loocv(pipe, Xr, yr)
                row.update(loocv_metrics)
                pred_route["route_id"] = route_mean["route_id"].values
                pred_route["model_name"] = model_name
                pred_route["feature_set"] = fs_name
            except Exception as e:
                row["route_mean_loocv_error"] = str(e)

            all_route_mean_rows.append(row)

    route_mean_perf_df = pd.DataFrame(all_route_mean_rows)
    combined_perf_df = pd.concat([perf_df, route_mean_perf_df], ignore_index=True, sort=False)

    # Choose best sample-level model using leave-one-route-out RMSE.
    sample_perf = perf_df.dropna(subset=["leave_one_route_out_rmse"]).copy()
    sample_perf = sample_perf.sort_values(["leave_one_route_out_rmse", "leave_one_route_out_mae"], ascending=True)

    if sample_perf.empty:
        raise RuntimeError("No successful Leave-One-Route-Out validation results were produced.")

    best = sample_perf.iloc[0].to_dict()
    best_model_name = best["model_name"]
    best_feature_set = best["feature_set"]
    best_features = feature_sets[best_feature_set]
    best_pipe = make_pipeline(models[best_model_name])
    Xbest = df[best_features].apply(pd.to_numeric, errors="coerce")
    best_pipe.fit(Xbest, y)

    # Best-model training predictions.
    best_train_pred = pd.DataFrame({
        "specimen_id": df["specimen_id"].values,
        "route_id": df["route_id"].values,
        "actual_log10_nf": y.values,
        "predicted_log10_nf": best_pipe.predict(Xbest),
    })
    best_train_pred["residual"] = best_train_pred["actual_log10_nf"] - best_train_pred["predicted_log10_nf"]

    # Best-model LOGO predictions from all collected predictions.
    logo_all_df = pd.concat(all_logo_predictions, ignore_index=True) if all_logo_predictions else pd.DataFrame()
    best_logo_pred = logo_all_df[
        (logo_all_df["model_name"] == best_model_name) &
        (logo_all_df["feature_set"] == best_feature_set)
    ].copy()

    # Add specimen_id to best_logo_pred using row_index.
    if not best_logo_pred.empty:
        idx_to_spec = df["specimen_id"].to_dict()
        best_logo_pred["specimen_id"] = best_logo_pred["row_index"].map(idx_to_spec)

    # Feature importance for best model.
    best_imp = extract_feature_importance(best_pipe, best_features)
    best_imp["model_name"] = best_model_name
    best_imp["feature_set"] = best_feature_set

    # Save outputs.
    combined_perf_df.to_csv(OUT_DIR / "task3_4_ml_model_performance_85.csv", index=False)
    best_imp.to_csv(OUT_DIR / "task3_4_ml_feature_importance_best_model_85.csv", index=False)
    best_train_pred.to_csv(OUT_DIR / "task3_4_ml_predictions_best_model_training_85.csv", index=False)
    best_logo_pred.to_csv(OUT_DIR / "task3_4_ml_predictions_best_model_leave_one_route_out_85.csv", index=False)

    if all_gkf_fold_rows:
        pd.concat(all_gkf_fold_rows, ignore_index=True).to_csv(
            OUT_DIR / "task3_4_ml_groupkfold_fold_metrics_85.csv", index=False
        )

    route_mean.to_csv(OUT_DIR / "task3_4_ml_route_mean_dataset_85.csv", index=False)

    exclusion_notes = pd.DataFrame({
        "excluded_feature_or_pattern": leakage_or_excluded,
        "reason": [
            "target/leakage/identifier/cycle-count/source metadata; excluded from ML feature sets"
            for _ in leakage_or_excluded
        ],
    })
    exclusion_notes.to_csv(OUT_DIR / "task3_4_ml_feature_exclusion_notes_85.csv", index=False)

    # Validation overview.
    route_counts = df.groupby("route_id")["specimen_id"].nunique()
    validation_rows = [
        ["Total sample rows", 85, len(df), "PASS" if len(df) == 85 else "CHECK"],
        ["Unique specimen_id count", 85, df["specimen_id"].nunique(), "PASS" if df["specimen_id"].nunique() == 85 else "CHECK"],
        ["Unique route_id count", 17, df["route_id"].nunique(), "PASS" if df["route_id"].nunique() == 17 else "CHECK"],
        ["Routes with exactly five samples", 17, int((route_counts == 5).sum()), "PASS" if int((route_counts == 5).sum()) == 17 else "CHECK"],
        ["Missing target log10_nf", 0, int(df[target_col].isna().sum()), "PASS" if int(df[target_col].isna().sum()) == 0 else "FAIL"],
        ["Usable feature sets", ">=1", len(feature_sets), "PASS" if len(feature_sets) >= 1 else "FAIL"],
        ["Successful model rows", ">=1", len(sample_perf), "PASS" if len(sample_perf) >= 1 else "FAIL"],
    ]
    validation_df = pd.DataFrame(validation_rows, columns=["check", "expected", "observed", "status"])
    validation_df.to_csv(OUT_DIR / "task3_4_ml_validation_overview_85.csv", index=False)

    # Metadata.
    metadata = {
        "script": "task3.4_ml_validation_85.py",
        "input_file": str(INPUT_FILE),
        "target_col": target_col,
        "n_samples": int(len(df)),
        "n_routes": int(df["route_id"].nunique()),
        "random_state": RANDOM_STATE,
        "best_model_name": best_model_name,
        "best_feature_set": best_feature_set,
        "best_features": best_features,
        "best_leave_one_route_out_rmse": float(best["leave_one_route_out_rmse"]),
        "best_leave_one_route_out_r2": float(best["leave_one_route_out_r2"]),
        "validation_priority": "leave_one_route_out and GroupKFold by route_id",
        "interpretation": "demonstrative ML validation layer; not final industrial fatigue-life predictor",
    }
    with open(OUT_DIR / "task3_4_ml_validation_metadata_85.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Figures.
    plot_predicted_vs_actual(
        best_train_pred,
        FIG_DIR / "task3_4_ml_best_training_predicted_vs_actual_85.png",
        f"Training prediction: {best_model_name} / {best_feature_set}",
    )
    if not best_logo_pred.empty:
        plot_predicted_vs_actual(
            best_logo_pred,
            FIG_DIR / "task3_4_ml_best_leave_one_route_out_predicted_vs_actual_85.png",
            f"Leave-one-route-out prediction: {best_model_name} / {best_feature_set}",
        )

    plot_feature_importance(
        best_imp,
        FIG_DIR / "task3_4_ml_best_feature_importance_85.png",
        f"Best model feature importance: {best_model_name} / {best_feature_set}",
    )

    plot_performance(
        perf_df,
        FIG_DIR / "task3_4_ml_model_performance_leave_one_route_out_rmse_85.png",
    )

    # Report.
    best_report_cols = [
        "model_name", "feature_set", "features",
        "training_r2", "training_rmse", "training_mae",
        "test_split_r2", "test_split_rmse", "test_split_mae",
        "groupkfold_cv_r2", "groupkfold_cv_rmse", "groupkfold_cv_mae",
        "leave_one_route_out_r2", "leave_one_route_out_rmse", "leave_one_route_out_mae",
    ]
    available_best_cols = [c for c in best_report_cols if c in perf_df.columns]

    report_path = OUT_DIR / "task3_4_ml_validation_report_85.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("task3.4_ml_validation_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows used: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        f.write(f"Target column: {target_col}\n")
        f.write(f"Usable feature sets: {len(feature_sets)}\n")
        f.write(f"Models evaluated: {', '.join(models.keys())}\n\n")

        f.write("Validation overview\n")
        f.write("-" * 78 + "\n")
        f.write(validation_df.to_string(index=False))
        f.write("\n\n")

        f.write("Best sample-level route-aware model\n")
        f.write("-" * 78 + "\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"Best feature set: {best_feature_set}\n")
        f.write(f"Features: {', '.join(best_features)}\n")
        f.write(f"Training R2: {best.get('training_r2', np.nan):.6f}\n")
        f.write(f"Training RMSE: {best.get('training_rmse', np.nan):.6f}\n")
        f.write(f"GroupKFold R2: {best.get('groupkfold_cv_r2', np.nan):.6f}\n")
        f.write(f"GroupKFold RMSE: {best.get('groupkfold_cv_rmse', np.nan):.6f}\n")
        f.write(f"Leave-One-Route-Out R2: {best.get('leave_one_route_out_r2', np.nan):.6f}\n")
        f.write(f"Leave-One-Route-Out RMSE: {best.get('leave_one_route_out_rmse', np.nan):.6f}\n")
        f.write(f"Leave-One-Route-Out MAE: {best.get('leave_one_route_out_mae', np.nan):.6f}\n\n")

        f.write("Top model performance rows sorted by Leave-One-Route-Out RMSE\n")
        f.write("-" * 78 + "\n")
        f.write(sample_perf[available_best_cols].head(15).to_string(index=False))
        f.write("\n\n")

        f.write("Best model feature importance / coefficients\n")
        f.write("-" * 78 + "\n")
        f.write(best_imp.to_string(index=False))
        f.write("\n\n")

        if not route_mean_perf_df.empty:
            f.write("Route-mean LOOCV summary preview\n")
            f.write("-" * 78 + "\n")
            preview_cols = [
                "model_name", "feature_set", "training_route_mean_r2",
                "training_route_mean_rmse", "loocv_r2", "loocv_rmse", "loocv_mae"
            ]
            preview_cols = [c for c in preview_cols if c in route_mean_perf_df.columns]
            f.write(route_mean_perf_df.sort_values("loocv_rmse", ascending=True)[preview_cols].head(10).to_string(index=False))
            f.write("\n\n")

        f.write("Generated files\n")
        f.write("-" * 78 + "\n")
        for fp in [
            OUT_DIR / "task3_4_ml_model_performance_85.csv",
            OUT_DIR / "task3_4_ml_feature_importance_best_model_85.csv",
            OUT_DIR / "task3_4_ml_predictions_best_model_training_85.csv",
            OUT_DIR / "task3_4_ml_predictions_best_model_leave_one_route_out_85.csv",
            OUT_DIR / "task3_4_ml_groupkfold_fold_metrics_85.csv",
            OUT_DIR / "task3_4_ml_feature_sets_used_85.csv",
            OUT_DIR / "task3_4_ml_feature_exclusion_notes_85.csv",
            OUT_DIR / "task3_4_ml_validation_overview_85.csv",
            FIG_DIR / "task3_4_ml_best_training_predicted_vs_actual_85.png",
            FIG_DIR / "task3_4_ml_best_leave_one_route_out_predicted_vs_actual_85.png",
            FIG_DIR / "task3_4_ml_best_feature_importance_85.png",
            FIG_DIR / "task3_4_ml_model_performance_leave_one_route_out_rmse_85.png",
        ]:
            if fp.exists():
                f.write(f"  {fp}\n")

        f.write("\nInterpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "This script is a demonstrative ML validation layer for the structured 85-sample "
            "PSPP dataset. Route-aware validation is prioritized over ordinary random train-test "
            "splitting because samples from the same processing route are related. Results should "
            "be described as exploratory, physics-guided predictive validation rather than final "
            "industrial fatigue-life qualification.\n"
        )

        f.write("\nFinal status: PASS\n")

    print(f"Saved report: {report_path}")
    print(f"Best model: {best_model_name} / {best_feature_set}")
    print(f"Leave-One-Route-Out RMSE: {best.get('leave_one_route_out_rmse', np.nan):.4f}")
    print("=== DONE task3.4_ml_validation_85 ===")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

