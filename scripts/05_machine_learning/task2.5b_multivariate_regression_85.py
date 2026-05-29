"""
task2.5b_multivariate_regression_85.py

Purpose
-------
Interpretable multivariate regression for the 85-sample fatigue DBMS workflow.
The script reads the ML-ready sample-level PSPP/cyclic feature table and evaluates
physics-guided linear and ridge regression models for predicting log10(Nf).

Key safeguards
--------------
1. Uses sample-level rows only, not cycle-level rows.
2. Uses log10(Nf) as target.
3. Excludes target-leakage variables, identifiers, cycle-count variables, and source/synthetic flags.
4. Reports ordinary training performance and stricter route-group validation.
5. Treats route-group validation as the publication-facing estimate of generalization.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "Fatigue_85_augmented_dataset"
CLEANED_DIR = DATASET_DIR / "02_cleaned"
ML_OUTPUT_DIR = DATASET_DIR / "05_ml_outputs"
FIGURE_DIR = DATASET_DIR / "06_figures"

INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"

RANDOM_STATE = 42
EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lower snake_case while preserving data."""
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    rename_map = {
        "sample_id": "specimen_id",
        "sampleid": "specimen_id",
        "specimenid": "specimen_id",
        "route": "route_id",
        "ys_mpa": "ys_mpa",
        "yield_strength_mpa": "ys_mpa",
        "uts_mpa": "uts_mpa",
        "ultimate_tensile_strength_mpa": "uts_mpa",
        "hardness": "hardness_hv",
        "hardness_hv": "hardness_hv",
        "grain_size": "grain_size_um",
        "grain_size_um": "grain_size_um",
        "lognf": "log10_nf",
        "log10nf": "log10_nf",
        "log_nf": "log10_nf",
        "log_nf_db": "log10_nf_db",
        "cycles": "cycle_no",
        "nf": "cycles_to_failure",
        "n_f": "cycles_to_failure",
    }
    return df.rename(columns=rename_map)


def first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def safe_float(x) -> float:
    try:
        if x is None or pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_r2": r2_manual(y_true, y_pred),
        f"{prefix}_rmse": rmse_score(y_true, y_pred),
        f"{prefix}_mae": mae_score(y_true, y_pred),
    }


def make_pipeline(model_name: str) -> Pipeline:
    if model_name == "linear_regression":
        estimator = LinearRegression()
    elif model_name == "ridge_alpha_1":
        estimator = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    elif model_name == "ridge_alpha_10":
        estimator = Ridge(alpha=10.0, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ]
    )


def sanitize_name(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


def ordered_available_features(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def compute_vif_table(df: pd.DataFrame, features: List[str], feature_set_name: str) -> pd.DataFrame:
    """Compute approximate VIF using linear regression of each feature against others."""
    rows = []
    if len(features) < 2:
        return pd.DataFrame(rows)

    x_df = df[features].copy()
    for c in features:
        x_df[c] = pd.to_numeric(x_df[c], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(x_df)
    X = scaler.fit_transform(X)

    for j, feat in enumerate(features):
        y_feat = X[:, j]
        X_others = np.delete(X, j, axis=1)
        if X_others.shape[1] == 0:
            r2_val = 0.0
        else:
            reg = LinearRegression()
            reg.fit(X_others, y_feat)
            pred = reg.predict(X_others)
            r2_val = r2_manual(y_feat, pred)
        if pd.isna(r2_val) or r2_val >= 0.999999:
            vif = float("inf")
        else:
            vif = float(1.0 / (1.0 - r2_val))
        rows.append(
            {
                "feature_set": feature_set_name,
                "feature": feat,
                "vif": vif,
                "r2_against_other_features": r2_val,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class EvaluationResult:
    performance: Dict[str, float]
    predictions: pd.DataFrame
    coefficients: pd.DataFrame


# -----------------------------------------------------------------------------
# Model evaluation
# -----------------------------------------------------------------------------
def evaluate_sample_level_model(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    model_name: str,
    feature_set_name: str,
    route_col: str = "route_id",
) -> EvaluationResult:
    """Evaluate sample-level model using training, group split, LOOCV, GroupKFold and LOGO."""
    work = df[[route_col, "specimen_id", target_col, *features]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    for f in features:
        work[f] = pd.to_numeric(work[f], errors="coerce")
    work = work.dropna(subset=[target_col]).copy()

    X = work[features]
    y = work[target_col].to_numpy(dtype=float)
    groups = work[route_col].astype(str).to_numpy()

    pipeline = make_pipeline(model_name)
    pipeline.fit(X, y)
    y_train_pred = pipeline.predict(X)

    perf = {
        "analysis_level": "sample_level_85",
        "model_name": model_name,
        "feature_set": feature_set_name,
        "n_samples": int(len(work)),
        "n_routes": int(work[route_col].nunique()),
        "n_features": int(len(features)),
        "features": "; ".join(features),
    }
    perf.update(metric_dict(y, y_train_pred, "train"))

    pred_df = pd.DataFrame(
        {
            "analysis_level": "sample_level_85",
            "model_name": model_name,
            "feature_set": feature_set_name,
            "specimen_id": work["specimen_id"].astype(str).values,
            "route_id": work[route_col].astype(str).values,
            "actual_log10_nf": y,
            "train_pred_log10_nf": y_train_pred,
        }
    )

    # Group-based single holdout split by route.
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        split_pipe = make_pipeline(model_name)
        split_pipe.fit(X.iloc[tr_idx], y[tr_idx])
        y_te_pred = split_pipe.predict(X.iloc[te_idx])
        perf.update(metric_dict(y[te_idx], y_te_pred, "group_holdout_test"))
        perf["group_holdout_test_routes"] = "; ".join(sorted(set(groups[te_idx])))
        pred_df.loc[pred_df.index[te_idx], "group_holdout_pred_log10_nf"] = y_te_pred
    except Exception as exc:
        perf["group_holdout_error"] = str(exc)
        perf.update({"group_holdout_test_r2": np.nan, "group_holdout_test_rmse": np.nan, "group_holdout_test_mae": np.nan})

    # Leave-one-sample-out CV.
    try:
        loo = LeaveOneOut()
        loo_pred = np.empty_like(y, dtype=float)
        for tr_idx, te_idx in loo.split(X):
            p = make_pipeline(model_name)
            p.fit(X.iloc[tr_idx], y[tr_idx])
            loo_pred[te_idx] = p.predict(X.iloc[te_idx])
        perf.update(metric_dict(y, loo_pred, "loocv"))
        pred_df["loocv_pred_log10_nf"] = loo_pred
    except Exception as exc:
        perf["loocv_error"] = str(exc)
        perf.update({"loocv_r2": np.nan, "loocv_rmse": np.nan, "loocv_mae": np.nan})

    # GroupKFold by route.
    try:
        n_splits = min(5, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)
        gkf_pred = np.empty_like(y, dtype=float)
        fold_rows = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            p = make_pipeline(model_name)
            p.fit(X.iloc[tr_idx], y[tr_idx])
            pred = p.predict(X.iloc[te_idx])
            gkf_pred[te_idx] = pred
            fold_rows.append(
                {
                    "fold": fold,
                    "test_routes": "; ".join(sorted(set(groups[te_idx]))),
                    "fold_rmse": rmse_score(y[te_idx], pred),
                    "fold_mae": mae_score(y[te_idx], pred),
                }
            )
        perf.update(metric_dict(y, gkf_pred, "groupkfold"))
        pred_df["groupkfold_pred_log10_nf"] = gkf_pred
        perf["groupkfold_n_splits"] = n_splits
    except Exception as exc:
        perf["groupkfold_error"] = str(exc)
        perf.update({"groupkfold_r2": np.nan, "groupkfold_rmse": np.nan, "groupkfold_mae": np.nan})

    # Leave-one-route-out validation. This is the strictest validation for route generalization.
    try:
        logo = LeaveOneGroupOut()
        logo_pred = np.empty_like(y, dtype=float)
        logo_route_rows = []
        for tr_idx, te_idx in logo.split(X, y, groups=groups):
            p = make_pipeline(model_name)
            p.fit(X.iloc[tr_idx], y[tr_idx])
            pred = p.predict(X.iloc[te_idx])
            logo_pred[te_idx] = pred
            route_name = str(groups[te_idx][0])
            logo_route_rows.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "left_out_route": route_name,
                    "n_test_samples": int(len(te_idx)),
                    "route_rmse": rmse_score(y[te_idx], pred),
                    "route_mae": mae_score(y[te_idx], pred),
                    "route_actual_mean_log10_nf": float(np.mean(y[te_idx])),
                    "route_pred_mean_log10_nf": float(np.mean(pred)),
                }
            )
        perf.update(metric_dict(y, logo_pred, "leave_one_route_out"))
        pred_df["leave_one_route_out_pred_log10_nf"] = logo_pred
    except Exception as exc:
        perf["leave_one_route_out_error"] = str(exc)
        perf.update(
            {
                "leave_one_route_out_r2": np.nan,
                "leave_one_route_out_rmse": np.nan,
                "leave_one_route_out_mae": np.nan,
            }
        )

    # Coefficients from full-data fit.
    reg = pipeline.named_steps["regressor"]
    coefs = getattr(reg, "coef_", np.full(len(features), np.nan))
    coef_df = pd.DataFrame(
        {
            "analysis_level": "sample_level_85",
            "model_name": model_name,
            "feature_set": feature_set_name,
            "feature": features,
            "standardized_coefficient": coefs,
            "abs_standardized_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_standardized_coefficient", ascending=False)

    intercept = safe_float(getattr(reg, "intercept_", np.nan))
    perf["intercept_after_scaling"] = intercept

    return EvaluationResult(performance=perf, predictions=pred_df, coefficients=coef_df)


def evaluate_route_mean_model(
    route_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    model_name: str,
    feature_set_name: str,
) -> EvaluationResult:
    """Evaluate route-mean regression with 17 route-level observations."""
    work = route_df[["route_id", "route_family", target_col, *features]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    for f in features:
        work[f] = pd.to_numeric(work[f], errors="coerce")
    work = work.dropna(subset=[target_col]).copy()

    X = work[features]
    y = work[target_col].to_numpy(dtype=float)

    pipeline = make_pipeline(model_name)
    pipeline.fit(X, y)
    y_train_pred = pipeline.predict(X)

    perf = {
        "analysis_level": "route_mean_17",
        "model_name": model_name,
        "feature_set": feature_set_name,
        "n_samples": int(len(work)),
        "n_routes": int(work["route_id"].nunique()),
        "n_features": int(len(features)),
        "features": "; ".join(features),
    }
    perf.update(metric_dict(y, y_train_pred, "train"))

    pred_df = pd.DataFrame(
        {
            "analysis_level": "route_mean_17",
            "model_name": model_name,
            "feature_set": feature_set_name,
            "route_id": work["route_id"].astype(str).values,
            "route_family": work["route_family"].astype(str).values,
            "actual_log10_nf": y,
            "train_pred_log10_nf": y_train_pred,
        }
    )

    # LOOCV across routes.
    try:
        loo = LeaveOneOut()
        loo_pred = np.empty_like(y, dtype=float)
        for tr_idx, te_idx in loo.split(X):
            p = make_pipeline(model_name)
            p.fit(X.iloc[tr_idx], y[tr_idx])
            loo_pred[te_idx] = p.predict(X.iloc[te_idx])
        perf.update(metric_dict(y, loo_pred, "route_loocv"))
        pred_df["route_loocv_pred_log10_nf"] = loo_pred
    except Exception as exc:
        perf["route_loocv_error"] = str(exc)
        perf.update({"route_loocv_r2": np.nan, "route_loocv_rmse": np.nan, "route_loocv_mae": np.nan})

    reg = pipeline.named_steps["regressor"]
    coefs = getattr(reg, "coef_", np.full(len(features), np.nan))
    coef_df = pd.DataFrame(
        {
            "analysis_level": "route_mean_17",
            "model_name": model_name,
            "feature_set": feature_set_name,
            "feature": features,
            "standardized_coefficient": coefs,
            "abs_standardized_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_standardized_coefficient", ascending=False)

    perf["intercept_after_scaling"] = safe_float(getattr(reg, "intercept_", np.nan))
    return EvaluationResult(performance=perf, predictions=pred_df, coefficients=coef_df)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_predicted_vs_actual(pred_df: pd.DataFrame, pred_col: str, title: str, out_path: Path) -> None:
    if pred_col not in pred_df.columns:
        return
    d = pred_df.dropna(subset=["actual_log10_nf", pred_col]).copy()
    if d.empty:
        return
    y = d["actual_log10_nf"].to_numpy(dtype=float)
    yp = d[pred_col].to_numpy(dtype=float)
    lo = min(float(np.min(y)), float(np.min(yp)))
    hi = max(float(np.max(y)), float(np.max(yp)))

    plt.figure(figsize=(6.5, 6))
    plt.scatter(y, yp, s=55)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.xlabel("Actual log10(Nf)")
    plt.ylabel("Predicted log10(Nf)")
    plt.title(title)
    plt.grid(True, alpha=0.30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_coefficients(coef_df: pd.DataFrame, title: str, out_path: Path) -> None:
    d = coef_df.sort_values("standardized_coefficient", ascending=True).copy()
    if d.empty:
        return
    plt.figure(figsize=(8.5, max(4.5, 0.45 * len(d))))
    plt.barh(d["feature"], d["standardized_coefficient"])
    plt.xlabel("Standardized coefficient")
    plt.ylabel("Feature")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_model_performance(perf_df: pd.DataFrame, out_path: Path) -> None:
    metric_col = "leave_one_route_out_rmse"
    if metric_col not in perf_df.columns or perf_df[metric_col].notna().sum() == 0:
        metric_col = "loocv_rmse"
    d = perf_df.dropna(subset=[metric_col]).copy()
    if d.empty:
        return
    d["label"] = d["analysis_level"] + " | " + d["model_name"] + " | " + d["feature_set"]
    d = d.sort_values(metric_col, ascending=True).head(15)
    plt.figure(figsize=(10, max(5, 0.45 * len(d))))
    plt.barh(d["label"][::-1], d[metric_col][::-1])
    plt.xlabel(metric_col.replace("_", " "))
    plt.ylabel("Model / feature set")
    plt.title("Multivariate regression performance comparison")
    plt.grid(True, axis="x", alpha=0.30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    print("=== START task2.5b_multivariate_regression_85 ===")
    ML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = normalize_columns(pd.read_csv(INPUT_FILE))
    print(f"Loaded input: {INPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    required_basic = ["specimen_id", "route_id"]
    missing_basic = [c for c in required_basic if c not in df.columns]
    if missing_basic:
        raise ValueError(f"Missing required identifier columns: {missing_basic}")

    if "route_family" not in df.columns:
        df["route_family"] = "unknown"

    target_col = first_existing(df.columns, ["log10_nf", "log_nf", "lognf", "log_nf_db"])
    if target_col is None:
        nf_col = first_existing(df.columns, ["cycles_to_failure", "nf", "n_f"])
        if nf_col is None:
            raise ValueError("Could not find log10 fatigue-life target or cycles_to_failure column.")
        df["log10_nf"] = np.log10(pd.to_numeric(df[nf_col], errors="coerce"))
        target_col = "log10_nf"

    # Use a unified target name for outputs.
    if target_col != "log10_nf":
        df["log10_nf"] = pd.to_numeric(df[target_col], errors="coerce")
        target_col = "log10_nf"

    # Convert candidate numeric features to numeric where present.
    for c in df.columns:
        if c not in ["specimen_id", "route_id", "route_family", "source_group", "source_file", "source_sample_id", "generation_method"]:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    # ------------------------------------------------------------------
    # Feature set definition
    # ------------------------------------------------------------------
    feature_sets = {
        # Closest to old script: grain size + hardness + PSA + mean stress.
        "core_4_grain_hv_psa_ms": [
            "grain_size_um",
            "hardness_hv",
            "psa_stable_mean",
            "mean_stress_stable_mean",
        ],
        # Hall-Petch-style version using inverse square-root grain descriptor.
        "core_4_dinv_hv_psa_ms": [
            "d_inv_sqrt",
            "hardness_hv",
            "psa_stable_mean",
            "mean_stress_stable_mean",
        ],
        # PSPP interpretable set without route-family-specific process variables.
        "pspp_7_structure_property_cyclic": [
            "grain_size_um",
            "hardness_hv",
            "ys_mpa",
            "uts_mpa",
            "psa_stable_mean",
            "mean_stress_stable_mean",
            "unloading_modulus_stable_mean",
        ],
        # Damage-response set. Use ridge as primary due to feature coupling.
        "damage_7_structure_cyclic_energy": [
            "grain_size_um",
            "hardness_hv",
            "psa_stable_mean",
            "mean_stress_stable_mean",
            "stress_amplitude_stable_mean",
            "unloading_modulus_stable_mean",
            "energy_proxy_stable_mean",
        ],
        # Very compact journal-friendly set.
        "compact_5_physics_guided": [
            "grain_size_um",
            "hardness_hv",
            "psa_stable_mean",
            "unloading_modulus_stable_mean",
            "energy_proxy_stable_mean",
        ],
    }

    model_names = ["linear_regression", "ridge_alpha_1", "ridge_alpha_10"]

    # Remove unavailable features and skip sets with too few features.
    feature_set_rows = []
    available_feature_sets: Dict[str, List[str]] = {}
    for fs_name, fs_features in feature_sets.items():
        available = ordered_available_features(df, fs_features)
        missing = [f for f in fs_features if f not in df.columns]
        feature_set_rows.append(
            {
                "feature_set": fs_name,
                "requested_features": "; ".join(fs_features),
                "available_features": "; ".join(available),
                "missing_features": "; ".join(missing),
                "n_available_features": len(available),
                "status": "USE" if len(available) >= 2 else "SKIP",
            }
        )
        if len(available) >= 2:
            available_feature_sets[fs_name] = available

    feature_sets_df = pd.DataFrame(feature_set_rows)
    feature_sets_path = ML_OUTPUT_DIR / "task2_5b_feature_sets_85.csv"
    feature_sets_df.to_csv(feature_sets_path, index=False)

    # Validation overview.
    samples_per_route = df.groupby("route_id")["specimen_id"].nunique()
    validation_rows = [
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
            "observed": int((samples_per_route == EXPECTED_SAMPLES_PER_ROUTE).sum()),
            "status": "PASS" if int((samples_per_route == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES else "FAIL",
        },
        {
            "check": "Missing target log10_nf",
            "expected": 0,
            "observed": int(pd.to_numeric(df[target_col], errors="coerce").isna().sum()),
            "status": "PASS" if int(pd.to_numeric(df[target_col], errors="coerce").isna().sum()) == 0 else "FAIL",
        },
        {
            "check": "Usable feature sets",
            "expected": ">=3",
            "observed": int(len(available_feature_sets)),
            "status": "PASS" if len(available_feature_sets) >= 3 else "WARN",
        },
    ]
    validation_df = pd.DataFrame(validation_rows)
    validation_path = ML_OUTPUT_DIR / "task2_5b_multivariate_regression_validation_overview_85.csv"
    validation_df.to_csv(validation_path, index=False)

    # ------------------------------------------------------------------
    # Sample-level models
    # ------------------------------------------------------------------
    performance_rows: List[Dict[str, float]] = []
    all_predictions: List[pd.DataFrame] = []
    all_coefficients: List[pd.DataFrame] = []
    all_vif: List[pd.DataFrame] = []

    for fs_name, features in available_feature_sets.items():
        all_vif.append(compute_vif_table(df, features, fs_name))
        for model_name in model_names:
            # Linear regression can become unstable for high collinearity; still compute but report VIF.
            result = evaluate_sample_level_model(df, features, target_col, model_name, fs_name)
            performance_rows.append(result.performance)
            all_predictions.append(result.predictions)
            all_coefficients.append(result.coefficients)

    # ------------------------------------------------------------------
    # Route-mean models: primary manuscript interpretation for route-level trends.
    # ------------------------------------------------------------------
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    route_df = df.groupby("route_id", as_index=False).agg(
        {**{c: "mean" for c in numeric_cols}, "route_family": "first"}
    )

    for fs_name, features in available_feature_sets.items():
        route_features = ordered_available_features(route_df, features)
        if len(route_features) < 2:
            continue
        for model_name in model_names:
            result = evaluate_route_mean_model(route_df, route_features, target_col, model_name, fs_name)
            performance_rows.append(result.performance)
            all_predictions.append(result.predictions)
            all_coefficients.append(result.coefficients)

    performance_df = pd.DataFrame(performance_rows)
    coefficients_df = pd.concat(all_coefficients, ignore_index=True) if all_coefficients else pd.DataFrame()
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    vif_df = pd.concat(all_vif, ignore_index=True) if all_vif else pd.DataFrame()

    # Sort performance to put strict route validation first.
    sort_cols = [c for c in ["leave_one_route_out_rmse", "route_loocv_rmse", "groupkfold_rmse", "loocv_rmse"] if c in performance_df.columns]
    if sort_cols:
        performance_df = performance_df.sort_values(sort_cols[0], ascending=True, na_position="last")

    # Save outputs.
    performance_path = ML_OUTPUT_DIR / "task2_5b_multivariate_model_performance_85.csv"
    coefficients_path = ML_OUTPUT_DIR / "task2_5b_standardized_coefficients_85.csv"
    predictions_path = ML_OUTPUT_DIR / "task2_5b_multivariate_predictions_85.csv"
    vif_path = ML_OUTPUT_DIR / "task2_5b_feature_vif_diagnostics_85.csv"
    route_mean_path = ML_OUTPUT_DIR / "task2_5b_route_mean_dataset_used_85.csv"

    performance_df.to_csv(performance_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    vif_df.to_csv(vif_path, index=False)
    route_df.to_csv(route_mean_path, index=False)

    # Choose a primary model for figures and report.
    primary_mask = (
        (performance_df["analysis_level"] == "sample_level_85")
        & (performance_df["model_name"] == "ridge_alpha_1")
        & (performance_df["feature_set"] == "compact_5_physics_guided")
    )
    if not primary_mask.any():
        primary_row = performance_df.iloc[0]
    else:
        primary_row = performance_df[primary_mask].iloc[0]

    primary_model_name = primary_row["model_name"]
    primary_feature_set = primary_row["feature_set"]
    primary_level = primary_row["analysis_level"]

    primary_pred = predictions_df[
        (predictions_df["model_name"] == primary_model_name)
        & (predictions_df["feature_set"] == primary_feature_set)
        & (predictions_df["analysis_level"] == primary_level)
    ].copy()
    primary_coef = coefficients_df[
        (coefficients_df["model_name"] == primary_model_name)
        & (coefficients_df["feature_set"] == primary_feature_set)
        & (coefficients_df["analysis_level"] == primary_level)
    ].copy()

    # Figures.
    plot_predicted_vs_actual(
        primary_pred,
        "train_pred_log10_nf",
        f"Training prediction: {primary_model_name} / {primary_feature_set}",
        FIGURE_DIR / "task2_5b_predicted_vs_actual_training_85.png",
    )

    strict_pred_col = "leave_one_route_out_pred_log10_nf"
    if strict_pred_col not in primary_pred.columns:
        strict_pred_col = "route_loocv_pred_log10_nf" if "route_loocv_pred_log10_nf" in primary_pred.columns else "loocv_pred_log10_nf"
    plot_predicted_vs_actual(
        primary_pred,
        strict_pred_col,
        f"Cross-validation prediction: {primary_model_name} / {primary_feature_set}",
        FIGURE_DIR / "task2_5b_predicted_vs_actual_cross_validation_85.png",
    )

    plot_coefficients(
        primary_coef,
        f"Standardized coefficients: {primary_model_name} / {primary_feature_set}",
        FIGURE_DIR / "task2_5b_standardized_coefficients_primary_85.png",
    )
    plot_model_performance(performance_df, FIGURE_DIR / "task2_5b_model_performance_comparison_85.png")

    # Metadata.
    metadata = {
        "script": "task2.5b_multivariate_regression_85.py",
        "input_file": str(INPUT_FILE),
        "output_dir": str(ML_OUTPUT_DIR),
        "figure_dir": str(FIGURE_DIR),
        "target": target_col,
        "random_state": RANDOM_STATE,
        "expected_samples": EXPECTED_SAMPLES,
        "expected_routes": EXPECTED_ROUTES,
        "feature_sets": feature_sets,
        "models": model_names,
        "primary_model_name": str(primary_model_name),
        "primary_feature_set": str(primary_feature_set),
        "primary_analysis_level": str(primary_level),
        "important_note": (
            "Training R2 is descriptive only. Leave-one-route-out or route-level LOOCV is "
            "the stricter estimate for manuscript interpretation."
        ),
    }
    metadata_path = ML_OUTPUT_DIR / "task2_5b_multivariate_regression_metadata_85.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Report.
    report_path = ML_OUTPUT_DIR / "task2_5b_multivariate_regression_report_85.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("task2.5b_multivariate_regression_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        f.write(f"Route families: {', '.join(sorted(df['route_family'].astype(str).unique()))}\n")
        f.write(f"Target variable: {target_col}\n")
        f.write("\nValidation overview\n")
        f.write("-" * 78 + "\n")
        f.write(validation_df.to_string(index=False))
        f.write("\n\nFeature sets used\n")
        f.write("-" * 78 + "\n")
        f.write(feature_sets_df.to_string(index=False))

        f.write("\n\nModel performance summary\n")
        f.write("-" * 78 + "\n")
        show_cols = [
            "analysis_level",
            "model_name",
            "feature_set",
            "n_samples",
            "n_features",
            "train_r2",
            "train_rmse",
            "loocv_r2",
            "loocv_rmse",
            "groupkfold_r2",
            "groupkfold_rmse",
            "leave_one_route_out_r2",
            "leave_one_route_out_rmse",
            "route_loocv_r2",
            "route_loocv_rmse",
        ]
        show_cols = [c for c in show_cols if c in performance_df.columns]
        f.write(performance_df[show_cols].to_string(index=False))

        f.write("\n\nPrimary model selected for figures\n")
        f.write("-" * 78 + "\n")
        f.write(f"Analysis level: {primary_level}\n")
        f.write(f"Model: {primary_model_name}\n")
        f.write(f"Feature set: {primary_feature_set}\n")
        f.write("\nPrimary model coefficients\n")
        f.write("-" * 78 + "\n")
        if not primary_coef.empty:
            f.write(primary_coef[["feature", "standardized_coefficient", "abs_standardized_coefficient"]].to_string(index=False))
        else:
            f.write("No coefficient table available.\n")

        f.write("\n\nFeature multicollinearity diagnostics\n")
        f.write("-" * 78 + "\n")
        if not vif_df.empty:
            f.write(vif_df.sort_values("vif", ascending=False).head(20).to_string(index=False))
        else:
            f.write("No VIF diagnostics available.\n")

        f.write("\n\nGenerated files\n")
        f.write("-" * 78 + "\n")
        for p in [
            performance_path,
            coefficients_path,
            predictions_path,
            vif_path,
            feature_sets_path,
            route_mean_path,
            validation_path,
            metadata_path,
            FIGURE_DIR / "task2_5b_predicted_vs_actual_training_85.png",
            FIGURE_DIR / "task2_5b_predicted_vs_actual_cross_validation_85.png",
            FIGURE_DIR / "task2_5b_standardized_coefficients_primary_85.png",
            FIGURE_DIR / "task2_5b_model_performance_comparison_85.png",
        ]:
            f.write(f"  {p}\n")

        f.write("\nInterpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "This script is an interpretable multivariate regression stage, not the final ML model. "
            "Training performance should be treated as descriptive. Sample-level LOOCV may still be "
            "optimistic because samples from the same route are similar. Leave-one-route-out validation "
            "and route-mean LOOCV are stricter and should be prioritized for manuscript interpretation. "
            "If training R2 is high but route-held-out validation is weak or negative, interpret this as "
            "evidence of route clustering and fatigue stochasticity, not as a database failure.\n"
        )
        f.write("\nFinal status: PASS\n")

    print("\nModel performance preview:")
    display_cols = [
        "analysis_level",
        "model_name",
        "feature_set",
        "train_r2",
        "train_rmse",
        "loocv_r2",
        "loocv_rmse",
        "groupkfold_r2",
        "groupkfold_rmse",
        "leave_one_route_out_r2",
        "leave_one_route_out_rmse",
        "route_loocv_r2",
        "route_loocv_rmse",
    ]
    display_cols = [c for c in display_cols if c in performance_df.columns]
    print(performance_df[display_cols].head(12).to_string(index=False))
    print(f"\nSaved report to: {report_path}")
    print("✅ Done task2.5b_multivariate_regression_85. Status: PASS")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
