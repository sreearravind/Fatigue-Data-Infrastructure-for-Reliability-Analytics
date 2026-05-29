"""
task5_correlations_85.py

Purpose
-------
Create PSPP-oriented correlation matrices and heatmaps for the 85-sample
fatigue dataset. This script reads the ML-ready sample-level dataset
(sample_level_features_85.csv), selects physically meaningful descriptors,
computes Pearson and Spearman correlations, ranks target associations, and
exports manuscript-ready correlation tables and figures.

Important
---------
This script is for statistical interpretation and manuscript visualization.
It includes log10_nf as the target in selected correlation matrices, but it
excludes direct leakage variables from feature-ranking outputs where needed.
For ML modelling, use the ML-safe ranked features and avoid target-derived
or cycle-count-derived variables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# =============================================================================
# 0. PATH CONFIGURATION
# =============================================================================

EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5
RANDOM_SEED = 42


def resolve_project_root() -> Path:
    """Resolve project root whether script is run from db_scripts_85 or elsewhere."""
    script_dir = Path(__file__).resolve().parent

    # Expected placement:
    # <root>/db_scripts_85/task5_correlations_85.py
    if script_dir.name.lower() == "db_scripts_85":
        return script_dir.parent

    # Fallback for direct execution from project root
    if (script_dir / "Fatigue_85_augmented_dataset").exists():
        return script_dir

    # Last fallback: parent of current directory
    return script_dir.parent


PROJECT_ROOT = resolve_project_root()
DATASET_DIR = PROJECT_ROOT / "Fatigue_85_augmented_dataset"
CLEANED_DIR = DATASET_DIR / "02_cleaned"
STATS_DIR = DATASET_DIR / "04_statistics_outputs"
FIG_DIR = DATASET_DIR / "06_figures"

INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================


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
    return df


def ensure_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create common derived fields if not already available."""
    df = df.copy()

    if "log10_nf" not in df.columns:
        if "log_nf" in df.columns:
            df["log10_nf"] = pd.to_numeric(df["log_nf"], errors="coerce")
        elif "cycles_to_failure" in df.columns:
            nf = pd.to_numeric(df["cycles_to_failure"], errors="coerce")
            df["log10_nf"] = np.where(nf > 0, np.log10(nf), np.nan)

    if "d_inv_sqrt" not in df.columns and "grain_size_um" in df.columns:
        grain = pd.to_numeric(df["grain_size_um"], errors="coerce")
        df["d_inv_sqrt"] = np.where(grain > 0, 1 / np.sqrt(grain), np.nan)

    if "strength_ratio" not in df.columns and {"ys_mpa", "uts_mpa"}.issubset(df.columns):
        ys = pd.to_numeric(df["ys_mpa"], errors="coerce")
        uts = pd.to_numeric(df["uts_mpa"], errors="coerce")
        df["strength_ratio"] = np.where(uts != 0, ys / uts, np.nan)

    # Route-family indicator variables can help interpret process-family effects.
    # These are not recommended as generic ML features, but they are useful in a
    # PSPP correlation heatmap.
    if "route_family" in df.columns:
        family = df["route_family"].astype(str).str.strip().str.upper()
        for fam in ["AR", "HT", "DCT", "ECAP"]:
            df[f"is_{fam.lower()}"] = (family == fam).astype(int)

    return df


def available_features(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for col in candidates:
        if col in df.columns and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def safe_corr(
    x: pd.Series,
    y: pd.Series,
    method: str = "spearman",
    min_n: int = 4,
) -> Tuple[float, float, int]:
    data = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n < min_n:
        return np.nan, np.nan, n
    xvals = data.iloc[:, 0].astype(float).values
    yvals = data.iloc[:, 1].astype(float).values
    if np.nanstd(xvals) == 0 or np.nanstd(yvals) == 0:
        return np.nan, np.nan, n
    try:
        if method == "pearson":
            r, p = pearsonr(xvals, yvals)
        elif method == "spearman":
            r, p = spearmanr(xvals, yvals)
        else:
            raise ValueError(f"Unknown method: {method}")
        return float(r), float(p), int(n)
    except Exception:
        return np.nan, np.nan, n


def pairwise_corr_matrix(
    df: pd.DataFrame,
    cols: List[str],
    method: str = "spearman",
    min_n: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    nmat = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                mat.loc[c1, c2] = 1.0
                nmat.loc[c1, c2] = int(df[c1].replace([np.inf, -np.inf], np.nan).dropna().shape[0])
            else:
                r, _p, n = safe_corr(df[c1], df[c2], method=method, min_n=min_n)
                mat.loc[c1, c2] = r
                nmat.loc[c1, c2] = n
    return mat, nmat


def plot_heatmap(
    corr: pd.DataFrame,
    output_path: Path,
    title: str,
    annotate: bool = True,
    cmap_name: str = "coolwarm",
) -> None:
    labels = corr.columns.tolist()
    values = corr.values.astype(float)
    masked_values = np.ma.masked_invalid(values)

    fig_w = max(11, 0.62 * len(labels))
    fig_h = max(9, 0.58 * len(labels))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(masked_values, vmin=-1, vmax=1, cmap=cmap_name, aspect="auto")
    plt.colorbar(im, label="Correlation coefficient")
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title(title)

    if annotate and len(labels) <= 24:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = values[i, j]
                if np.isfinite(val):
                    text_color = "white" if abs(val) > 0.65 else "black"
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=text_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = "Feature",
    n_top: int = 20,
) -> None:
    if df.empty:
        return
    plot_df = df.copy().head(n_top).iloc[::-1]
    plt.figure(figsize=(10.5, max(7, 0.36 * len(plot_df))))
    plt.barh(plot_df[y_col], plot_df[x_col])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def feature_layer_map() -> Dict[str, str]:
    process = [
        "soak_hours", "ecap_angle_deg", "sample_number", "is_ar", "is_ht", "is_dct", "is_ecap"
    ]
    structure = [
        "grain_size_um", "d_inv_sqrt", "d_inv_sqrt_db", "hardness_hv"
    ]
    property_features = [
        "ys_mpa", "uts_mpa", "elongation_percent", "strength_ratio", "strength_ratio_db"
    ]
    cyclic = [
        "psa_initial_mean", "psa_stable_mean", "psa_final_mean", "psa_mean",
        "psa_stable_std", "psa_stable_cov", "psa_slope",
        "mean_stress_initial_mean", "mean_stress_stable_mean", "mean_stress_final_mean",
        "mean_stress_mean", "mean_stress_stable_std", "mean_stress_slope",
        "stress_amplitude_initial_mean", "stress_amplitude_stable_mean", "stress_amplitude_final_mean",
        "stress_amplitude_mean", "stress_amplitude_stable_std", "stress_amplitude_slope",
        "stress_range_mean", "stress_range_stable_mean",
        "unloading_modulus_initial_mean", "unloading_modulus_stable_mean",
        "unloading_modulus_final_mean", "unloading_modulus_mean", "unloading_modulus_slope",
        "max_stress_peak", "min_stress_minimum", "max_stress_mpa", "min_stress_mpa",
        "energy_proxy_initial_mean", "energy_proxy_stable_mean", "energy_proxy_final_mean",
        "energy_proxy_mean", "energy_proxy_slope", "energy_proxy_max", "energy_proxy_stable_max",
    ]
    target = ["log10_nf", "log_nf", "cycles_to_failure"]

    fmap: Dict[str, str] = {}
    for c in process:
        fmap[c] = "Process"
    for c in structure:
        fmap[c] = "Structure"
    for c in property_features:
        fmap[c] = "Property"
    for c in cyclic:
        fmap[c] = "Performance/Cyclic"
    for c in target:
        fmap[c] = "Target/Performance"
    return fmap


LAYER_MAP = feature_layer_map()


def get_layer(feature: str) -> str:
    if feature in LAYER_MAP:
        return LAYER_MAP[feature]
    if "energy" in feature or "stress" in feature or "psa" in feature or "modulus" in feature:
        return "Performance/Cyclic"
    if "grain" in feature or "hardness" in feature or "d_inv" in feature:
        return "Structure"
    if "ys" in feature or "uts" in feature or "elongation" in feature or "strength" in feature:
        return "Property"
    if "soak" in feature or "ecap" in feature or feature.startswith("is_"):
        return "Process"
    return "Other"


def mechanistic_note(feature: str, rho: float) -> str:
    direction = "positive" if pd.notna(rho) and rho > 0 else "negative"
    if feature in ["grain_size_um"]:
        return "Smaller grains are associated with higher fatigue life" if direction == "negative" else "Unexpected grain-size direction; verify subset effects"
    if feature in ["d_inv_sqrt", "d_inv_sqrt_db"]:
        return "Grain refinement descriptor increases with fatigue life" if direction == "positive" else "Unexpected Hall-Petch descriptor direction"
    if "hardness" in feature:
        return "Higher hardness generally reflects strengthening and improved cyclic resistance"
    if feature in ["ys_mpa", "uts_mpa", "strength_ratio"]:
        return "Strength-related property associated with resistance to plastic deformation"
    if "psa" in feature:
        return "Plastic strain amplitude reflects cyclic plasticity and fatigue damage accumulation"
    if "mean_stress" in feature:
        return "Mean stress evolution influences crack initiation and cyclic damage state"
    if "stress_amplitude" in feature or "stress_range" in feature:
        return "Cyclic stress amplitude/range reflects loading response and damage severity"
    if "unloading_modulus" in feature:
        return "Unloading modulus degradation captures cyclic stiffness/damage evolution"
    if "energy_proxy" in feature:
        return "Energy proxy represents approximate cyclic plastic energy dissipation"
    if feature == "soak_hours":
        return "DCT soaking duration is route-specific and should be interpreted within DCT routes"
    if feature == "ecap_angle_deg":
        return "ECAP channel angle is route-specific and should be interpreted only within ECAP routes"
    if feature.startswith("is_"):
        return "Route-family indicator captures processing-family separation, not a universal continuous mechanism"
    return "PSPP descriptor associated with fatigue-life variation"


# =============================================================================
# 2. MAIN ANALYSIS
# =============================================================================


def main() -> None:
    print("=== START task5_correlations_85 ===")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = normalize_columns(pd.read_csv(INPUT_FILE))
    df = ensure_core_features(df)

    if "specimen_id" not in df.columns:
        raise ValueError("Missing specimen_id column in sample-level feature file.")
    if "route_id" not in df.columns:
        raise ValueError("Missing route_id column in sample-level feature file.")
    if "log10_nf" not in df.columns:
        raise ValueError("Missing log10_nf column and could not derive it from cycles_to_failure.")

    # Convert non-ID object columns safely where appropriate.
    for col in df.columns:
        if col not in ["specimen_id", "route_id", "route_family", "process_subtype", "source_group", "source_file", "generation_method"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    target = "log10_nf"

    # Selected PSPP features for manuscript-oriented heatmap. Keep this compact.
    selected_candidates = [
        # Process
        "is_ecap", "is_dct", "soak_hours", "ecap_angle_deg",
        # Structure
        "grain_size_um", "d_inv_sqrt", "hardness_hv",
        # Property
        "ys_mpa", "uts_mpa", "strength_ratio", "elongation_percent",
        # Cyclic performance descriptors
        "psa_stable_mean", "psa_initial_mean", "mean_stress_stable_mean",
        "stress_amplitude_stable_mean", "stress_range_stable_mean",
        "unloading_modulus_stable_mean", "unloading_modulus_max",
        "energy_proxy_stable_mean", "energy_proxy_max",
        "max_stress_peak", "min_stress_minimum",
        # Target
        target,
    ]
    selected_cols = available_features(df, selected_candidates)

    # All numeric features for diagnostic target ranking, excluding obvious direct target leakage.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    leakage_keywords = [
        "cycles_to_failure", "log10_nf", "log_nf", "nf_", "cycle_no", "n_cycle_rows",
        "initial_region_rows", "stable_region_rows", "final_region_rows", "cycle_fraction",
        "coverage", "rank", "fatigue_efficiency", "is_synthetic", "sample_number", "sample_no"
    ]
    ml_safe_cols = []
    excluded_rows = []
    for col in numeric_cols:
        if col == target:
            continue
        reason = ""
        if any(key in col for key in leakage_keywords):
            reason = "excluded_to_avoid_target_or_cycle_count_leakage"
        elif col.startswith("is_"):
            reason = "excluded_from_ml_safe_ranking_as_route_family_indicator"
        if reason:
            excluded_rows.append({"feature": col, "reason": reason})
        else:
            ml_safe_cols.append(col)

    # Target correlation summary for ML-safe descriptors.
    target_rows = []
    for col in ml_safe_cols:
        pr, pp, pn = safe_corr(df[col], df[target], method="pearson", min_n=5)
        sr, sp, sn = safe_corr(df[col], df[target], method="spearman", min_n=5)
        if np.isfinite(pr) or np.isfinite(sr):
            target_rows.append({
                "feature": col,
                "pspp_layer": get_layer(col),
                "n_pairwise_pearson": pn,
                "pearson_r": pr,
                "pearson_p": pp,
                "n_pairwise_spearman": sn,
                "spearman_rho": sr,
                "spearman_p": sp,
                "abs_pearson_r": abs(pr) if pd.notna(pr) else np.nan,
                "abs_spearman_rho": abs(sr) if pd.notna(sr) else np.nan,
                "mechanistic_note": mechanistic_note(col, sr if pd.notna(sr) else pr),
            })

    target_corr = pd.DataFrame(target_rows)
    if not target_corr.empty:
        target_corr["combined_abs_rank_score"] = target_corr[["abs_pearson_r", "abs_spearman_rho"]].mean(axis=1)
        target_corr = target_corr.sort_values("combined_abs_rank_score", ascending=False).reset_index(drop=True)
        target_corr.insert(0, "rank", np.arange(1, len(target_corr) + 1))

    # Route-mean correlations: route-level interpretation for manuscript.
    route_group_cols = ["route_id"]
    if "route_family" in df.columns:
        route_group_cols.append("route_family")

    route_numeric_cols = [c for c in numeric_cols if c not in ["is_synthetic"]]
    route_mean = df[route_group_cols + route_numeric_cols].groupby(route_group_cols, dropna=False).mean(numeric_only=True).reset_index()
    if target not in route_mean.columns:
        raise ValueError("Could not form route-mean target column.")

    route_rows = []
    for col in [c for c in ml_safe_cols if c in route_mean.columns]:
        pr, pp, pn = safe_corr(route_mean[col], route_mean[target], method="pearson", min_n=5)
        sr, sp, sn = safe_corr(route_mean[col], route_mean[target], method="spearman", min_n=5)
        if np.isfinite(pr) or np.isfinite(sr):
            route_rows.append({
                "feature": col,
                "pspp_layer": get_layer(col),
                "n_pairwise_pearson": pn,
                "pearson_r": pr,
                "pearson_p": pp,
                "n_pairwise_spearman": sn,
                "spearman_rho": sr,
                "spearman_p": sp,
                "abs_pearson_r": abs(pr) if pd.notna(pr) else np.nan,
                "abs_spearman_rho": abs(sr) if pd.notna(sr) else np.nan,
                "mechanistic_note": mechanistic_note(col, sr if pd.notna(sr) else pr),
            })
    route_corr = pd.DataFrame(route_rows)
    if not route_corr.empty:
        route_corr["combined_abs_rank_score"] = route_corr[["abs_pearson_r", "abs_spearman_rho"]].mean(axis=1)
        route_corr = route_corr.sort_values("combined_abs_rank_score", ascending=False).reset_index(drop=True)
        route_corr.insert(0, "rank", np.arange(1, len(route_corr) + 1))

    # PSPP layer summary from target_corr.
    if not target_corr.empty:
        layer_summary = (
            target_corr.groupby("pspp_layer")
            .agg(
                n_features=("feature", "count"),
                mean_abs_pearson=("abs_pearson_r", "mean"),
                mean_abs_spearman=("abs_spearman_rho", "mean"),
                max_abs_spearman=("abs_spearman_rho", "max"),
                top_feature_by_spearman=("feature", lambda s: target_corr.loc[s.index].sort_values("abs_spearman_rho", ascending=False)["feature"].iloc[0]),
            )
            .reset_index()
            .sort_values("mean_abs_spearman", ascending=False)
        )
    else:
        layer_summary = pd.DataFrame()

    # Selected PSPP matrices.
    pearson_matrix, pairwise_n_pearson = pairwise_corr_matrix(df, selected_cols, method="pearson", min_n=5)
    spearman_matrix, pairwise_n_spearman = pairwise_corr_matrix(df, selected_cols, method="spearman", min_n=5)

    # High correlation descriptor pairs within selected matrix.
    high_pairs = []
    for i, f1 in enumerate(selected_cols):
        for j, f2 in enumerate(selected_cols):
            if j <= i:
                continue
            rho = spearman_matrix.loc[f1, f2]
            r = pearson_matrix.loc[f1, f2]
            n_pair = pairwise_n_spearman.loc[f1, f2]
            if pd.notna(rho) and abs(rho) >= 0.70:
                high_pairs.append({
                    "feature_1": f1,
                    "layer_1": get_layer(f1),
                    "feature_2": f2,
                    "layer_2": get_layer(f2),
                    "spearman_rho": rho,
                    "pearson_r": r,
                    "abs_spearman_rho": abs(rho),
                    "n_pairwise": int(n_pair),
                    "interpretation_note": f"High association between {get_layer(f1)} and {get_layer(f2)} descriptors",
                })
    high_pairs_df = pd.DataFrame(high_pairs)
    if not high_pairs_df.empty:
        high_pairs_df = high_pairs_df.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)

    # Feature dictionary.
    dictionary_rows = []
    for col in selected_cols:
        dictionary_rows.append({
            "feature": col,
            "pspp_layer": get_layer(col),
            "available_non_missing_rows": int(df[col].replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
            "included_in_selected_heatmap": True,
            "note": mechanistic_note(col, np.nan),
        })
    feature_dictionary = pd.DataFrame(dictionary_rows)

    # Validation overview.
    route_counts = df.groupby("route_id")["specimen_id"].nunique() if "route_id" in df.columns else pd.Series(dtype=int)
    validations = [
        {"check": "Total sample rows", "expected": EXPECTED_SAMPLES, "observed": len(df), "status": "PASS" if len(df) == EXPECTED_SAMPLES else "FAIL"},
        {"check": "Unique specimen_id count", "expected": EXPECTED_SAMPLES, "observed": df["specimen_id"].nunique(), "status": "PASS" if df["specimen_id"].nunique() == EXPECTED_SAMPLES else "FAIL"},
        {"check": "Unique route_id count", "expected": EXPECTED_ROUTES, "observed": df["route_id"].nunique(), "status": "PASS" if df["route_id"].nunique() == EXPECTED_ROUTES else "FAIL"},
        {"check": "Routes with exactly five samples", "expected": EXPECTED_ROUTES, "observed": int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()), "status": "PASS" if int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES else "FAIL"},
        {"check": "Missing target log10_nf", "expected": 0, "observed": int(df[target].isna().sum()), "status": "PASS" if int(df[target].isna().sum()) == 0 else "FAIL"},
        {"check": "Selected PSPP heatmap features", "expected": ">=10", "observed": len(selected_cols), "status": "PASS" if len(selected_cols) >= 10 else "FAIL"},
        {"check": "ML-safe target-ranked features", "expected": ">=10", "observed": len(target_corr), "status": "PASS" if len(target_corr) >= 10 else "FAIL"},
        {"check": "Route-mean target-ranked features", "expected": ">=10", "observed": len(route_corr), "status": "PASS" if len(route_corr) >= 10 else "FAIL"},
    ]
    validation_df = pd.DataFrame(validations)
    final_status = "PASS" if (validation_df["status"] == "PASS").all() else "CHECK_REQUIRED"

    # Output paths.
    out_report = STATS_DIR / "task5_correlations_report_85.txt"
    out_pearson_matrix = STATS_DIR / "task5_pspp_selected_correlation_matrix_pearson_85.csv"
    out_spearman_matrix = STATS_DIR / "task5_pspp_selected_correlation_matrix_spearman_85.csv"
    out_pairwise_n_pearson = STATS_DIR / "task5_pspp_selected_pairwise_n_pearson_85.csv"
    out_pairwise_n_spearman = STATS_DIR / "task5_pspp_selected_pairwise_n_spearman_85.csv"
    out_target = STATS_DIR / "task5_target_correlation_summary_ml_safe_85.csv"
    out_route_target = STATS_DIR / "task5_route_mean_target_correlation_summary_85.csv"
    out_high_pairs = STATS_DIR / "task5_high_correlation_pairs_85.csv"
    out_layer = STATS_DIR / "task5_pspp_layer_correlation_summary_85.csv"
    out_dictionary = STATS_DIR / "task5_correlation_feature_dictionary_85.csv"
    out_exclusion = STATS_DIR / "task5_correlation_exclusion_log_85.csv"
    out_validation = STATS_DIR / "task5_correlation_validation_overview_85.csv"
    out_metadata = STATS_DIR / "task5_correlation_metadata_85.json"

    # Save CSVs.
    pearson_matrix.to_csv(out_pearson_matrix)
    spearman_matrix.to_csv(out_spearman_matrix)
    pairwise_n_pearson.to_csv(out_pairwise_n_pearson)
    pairwise_n_spearman.to_csv(out_pairwise_n_spearman)
    target_corr.to_csv(out_target, index=False)
    route_corr.to_csv(out_route_target, index=False)
    high_pairs_df.to_csv(out_high_pairs, index=False)
    layer_summary.to_csv(out_layer, index=False)
    feature_dictionary.to_csv(out_dictionary, index=False)
    pd.DataFrame(excluded_rows).to_csv(out_exclusion, index=False)
    validation_df.to_csv(out_validation, index=False)

    metadata = {
        "script": "task5_correlations_85.py",
        "input_file": str(INPUT_FILE),
        "n_rows": int(len(df)),
        "n_samples": int(df["specimen_id"].nunique()),
        "n_routes": int(df["route_id"].nunique()),
        "target": target,
        "selected_heatmap_features": selected_cols,
        "n_ml_safe_features_ranked": int(len(target_corr)),
        "n_route_mean_features_ranked": int(len(route_corr)),
        "method_note": "Pearson and Spearman correlations calculated using sample-level data; route-mean correlations calculated after grouping by route_id.",
        "leakage_note": "Direct target, log target, cycle-count, and coverage variables excluded from ML-safe target ranking.",
        "final_status": final_status,
    }
    with open(out_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Figures.
    fig_spear_heatmap = FIG_DIR / "task5_pspp_spearman_heatmap_85.png"
    fig_pear_heatmap = FIG_DIR / "task5_pspp_pearson_heatmap_85.png"
    fig_target_spear = FIG_DIR / "task5_target_top20_ml_safe_spearman_85.png"
    fig_target_pear = FIG_DIR / "task5_target_top20_ml_safe_pearson_85.png"
    fig_route_spear = FIG_DIR / "task5_route_mean_target_top20_spearman_85.png"
    fig_layer = FIG_DIR / "task5_pspp_layer_mean_abs_spearman_85.png"

    plot_heatmap(spearman_matrix, fig_spear_heatmap, "PSPP Spearman correlation heatmap for 85-sample dataset")
    plot_heatmap(pearson_matrix, fig_pear_heatmap, "PSPP Pearson correlation heatmap for 85-sample dataset")

    if not target_corr.empty:
        plot_bar(
            target_corr.sort_values("abs_spearman_rho", ascending=False),
            "abs_spearman_rho",
            "feature",
            fig_target_spear,
            "Top 20 ML-safe features by absolute Spearman correlation with log10(Nf)",
            "Absolute Spearman correlation",
        )
        plot_bar(
            target_corr.sort_values("abs_pearson_r", ascending=False),
            "abs_pearson_r",
            "feature",
            fig_target_pear,
            "Top 20 ML-safe features by absolute Pearson correlation with log10(Nf)",
            "Absolute Pearson correlation",
        )

    if not route_corr.empty:
        plot_bar(
            route_corr.sort_values("abs_spearman_rho", ascending=False),
            "abs_spearman_rho",
            "feature",
            fig_route_spear,
            "Route-mean top 20 features by absolute Spearman correlation with log10(Nf)",
            "Absolute Spearman correlation",
        )

    if not layer_summary.empty:
        plot_layer = layer_summary.sort_values("mean_abs_spearman", ascending=True)
        plt.figure(figsize=(8.5, max(4.8, 0.5 * len(plot_layer))))
        plt.barh(plot_layer["pspp_layer"], plot_layer["mean_abs_spearman"])
        plt.xlabel("Mean absolute Spearman correlation with log10(Nf)")
        plt.ylabel("PSPP layer")
        plt.title("PSPP layer-level fatigue-life association")
        plt.tight_layout()
        plt.savefig(fig_layer, dpi=300)
        plt.close()

    # Text report.
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("task5_correlations_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique()}\n")
        f.write(f"Unique routes: {df['route_id'].nunique()}\n")
        if "route_family" in df.columns:
            families = ", ".join(sorted(df["route_family"].dropna().astype(str).unique()))
            f.write(f"Route families: {families}\n")
        f.write(f"Target variable: {target}\n")
        f.write(f"Selected PSPP heatmap features: {len(selected_cols)}\n")
        f.write(f"ML-safe target-ranked features: {len(target_corr)}\n")
        f.write(f"Route-mean target-ranked features: {len(route_corr)}\n\n")

        f.write("Validation overview\n")
        f.write("-" * 78 + "\n")
        f.write(validation_df.to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 ML-safe features by combined target-correlation score\n")
        f.write("-" * 78 + "\n")
        if target_corr.empty:
            f.write("No target correlations calculated.\n")
        else:
            preview_cols = ["rank", "feature", "pspp_layer", "n_pairwise_spearman", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "mechanistic_note"]
            f.write(target_corr[preview_cols].head(20).to_string(index=False))
        f.write("\n\n")

        f.write("Top 20 route-mean features by combined target-correlation score\n")
        f.write("-" * 78 + "\n")
        if route_corr.empty:
            f.write("No route-mean target correlations calculated.\n")
        else:
            preview_cols = ["rank", "feature", "pspp_layer", "n_pairwise_spearman", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "mechanistic_note"]
            f.write(route_corr[preview_cols].head(20).to_string(index=False))
        f.write("\n\n")

        f.write("PSPP layer summary\n")
        f.write("-" * 78 + "\n")
        if layer_summary.empty:
            f.write("No layer summary calculated.\n")
        else:
            f.write(layer_summary.to_string(index=False))
        f.write("\n\n")

        f.write("High selected-feature Spearman pairs |rho| >= 0.70\n")
        f.write("-" * 78 + "\n")
        if high_pairs_df.empty:
            f.write("No high-correlation pairs above threshold.\n")
        else:
            f.write(high_pairs_df.head(30).to_string(index=False))
        f.write("\n\n")

        f.write("Generated files\n")
        f.write("-" * 78 + "\n")
        for p in [
            out_pearson_matrix, out_spearman_matrix, out_pairwise_n_pearson, out_pairwise_n_spearman,
            out_target, out_route_target, out_high_pairs, out_layer, out_dictionary, out_exclusion,
            out_validation, out_metadata, fig_spear_heatmap, fig_pear_heatmap, fig_target_spear,
            fig_target_pear, fig_route_spear, fig_layer,
        ]:
            f.write(f"  {p}\n")
        f.write("\n")

        f.write("Interpretation note\n")
        f.write("-" * 78 + "\n")
        f.write(
            "This script creates a PSPP-oriented correlation layer for the 85-row sample-level dataset. "
            "Spearman correlation is emphasized for manuscript interpretation because fatigue-life data are route-clustered and globally non-normal. "
            "Direct fatigue-life, log-fatigue-life, cycle-count and cycle-coverage variables are excluded from ML-safe ranking to avoid target leakage. "
            "Process-specific variables such as ECAP angle and DCT soaking time should be interpreted within their applicable route families rather than as universal continuous predictors.\n\n"
        )
        f.write(f"Final status: {final_status}\n")

    print("\nValidation overview:")
    print(validation_df.to_string(index=False))
    print("\nTop ML-safe target correlations:")
    if not target_corr.empty:
        print(target_corr[["rank", "feature", "pspp_layer", "pearson_r", "spearman_rho", "combined_abs_rank_score"]].head(15).to_string(index=False))
    print(f"\nSaved report to: {out_report}")
    print(f"✅ Done task5_correlations_85. Status: {final_status}")


if __name__ == "__main__":
    main()
