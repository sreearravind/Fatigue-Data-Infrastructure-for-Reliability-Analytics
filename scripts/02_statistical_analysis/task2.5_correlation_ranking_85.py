"""
task2.5_correlation_ranking_85.py

Purpose
-------
Rank PSPP-aligned sample-level descriptors against fatigue life for the
85-sample fatigue DBMS workflow.

Input:
    Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv

Main outputs:
    task2_5_feature_correlation_ranking_ml_safe_85.csv
    task2_5_feature_correlation_ranking_all_numeric_diagnostic_85.csv
    task2_5_route_mean_correlation_ranking_85.csv
    task2_5_feature_exclusion_log_85.csv
    task2_5_correlation_ranking_report_85.txt

Important:
    The ML-safe ranking excludes target-leakage features such as fatigue life,
    log(Nf), cycle counts, cycle fractions, route/sample identifiers, and
    generated/synthetic tracking columns.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

EXPECTED_SAMPLES = 85
EXPECTED_ROUTES = 17
EXPECTED_SAMPLES_PER_ROUTE = 5


def find_dataset_base_dir() -> Path:
    """Find the Fatigue_85_augmented_dataset folder robustly."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "Fatigue_85_augmented_dataset",
        script_dir / "Fatigue_85_augmented_dataset",
        Path.cwd().parent / "Fatigue_85_augmented_dataset",
        Path.cwd() / "Fatigue_85_augmented_dataset",
        Path(r"data"),
    ]
    for candidate in candidates:
        if (candidate / "02_cleaned" / "sample_level_features_85.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Fatigue_85_augmented_dataset/02_cleaned/sample_level_features_85.csv. "
        "Place this script in db_scripts_85 or update the hardcoded path."
    )


BASE_DIR = find_dataset_base_dir()
CLEANED_DIR = BASE_DIR / "02_cleaned"
STAT_DIR = BASE_DIR / "04_statistics_outputs"
FIG_DIR = BASE_DIR / "06_figures"
INPUT_FILE = CLEANED_DIR / "sample_level_features_85.csv"


def ensure_dirs() -> None:
    STAT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index(df.columns).astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    return df


def find_target_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Find or create a clean log10 fatigue-life target."""
    df = df.copy()
    candidates = ["log10_nf", "log_nf", "log_nf_db", "log10Nf", "log10nf", "logNf", "lognf"]
    for col in candidates:
        if col in df.columns:
            df["target_log10_nf"] = pd.to_numeric(df[col], errors="coerce")
            return df, "target_log10_nf"

    if "cycles_to_failure" not in df.columns:
        raise ValueError("No log10 fatigue-life target found and cycles_to_failure is missing.")

    nf = pd.to_numeric(df["cycles_to_failure"], errors="coerce")
    if (nf <= 0).any():
        raise ValueError("cycles_to_failure contains non-positive values.")
    df["target_log10_nf"] = np.log10(nf)
    return df, "target_log10_nf"


def classify_pspp_layer(feature: str) -> str:
    f = feature.lower()
    if any(k in f for k in ["soak", "ecap_angle", "tsa", "frequency", "temperature", "process"]):
        return "Process"
    if any(k in f for k in ["grain", "d_inv", "d_minus", "microstructure"]):
        return "Structure"
    if any(k in f for k in ["ys", "uts", "hardness", "elongation", "strength_ratio"]):
        return "Property"
    if any(k in f for k in [
        "psa", "stress", "modulus", "energy", "plastic", "amplitude",
        "slope", "stable", "initial", "final", "peak", "minimum", "mean_stress"
    ]):
        return "Performance/Cyclic"
    if any(k in f for k in ["cycle", "row", "coverage", "sample", "synthetic", "route"]):
        return "Metadata/Diagnostic"
    return "Other"


def leakage_reason(feature: str) -> str:
    """Return exclusion reason for ML-safe ranking, or empty string if safe."""
    f = feature.lower()
    exact_exclude = {
        "target_log10_nf", "cycles_to_failure", "log10_nf", "log_nf", "log_nf_db",
        "log10nf", "lognf", "nf", "sample_no", "sample_number", "is_synthetic",
    }
    if f in exact_exclude:
        return "target/leakage or sample-generation metadata"

    if any(k in f for k in [
        "cycles_to_failure", "log10", "log_nf", "lognf", "fatigue_efficiency",
        "cycle_no", "n_cycle", "cycle_fraction", "region_rows", "coverage_ratio",
        "stable_region_rows", "initial_region_rows", "final_region_rows", "rows", "rank",
    ]):
        return "directly derived from fatigue life/cycle count"

    if any(k in f for k in ["specimen_id", "sample_id", "route_id", "source", "file", "generation_method"]):
        return "identifier/source metadata"

    return ""


def safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def benjamini_hochberg(p_values: Iterable[float]) -> List[float]:
    p = np.asarray([np.nan if v is None else float(v) for v in p_values], dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return q.tolist()
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = prev
    out_valid = np.empty(m, dtype=float)
    out_valid[order] = np.minimum(adjusted, 1.0)
    q[valid] = out_valid
    return q.tolist()


def compute_correlation_table(df: pd.DataFrame, target_col: str, feature_cols: List[str], analysis_level: str) -> pd.DataFrame:
    rows = []
    y_all = safe_numeric_series(df, target_col)
    for feature in feature_cols:
        x_all = safe_numeric_series(df, feature)
        pair = pd.DataFrame({"x": x_all, "y": y_all}).dropna()
        n = len(pair)
        base = {
            "analysis_level": analysis_level,
            "feature": feature,
            "pspp_layer": classify_pspp_layer(feature),
            "n_pairwise": n,
        }
        if n < 3:
            rows.append({**base, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
                         "abs_pearson_r": np.nan, "abs_spearman_rho": np.nan, "combined_abs_score": np.nan,
                         "direction": "insufficient data", "note": "n < 3"})
            continue
        if pair["x"].nunique(dropna=True) < 2:
            rows.append({**base, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan,
                         "abs_pearson_r": np.nan, "abs_spearman_rho": np.nan, "combined_abs_score": np.nan,
                         "direction": "constant feature", "note": "constant feature"})
            continue
        try:
            pr, pp = pearsonr(pair["x"], pair["y"])
        except Exception:
            pr, pp = np.nan, np.nan
        try:
            sr, sp = spearmanr(pair["x"], pair["y"])
        except Exception:
            sr, sp = np.nan, np.nan
        direction = "positive association with log10(Nf)" if pr > 0 else "negative association with log10(Nf)"
        if np.isnan(pr):
            direction = "not estimated"
        rows.append({**base, "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp,
                     "abs_pearson_r": abs(pr) if not np.isnan(pr) else np.nan,
                     "abs_spearman_rho": abs(sr) if not np.isnan(sr) else np.nan,
                     "combined_abs_score": np.nanmean([abs(pr), abs(sr)]),
                     "direction": direction, "note": ""})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pearson_q_bh"] = benjamini_hochberg(out["pearson_p"])
        out["spearman_q_bh"] = benjamini_hochberg(out["spearman_p"])
        out = out.sort_values(["combined_abs_score", "abs_spearman_rho", "abs_pearson_r"], ascending=False, na_position="last").reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_feature_exclusion_log(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        reason = "target variable" if col == target_col else leakage_reason(col)
        rows.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "pspp_layer": classify_pspp_layer(col),
            "included_in_ml_safe_ranking": reason == "",
            "exclusion_reason": reason,
            "n_missing": int(pd.to_numeric(df[col], errors="coerce").isna().sum()),
            "n_unique": int(pd.to_numeric(df[col], errors="coerce").nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values(["included_in_ml_safe_ranking", "feature"], ascending=[True, True])


def route_mean_dataframe(df: pd.DataFrame, target_col: str, safe_features: List[str]) -> pd.DataFrame:
    group_cols = ["route_id"]
    if "route_family" in df.columns:
        group_cols.append("route_family")
    agg_cols = [target_col] + [c for c in safe_features if c in df.columns]
    route_mean = df[group_cols + agg_cols].groupby(group_cols, as_index=False).mean(numeric_only=True)
    n_df = df.groupby("route_id").size().reset_index(name="n_samples")
    return route_mean.merge(n_df, on="route_id", how="left")


def pssp_layer_summary(corr_df: pd.DataFrame) -> pd.DataFrame:
    if corr_df.empty:
        return pd.DataFrame()
    rows = []
    for layer, sub in corr_df.dropna(subset=["combined_abs_score"]).groupby("pspp_layer"):
        top = sub.sort_values("combined_abs_score", ascending=False).iloc[0]
        rows.append({
            "pspp_layer": layer,
            "n_features": len(sub),
            "mean_abs_pearson_r": sub["abs_pearson_r"].mean(),
            "mean_abs_spearman_rho": sub["abs_spearman_rho"].mean(),
            "max_abs_pearson_r": sub["abs_pearson_r"].max(),
            "max_abs_spearman_rho": sub["abs_spearman_rho"].max(),
            "top_feature_by_combined_score": top["feature"],
        })
    return pd.DataFrame(rows).sort_values("mean_abs_spearman_rho", ascending=False)


def make_bar_plot(df: pd.DataFrame, value_col: str, title: str, xlabel: str, output_path: Path, top_n: int = 20) -> None:
    plot_df = df.dropna(subset=[value_col]).head(top_n).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.iloc[::-1]
    plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"].astype(str), plot_df[value_col].astype(float))
    plt.xlabel(xlabel)
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def make_layer_plot(layer_df: pd.DataFrame, output_path: Path) -> None:
    if layer_df.empty:
        return
    plot_df = layer_df.sort_values("mean_abs_spearman_rho", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["pspp_layer"], plot_df["mean_abs_spearman_rho"])
    plt.xlabel("Mean absolute Spearman correlation with log10(Nf)")
    plt.ylabel("PSPP layer")
    plt.title("PSPP layer-wise average fatigue-life association")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def validation_overview(df: pd.DataFrame, safe_df: pd.DataFrame, route_df: pd.DataFrame) -> pd.DataFrame:
    route_counts = df.groupby("route_id").size() if "route_id" in df.columns else pd.Series(dtype=int)
    checks = [
        ["Total sample rows", EXPECTED_SAMPLES, len(df), len(df) == EXPECTED_SAMPLES],
        ["Unique specimen_id count", EXPECTED_SAMPLES, int(df["specimen_id"].nunique()) if "specimen_id" in df.columns else "missing", "specimen_id" in df.columns and df["specimen_id"].nunique() == EXPECTED_SAMPLES],
        ["Unique route_id count", EXPECTED_ROUTES, int(df["route_id"].nunique()) if "route_id" in df.columns else "missing", "route_id" in df.columns and df["route_id"].nunique() == EXPECTED_ROUTES],
        ["Routes with exactly five samples", EXPECTED_ROUTES, int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()) if len(route_counts) else 0, len(route_counts) and int((route_counts == EXPECTED_SAMPLES_PER_ROUTE).sum()) == EXPECTED_ROUTES],
        ["ML-safe ranked numeric features", ">= 5", int(safe_df["feature"].nunique()) if not safe_df.empty else 0, not safe_df.empty and safe_df["feature"].nunique() >= 5],
        ["Route-mean ranked numeric features", ">= 5", int(route_df["feature"].nunique()) if not route_df.empty else 0, not route_df.empty and route_df["feature"].nunique() >= 5],
        ["Missing target_log10_nf", 0, int(df["target_log10_nf"].isna().sum()) if "target_log10_nf" in df.columns else "missing", "target_log10_nf" in df.columns and df["target_log10_nf"].isna().sum() == 0],
    ]
    return pd.DataFrame([{"check": c, "expected": e, "observed": o, "status": "PASS" if ok else "FAIL"} for c, e, o, ok in checks])


def write_report(report_path: Path, df: pd.DataFrame, validation_df: pd.DataFrame, safe_corr: pd.DataFrame,
                 all_corr: pd.DataFrame, route_corr: pd.DataFrame, layer_df: pd.DataFrame,
                 exclusion_df: pd.DataFrame) -> None:
    final_status = "PASS" if (validation_df["status"] == "PASS").all() else "CHECK"
    cols = ["rank", "feature", "pspp_layer", "n_pairwise", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "combined_abs_score", "direction"]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("task2.5_correlation_ranking_85 report\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Unique samples: {df['specimen_id'].nunique() if 'specimen_id' in df.columns else 'NA'}\n")
        f.write(f"Unique routes: {df['route_id'].nunique() if 'route_id' in df.columns else 'NA'}\n")
        if "route_family" in df.columns:
            f.write(f"Route families: {', '.join(sorted(map(str, df['route_family'].dropna().unique())))}\n")
        f.write("\nValidation overview\n" + "-" * 78 + "\n")
        f.write(validation_df.to_string(index=False) + "\n\n")
        f.write("Top 20 ML-safe features by combined Pearson/Spearman ranking\n" + "-" * 78 + "\n")
        f.write(safe_corr[cols].head(20).to_string(index=False) + "\n\n")
        f.write("Top 20 route-mean features by combined Pearson/Spearman ranking\n" + "-" * 78 + "\n")
        f.write(route_corr[cols].head(20).to_string(index=False) + "\n\n")
        f.write("Top 15 all-numeric diagnostic ranking, including leakage-prone features\n" + "-" * 78 + "\n")
        f.write(all_corr[cols].head(15).to_string(index=False) + "\n\n")
        f.write("PSPP layer association summary using ML-safe sample-level features\n" + "-" * 78 + "\n")
        f.write((layer_df.to_string(index=False) if not layer_df.empty else "No layer summary generated.") + "\n\n")
        f.write("Leakage prevention summary\n" + "-" * 78 + "\n")
        f.write(f"Included ML-safe numeric features: {int(exclusion_df['included_in_ml_safe_ranking'].sum())}\n")
        f.write(f"Excluded numeric features: {int((~exclusion_df['included_in_ml_safe_ranking']).sum())}\n")
        f.write("Excluded groups include target variables, cycle-count/coverage variables, logNf-derived fatigue-efficiency variables, identifiers, and synthetic/source metadata.\n\n")
        f.write("Generated files\n" + "-" * 78 + "\n")
        for p in [
            STAT_DIR / "task2_5_feature_correlation_ranking_ml_safe_85.csv",
            STAT_DIR / "task2_5_feature_correlation_ranking_all_numeric_diagnostic_85.csv",
            STAT_DIR / "task2_5_route_mean_correlation_ranking_85.csv",
            STAT_DIR / "task2_5_pspp_layer_correlation_summary_85.csv",
            STAT_DIR / "task2_5_feature_exclusion_log_85.csv",
            STAT_DIR / "task2_5_top20_spearman_ranking_85.csv",
            STAT_DIR / "task2_5_correlation_validation_overview_85.csv",
            FIG_DIR / "task2_5_top20_ml_safe_spearman_85.png",
            FIG_DIR / "task2_5_top20_ml_safe_pearson_85.png",
            FIG_DIR / "task2_5_route_mean_top20_spearman_85.png",
            FIG_DIR / "task2_5_pspp_layer_mean_abs_spearman_85.png",
        ]:
            f.write(f"  {p}\n")
        f.write("\nInterpretation note\n" + "-" * 78 + "\n")
        f.write("This script ranks correlations at the 85-row sample level and the 17-route mean level. The ML-safe ranking excludes target-leakage features before model development. Correlation ranking is for descriptor screening and mechanistic interpretation, not causal proof. Route-mean ranking is useful for manuscript-level PSPP interpretation, while sample-level ranking helps prepare the ML feature set.\n\n")
        f.write(f"Final status: {final_status}\n")


def main() -> None:
    print("=== START task2.5_correlation_ranking_85 ===")
    ensure_dirs()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = normalize_column_names(df)
    df, target_col = find_target_column(df)

    print(f"Loaded rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Target column used: {target_col}")

    if "route_id" not in df.columns:
        raise ValueError("route_id column is required for route-mean ranking.")
    if "specimen_id" not in df.columns:
        raise ValueError("specimen_id column is required for validation.")

    exclusion_df = build_feature_exclusion_log(df, target_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = [c for c in numeric_cols if c != target_col]
    safe_features = exclusion_df.loc[exclusion_df["included_in_ml_safe_ranking"], "feature"].tolist()
    safe_features = [c for c in safe_features if c != target_col and c in df.columns]

    print(f"All numeric diagnostic features: {len(all_features)}")
    print(f"ML-safe numeric features: {len(safe_features)}")

    all_corr = compute_correlation_table(df, target_col, all_features, "sample_level_85_all_numeric_diagnostic")
    safe_corr = compute_correlation_table(df, target_col, safe_features, "sample_level_85_ml_safe")

    route_mean = route_mean_dataframe(df, target_col, safe_features)
    route_features = [c for c in safe_features if c in route_mean.columns and c not in ["n_samples", target_col]]
    route_corr = compute_correlation_table(route_mean, target_col, route_features, "route_mean_17_ml_safe")
    layer_df = pssp_layer_summary(safe_corr)
    validation_df = validation_overview(df, safe_corr, route_corr)

    safe_corr.to_csv(STAT_DIR / "task2_5_feature_correlation_ranking_ml_safe_85.csv", index=False)
    all_corr.to_csv(STAT_DIR / "task2_5_feature_correlation_ranking_all_numeric_diagnostic_85.csv", index=False)
    route_corr.to_csv(STAT_DIR / "task2_5_route_mean_correlation_ranking_85.csv", index=False)
    layer_df.to_csv(STAT_DIR / "task2_5_pspp_layer_correlation_summary_85.csv", index=False)
    exclusion_df.to_csv(STAT_DIR / "task2_5_feature_exclusion_log_85.csv", index=False)
    validation_df.to_csv(STAT_DIR / "task2_5_correlation_validation_overview_85.csv", index=False)

    top20_spearman = safe_corr.dropna(subset=["abs_spearman_rho"]).sort_values(["abs_spearman_rho", "abs_pearson_r"], ascending=False).head(20).reset_index(drop=True)
    top20_spearman.insert(0, "spearman_rank", np.arange(1, len(top20_spearman) + 1))
    top20_spearman.to_csv(STAT_DIR / "task2_5_top20_spearman_ranking_85.csv", index=False)

    make_bar_plot(safe_corr.sort_values("abs_spearman_rho", ascending=False), "abs_spearman_rho", "Top 20 ML-safe features by absolute Spearman correlation with log10(Nf)", "Absolute Spearman correlation", FIG_DIR / "task2_5_top20_ml_safe_spearman_85.png")
    make_bar_plot(safe_corr.sort_values("abs_pearson_r", ascending=False), "abs_pearson_r", "Top 20 ML-safe features by absolute Pearson correlation with log10(Nf)", "Absolute Pearson correlation", FIG_DIR / "task2_5_top20_ml_safe_pearson_85.png")
    make_bar_plot(route_corr.sort_values("abs_spearman_rho", ascending=False), "abs_spearman_rho", "Route-mean top 20 features by absolute Spearman correlation with log10(Nf)", "Absolute Spearman correlation", FIG_DIR / "task2_5_route_mean_top20_spearman_85.png")
    make_layer_plot(layer_df, FIG_DIR / "task2_5_pspp_layer_mean_abs_spearman_85.png")

    metadata = {
        "script": "task2.5_correlation_ranking_85.py",
        "input_file": str(INPUT_FILE),
        "n_rows": int(len(df)),
        "n_routes": int(df["route_id"].nunique()),
        "n_ml_safe_features": int(len(safe_features)),
        "n_all_numeric_features": int(len(all_features)),
        "target": "log10(Nf)",
        "leakage_prevention": True,
    }
    with open(STAT_DIR / "task2_5_correlation_ranking_metadata_85.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report_path = STAT_DIR / "task2_5_correlation_ranking_report_85.txt"
    write_report(report_path, df, validation_df, safe_corr, all_corr, route_corr, layer_df, exclusion_df)

    print("\nTop 10 ML-safe sample-level features:")
    print(safe_corr[["rank", "feature", "pspp_layer", "pearson_r", "pearson_p", "spearman_rho", "spearman_p", "combined_abs_score"]].head(10).to_string(index=False))
    print("\nValidation overview:")
    print(validation_df.to_string(index=False))
    final_status = "PASS" if (validation_df["status"] == "PASS").all() else "CHECK"
    print(f"\nSaved report to: {report_path}")
    print(f"✅ Done task2.5_correlation_ranking_85. Status: {final_status}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

