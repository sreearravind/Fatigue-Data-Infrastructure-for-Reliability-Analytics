# Fatigue Data Infrastructure for Reliability Analytics: Anonymous Review Repository

## 1. Purpose of this repository

This repository supports anonymous peer-review evaluation of a fatigue data infrastructure and reproducibility workflow. It provides the scripts, processed data products, manuscript-supporting outputs, and selected figures needed to verify the 85-sample fatigue DBMS workflow in a anonymous review setting.

## 2. Repository contents

The repository includes Python scripts, processed datasets, statistical outputs, reliability outputs, ML outputs, ICME/FAIR metadata outputs, a file manifest, and execution guidance are also included.

## 3. Dataset overview

The workflow contains an expanded 85-sample-level fatigue dataset across 17 processing routes. The sample-level records should be interpreted as augmented and validated sample-level records unless independently confirmed as fully experimental.

## 4. Workflow overview

The data are organized according to a Process-Structure-Property-Performance framework. The workflow supports reproducibility of descriptive statistics, bootstrap confidence intervals, Weibull reliability analysis, Hall-Petch validation, grain-size fatigue correlation, PSPP correlation mapping, route-aware ML validation, and manuscript output compilation.

## 5. Reproducibility instructions

Reviewers can validate the repository in two ways:

1. Run the Python scripts in the recommended order described in `REPRODUCIBILITY_GUIDE.md`.
2. Verify manuscript-supporting outputs directly from the included CSV, TXT, JSON, MD, and selected figure files without rebuilding the database.

## 6. Folder structure

- `scripts/`: workflow scripts grouped by stage
- `data/`: input, cleaned, database, statistics, ML, and manuscript-output files
- `figures/`: selected main and supplementary figures

## 7. Main scripts and execution order

1. `scripts/00_database_setup/`
2. `scripts/01_data_preparation/`
3. `scripts/02_statistical_analysis/`
4. `scripts/03_reliability_analysis/`
5. `scripts/04_microstructure_pspp/`
6. `scripts/05_machine_learning/`
7. `scripts/06_icme_fair_metadata/`

## 8. Main output files

Representative outputs include:

- `data/02_cleaned/sample_level_features_85.csv`
- `data/04_statistics_outputs/task3_1_global_descriptive_stats_85.csv`
- `data/04_statistics_outputs/task3_5B_weibull_global_parameters_85.csv`
- `data/05_ml_outputs/task3_4_ml_model_performance_85.csv`
- `data/07_manuscript_outputs/task6_manuscript_key_results_summary_85.md`
- `docs/workflow_reference/WORKFLOW_FILE_USAGE_REFERENCE_85.md`

## 9. Notes on expanded sample-level dataset

The repository contains an expanded and validated 85-sample-level fatigue dataset across 17 processing routes. This representation is intended for reproducibility review, statistical characterization, reliability analysis, PSPP linkage, and route-aware validation.
