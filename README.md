# Fatigue Data Infrastructure for Reliability Analytics

## Overview

This public repository accompanies the article **“Fatigue Data Infrastructure for Reliability Analytics: A Relational Database Framework for Traceable Fatigue Knowledge Integration.”** It provides the processed datasets, database schema resources, Python-based Extract–Transform–Load (ETL) scripts, SQL workflows, validation outputs, statistical and reliability analyses, Process–Structure–Property–Performance (PSPP) mappings, machine-learning results, and manuscript-supporting files used in the study.

The repository is intended to support transparent inspection, reproducibility, reuse, and future extension of the fatigue data infrastructure.

## Associated article

- **Article title:** Fatigue Data Infrastructure for Reliability Analytics: A Relational Database Framework for Traceable Fatigue Knowledge Integration
- **Journal:** Journal of Materials Engineering and Performance
- **Article type:** Original Research Article
- **Publication status:** Accepted for publication
- **DOI:** To be added after publication

## Dataset overview

The computational workflow contains an **expanded sample-level fatigue dataset comprising 85 records across 17 thermomechanical processing routes**, together with **159,967 cycle-level records** derived from fatigue-test histories.

The processing space includes:

- As-received Al 6063
- Heat-treated conditions
- Deep cryogenic treatment routes
- Equal Channel Angular Pressing routes

### Important dataset note

The 85-row sample-level dataset is an **experimentally anchored expanded dataset** developed to support database validation, statistical characterization, reliability analysis, PSPP mapping, and route-aware machine-learning assessment. It should not be interpreted as 85 fully independent experimental fatigue tests. The repository retains the source, route, and validation information required to distinguish experimentally derived records from expanded records.

## Technical framework

The data architecture follows the Process–Structure–Property–Performance hierarchy:

**Processing → Structure → Property → Cyclic Response → Fatigue Performance**

The workflow supports:

- Raw fatigue-data ingestion and identifier normalization
- Sample-level and cycle-level data consolidation
- Relational database construction
- Primary-key and foreign-key validation
- Stabilized cyclic descriptor extraction
- Descriptive statistical analysis
- Log-transformed fatigue-life analysis
- Bootstrap-based uncertainty quantification
- Two-parameter Weibull reliability modelling
- Survival-probability and B-life estimation
- Hall–Petch structure–property validation
- Grain-size–fatigue-life correlation
- PSPP correlation mapping
- SQL-based analytical reconstruction
- Route-aware Ridge-regression validation
- FAIR-aligned metadata generation
- Manuscript table and figure compilation

## Repository contents

The repository contains:

- Python scripts for database setup, ETL processing, statistical analysis, reliability modelling, PSPP analysis, machine learning, and metadata generation
- Processed Excel and CSV datasets
- SQL schema and database-loading resources
- Validation reports and output manifests
- Statistical and probabilistic reliability outputs
- Machine-learning performance and coefficient outputs
- ICME- and FAIR-aligned metadata files
- Selected manuscript and supplementary figures
- Reproducibility and workflow-reference documents
- Supplementary Tables S1–S12 in a single Excel workbook

## Folder structure

- `scripts/` — Python scripts grouped by workflow stage
- `data/` — source, cleaned, database-ready, statistical, reliability, ML, and manuscript-output files
- `figures/` — selected main-text and supplementary figures
- `docs/` — reproducibility guidance, workflow references, manifests, and supporting documentation

## Main workflow stages

1. `scripts/00_database_setup/`
2. `scripts/02_statistical_analysis/`
3. `scripts/03_reliability_analysis/`
4. `scripts/04_microstructure_pspp/`
5. `scripts/05_machine_learning/`
6. `scripts/06_icme_fair_metadata/`

The recommended execution sequence and file dependencies are documented in:

- `REPRODUCIBILITY_GUIDE.md`
- `docs/workflow_reference/WORKFLOW_FILE_USAGE_REFERENCE_85.md`

## Reproducibility

The study outputs can be inspected in two ways:

1. Execute the Python and SQL workflows in the sequence described in `REPRODUCIBILITY_GUIDE.md`.
2. Review the validated CSV, Excel, TXT, JSON, Markdown, and figure outputs directly without rebuilding the complete database.

For database-backed execution, configure the required PostgreSQL connection settings locally. Credentials and environment-specific paths are intentionally not stored in this public repository.

## Representative output files

Representative outputs include:

- `data/02_cleaned/sample_level_features_85.csv`
- `data/04_statistics_outputs/task3_1_global_descriptive_stats_85.csv`
- `data/04_statistics_outputs/task3_5B_weibull_global_parameters_85.csv`
- `data/05_ml_outputs/task3_4_ml_model_performance_85.csv`
- `data/07_manuscript_outputs/task6_manuscript_key_results_summary_85.md`
- `docs/workflow_reference/WORKFLOW_FILE_USAGE_REFERENCE_85.md`

## Supplementary material

Supplementary Tables S1–S12 are provided in:

`Supplementary_Tables_S1_to_S12_Fatigue_DBMS.xlsx`

Each supplementary table is included as a separate worksheet within the workbook.

## Intended use

This repository is provided for:

- Reproducibility assessment
- Fatigue-data infrastructure development
- ICME- and PSPP-aligned data integration
- Reliability-oriented fatigue analytics
- Materials-informatics research
- Educational and non-commercial research use
- Extension to additional fatigue datasets and processing routes

Users should preserve dataset provenance and clearly distinguish experimental records from expanded records when reusing or extending the data.

## Public repository

https://github.com/sreearravind/Fatigue-Data-Infrastructure-for-Reliability-Analytics

## Citation

Please cite the associated journal article when using the data, scripts, database structure, or analytical workflow. The complete citation and DOI will be added after publication.
