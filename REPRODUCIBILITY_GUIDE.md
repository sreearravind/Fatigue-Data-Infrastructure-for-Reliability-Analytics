# Reproducibility Guide

## Python environment setup

Create a Python environment compatible with the copied scripts, then install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Database setup scripts may require PostgreSQL configuration before execution.
- Manuscript-only validation can be performed using the existing CSV outputs without rebuilding PostgreSQL.
- File paths in copied scripts have been adjusted for repository-relative execution; if further local changes are made, update paths relative to the repository root.

## Recommended execution order

1. Database setup scripts
2. Data preparation / aggregation scripts
3. Statistical characterization scripts
4. Reliability analysis scripts
5. Microstructure-PSPP scripts
6. Machine-learning validation scripts
7. ICME/FAIR metadata scripts
8. Manuscript compiler script
