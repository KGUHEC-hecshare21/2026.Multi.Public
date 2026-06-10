# Workflow Overview

This document accompanies the source code under `src/` and explains how the
modules map onto the manuscript's methodology.

## Pipeline at a glance

```
config/config.yaml
        │
        ▼
src/preprocess_data.py         (Section 3.1)  load KMA CSVs, clean, split
src/construct_covariates.py    (Section 3.1)  add lag / rolling features
src/vif_screening.py           (Section 3.2)  drop variables with VIF > 10
src/feature_selection.py       (Section 3.3)  V2_FS — Pearson · MI · RF rank
src/pca_factor_analysis.py     (Section 3.3)  V3_PCA  / V4_FA transformers
        │
        ▼  (saved transformers in outputs/variable_selection/)
        │
src/feature_settings.py        helper used by every run_*.py
        │
        ├── src/run_sarimax.py   (Section 3.4 — statistical baseline)
        ├── src/run_lstm.py      (deep-learning baseline)
        ├── src/run_timesfm.py   (Foundation Model — TimesFM 2.5)
        ├── src/run_chronos.py   (Foundation Model — Chronos-2)
        ├── src/run_moirai.py    (Foundation Model — Moirai 1.0)
        └── src/run_ttm.py       (Foundation Model — Tiny Time Mixer)
        │
        ▼  (192 forecast CSVs in outputs/monthly/)
        │
src/evaluate_models.py         (Section 4)    six metrics × eight periods
src/make_figures_tables.py     (Section 4)    headline figures + tables
```

## File-naming convention

```
{Model}_{Setting}_{Station}.csv
SARIMAX_V1_ALL_Seoul.csv
TimesFM_V3_PCA_Busan.csv
TTM_V4_FA_Gangneung.csv
```

## Experimental matrix

| Axis | Cardinality | Values |
|------|------------:|--------|
| Models   | 6 | SARIMAX, LSTM, TimesFM 2.5, Chronos-2, Moirai 1.0, TTM |
| Settings | 4 | V1_ALL (9 vars), V2_FS (top-5), V3_PCA (≥85 % variance), V4_FA (Varimax) |
| Stations | 8 | Gangneung, Seoul, Incheon, Daejeon, Daegu, Jeonju, Gwangju, Busan |

Total: **192 forecasts × 60 monthly steps each**.

## Foundation-model installation

The four foundation-model wrappers (`run_timesfm`, `run_chronos`,
`run_moirai`, `run_ttm`) load their respective Python packages lazily.
If the package is **not** installed, a transparent in-context regression
fallback is used so the pipeline can still execute end-to-end (as in the
quick test).  To reproduce the manuscript numbers, install the libraries
listed at the bottom of `requirements.txt` and re-run the model.

## Reproducing the full benchmark

```bash
python -m src.preprocess_data
python -m src.construct_covariates
python -m src.vif_screening
python -m src.feature_selection
python -m src.pca_factor_analysis
python -m src.run_sarimax
python -m src.run_lstm
python -m src.run_timesfm
python -m src.run_chronos
python -m src.run_moirai
python -m src.run_ttm
python -m src.evaluate_models
python -m src.make_figures_tables
```
