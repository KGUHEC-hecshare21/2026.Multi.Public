"""End-to-end quick test on synthetic data.

This script demonstrates that the full computational pipeline works on a
self-contained sample. It is the example referenced in the journal
submission cover letter ("at least one quick-test or example file").

Steps executed:

1. Regenerate ``data/sample/`` if it does not already exist.
2. Run preprocessing, lag-feature construction, VIF screening, feature
   selection, and PCA / FA fitting on the sample.
3. Train and evaluate SARIMAX and TimesFM (linear-XReg fallback) on
   **one station (Seoul) and one feature setting (V1_ALL)** — small
   enough to finish in roughly a minute on a laptop CPU.
4. Persist ``outputs/quick_test/predictions.csv`` and
   ``outputs/quick_test/metrics.csv``.

To reproduce the full 192-experiment benchmark on real data, swap the
sample CSVs for the real KMA files and run the full workflow described
in README §7.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from examples.make_synthetic_data import write_sample, SAMPLE_DIR  # noqa: E402
from src import (                                                   # noqa: E402
    construct_covariates,
    evaluate_models,
    feature_selection,
    make_figures_tables,
    pca_factor_analysis,
    preprocess_data,
    run_sarimax,
    run_timesfm,
    vif_screening,
)
from src.config_utils import load_config                            # noqa: E402
from src.metrics import calc_all_metrics                            # noqa: E402

QUICK_CONFIG = REPO_ROOT / "config" / "quick_test_config.yaml"


def _write_quick_config(base_cfg: dict) -> None:
    """Point input_dir at data/sample/ and shrink the pipeline scope."""
    cfg = dict(base_cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["input_dir"] = "data/sample"
    cfg["paths"]["preprocessed_dir"] = "outputs/quick_test/preprocessed"
    cfg["paths"]["variable_selection_dir"] = "outputs/quick_test/variable_selection"
    cfg["paths"]["monthly_predictions_dir"] = "outputs/quick_test/monthly"
    cfg["paths"]["tables_dir"] = "outputs/quick_test/tables"
    cfg["paths"]["figures_dir"] = "outputs/quick_test/figures"
    cfg["data"] = {"daily_files": [], "monthly_file": "OBS_ASOS_MNH_2001.2025.csv"}
    cfg.pop("_repo_root", None)
    with open(QUICK_CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    base = load_config()
    _write_quick_config(base)

    if not (SAMPLE_DIR / "OBS_ASOS_MNH_2001.2025.csv").exists():
        write_sample()

    qcfg = str(QUICK_CONFIG)
    print("\n[1/7] preprocess_data ..."); preprocess_data.main(qcfg)
    print("[2/7] construct_covariates ..."); construct_covariates.main(qcfg)
    print("[3/7] vif_screening ..."); vif_screening.main(qcfg)
    print("[4/7] feature_selection ..."); feature_selection.main(qcfg)
    print("[5/7] pca_factor_analysis ..."); pca_factor_analysis.main(qcfg)

    print("[6/7] run_sarimax (1 station, V1_ALL) ...")
    sarimax_summary = run_sarimax.run(qcfg, stations=["Seoul"], settings_list=["V1_ALL"])

    print("[7/7] run_timesfm fallback (1 station, V1_ALL) ...")
    timesfm_summary = run_timesfm.run(qcfg, stations=["Seoul"], settings_list=["V1_ALL"])

    print("\n[evaluate] aggregating ...")
    evaluate_models.main(qcfg)
    make_figures_tables.main(qcfg)

    out_dir = REPO_ROOT / "outputs" / "quick_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for csv in sorted((out_dir / "monthly").glob("*.csv")):
        if csv.name.endswith("_summary.csv"):
            continue
        parts = csv.stem.split("_")
        model, setting, station = parts[0], "_".join(parts[1:3]), "_".join(parts[3:])
        df = pd.read_csv(csv)
        df["Model"] = model
        df["Setting"] = setting
        df["Station"] = station
        predictions.append(df)
    pred_all = pd.concat(predictions, ignore_index=True)
    pred_all.to_csv(out_dir / "predictions.csv", index=False)

    rows = []
    for (model, setting, station), df in pred_all.groupby(["Model", "Setting", "Station"]):
        rows.append({
            "Model": model, "Setting": setting, "Station": station,
            **calc_all_metrics(df["Observed"].to_numpy(), df["Predicted"].to_numpy()),
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    print("\nQuick-test metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nWrote:\n  {out_dir / 'predictions.csv'}\n  {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
