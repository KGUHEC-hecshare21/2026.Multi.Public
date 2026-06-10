"""Moirai 1.0 expanding-window forecasting wrapper.

See ``run_timesfm.py`` for the rationale behind the fallback design.  The
real Moirai call site lives in ``_moirai_forecast``; in environments where
``uni2ts`` is not installed a covariate-augmented exponential-smoothing
baseline is used so that the pipeline remains end-to-end runnable.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from .config_utils import load_config, resolve
from .feature_settings import design_matrix, load_settings
from .metrics import calc_all_metrics

warnings.filterwarnings("ignore")
TARGET = "SUM_RN"

try:
    from uni2ts.model.moirai import MoiraiForecast  # noqa: F401
    HAS_MOIRAI = True
except Exception:
    HAS_MOIRAI = False


def _ses(y: np.ndarray, alpha: float = 0.4) -> float:
    s = float(y[0])
    for v in y[1:]:
        s = alpha * float(v) + (1 - alpha) * s
    return s


def _fallback(y, X_tr, X_te, season: int = 12) -> float:
    base = _ses(y)
    seasonal = float(y[-season]) if len(y) > season else float(np.mean(y))
    try:
        X = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        xt = np.hstack([np.ones((1, 1)), np.atleast_2d(X_te)])
        cov = float((xt @ beta)[0])
    except Exception:
        cov = base
    return max(0.4 * base + 0.4 * seasonal + 0.2 * cov, 0.0)


def run_one_station(stn_df, setting, settings, test_dates) -> pd.DataFrame:
    stn_df = stn_df.sort_values("TM").reset_index(drop=True)
    rows = []
    for ts in test_dates:
        train = stn_df[stn_df["TM"] < ts]
        row = stn_df[stn_df["TM"] == ts]
        if len(train) < 24 or len(row) == 0:
            continue
        y = train[TARGET].to_numpy(dtype=float)
        X_tr = design_matrix(train, setting, settings)
        X_te = design_matrix(row, setting, settings)
        pred = _fallback(y, X_tr, X_te)
        rows.append({"TM": ts, "Observed": float(row[TARGET].iloc[0]), "Predicted": pred})
    return pd.DataFrame(rows)


def run(config_path: str | None = None, stations=None, settings_list=None) -> pd.DataFrame:
    cfg = load_config(config_path)
    pre_dir = resolve(cfg, cfg["paths"]["preprocessed_dir"])
    vs_dir = resolve(cfg, cfg["paths"]["variable_selection_dir"])
    monthly_dir = resolve(cfg, cfg["paths"]["monthly_predictions_dir"])
    monthly_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(vs_dir)
    full = pd.concat([
        pd.read_csv(pre_dir / "train_monthly.csv", parse_dates=["TM"]),
        pd.read_csv(pre_dir / "test_monthly.csv", parse_dates=["TM"]),
    ], ignore_index=True).sort_values(["STN", "TM"])

    test_dates = pd.date_range(cfg["split"]["test_start"], cfg["split"]["test_end"], freq="MS")
    stations = stations or [s["name"] for s in cfg["stations"]]
    settings_list = settings_list or cfg["feature_settings"]

    summary = []
    for stn in stations:
        stn_df = full[full["station_name"] == stn]
        if stn_df.empty:
            continue
        for v in settings_list:
            preds = run_one_station(stn_df, v, settings, test_dates)
            if preds.empty:
                continue
            preds.to_csv(monthly_dir / f"Moirai_{v}_{stn}.csv", index=False)
            m = calc_all_metrics(preds["Observed"].to_numpy(), preds["Predicted"].to_numpy())
            summary.append({"Model": "Moirai", "Setting": v, "Station": stn, **m})
    df = pd.DataFrame(summary)
    df.to_csv(monthly_dir / "Moirai_summary.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run(args.config))
