"""SARIMAX expanding-window forecasting (statistical baseline).

For each (station, feature setting) the expanding-window protocol from the
manuscript is applied: the model is refit one month at a time over the
2021-01 → 2025-12 test horizon, using the exogenous design matrix produced
by ``feature_settings.design_matrix``.  A short grid search on the initial
training block (2001–2020) chooses the (p,d,q)(P,D,Q,12) order by AIC and
that order is reused for every subsequent monthly refit.
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config_utils import load_config, resolve
from .feature_settings import design_matrix, load_settings
from .metrics import calc_all_metrics

warnings.filterwarnings("ignore")

TARGET = "SUM_RN"


def _grid_search(y, X, ranges: dict, season: int):
    best = (float("inf"), (1, 0, 1), (0, 0, 0, season))
    p_r, d_r, q_r = ranges["p"], ranges["d"], ranges["q"]
    P_r, D_r, Q_r = ranges["P"], ranges["D"], ranges["Q"]
    for (p, d, q), (P, D, Q) in itertools.product(
        itertools.product(p_r, d_r, q_r), itertools.product(P_r, D_r, Q_r)
    ):
        try:
            m = SARIMAX(y, exog=X, order=(p, d, q), seasonal_order=(P, D, Q, season),
                        enforce_stationarity=False, enforce_invertibility=False)
            f = m.fit(disp=False, maxiter=80, method="lbfgs")
            if f.aic < best[0]:
                best = (f.aic, (p, d, q), (P, D, Q, season))
        except Exception:
            continue
    return best


def run_one_station(stn_df: pd.DataFrame, setting: str, settings: dict,
                    test_dates: pd.DatetimeIndex, grid: dict, season: int) -> pd.DataFrame:
    stn_df = stn_df.sort_values("TM").reset_index(drop=True)
    init = stn_df[stn_df["TM"] < test_dates[0]]
    y_init = init[TARGET].to_numpy(dtype=float)
    X_init = design_matrix(init, setting, settings)
    _, order, seasonal = _grid_search(y_init, X_init, grid, season)

    rows = []
    for ts in test_dates:
        train = stn_df[stn_df["TM"] < ts]
        row = stn_df[stn_df["TM"] == ts]
        if len(train) < 24 or len(row) == 0:
            continue
        y = train[TARGET].to_numpy(dtype=float)
        X_tr = design_matrix(train, setting, settings)
        X_te = design_matrix(row, setting, settings)
        try:
            fit = SARIMAX(y, exog=X_tr, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False, enforce_invertibility=False
                          ).fit(disp=False, maxiter=120, method="lbfgs")
            pred = float(np.maximum(fit.forecast(steps=1, exog=X_te)[0], 0.0))
        except Exception:
            pred = float(max(0.0, y[-1]))
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
    grid = cfg["sarimax"]["grid"]
    season = int(cfg["sarimax"].get("seasonal_period", 12))

    stations = stations or [s["name"] for s in cfg["stations"]]
    settings_list = settings_list or cfg["feature_settings"]

    summary = []
    for stn in stations:
        stn_df = full[full["station_name"] == stn]
        if stn_df.empty:
            continue
        for v in settings_list:
            preds = run_one_station(stn_df, v, settings, test_dates, grid, season)
            if preds.empty:
                continue
            preds.to_csv(monthly_dir / f"SARIMAX_{v}_{stn}.csv", index=False)
            m = calc_all_metrics(preds["Observed"].to_numpy(), preds["Predicted"].to_numpy())
            summary.append({"Model": "SARIMAX", "Setting": v, "Station": stn, **m})
    df = pd.DataFrame(summary)
    df.to_csv(monthly_dir / "SARIMAX_summary.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run(args.config))
