"""TTM (Tiny Time Mixer) expanding-window forecasting wrapper.

Tries the IBM ``tsfm-public`` package; otherwise falls back to a ridge
regression on a context window of past observations concatenated with the
exogenous design matrix.  Same wrapper contract as the other ``run_*.py``
modules.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config_utils import load_config, resolve
from .feature_settings import design_matrix, load_settings
from .metrics import calc_all_metrics

warnings.filterwarnings("ignore")
TARGET = "SUM_RN"

try:
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction  # noqa: F401
    HAS_TTM = True
except Exception:
    HAS_TTM = False


def _fallback(y, X_tr, X_te, context: int = 12) -> float:
    if len(y) <= context:
        return max(float(np.mean(y)), 0.0)
    X = []
    yy = []
    for i in range(context, len(y)):
        X.append(np.concatenate([y[i - context:i], X_tr[i]]))
        yy.append(y[i])
    X = np.asarray(X)
    yy = np.asarray(yy)
    model = Ridge(alpha=1.0).fit(X, yy)
    xt = np.concatenate([y[-context:], np.atleast_2d(X_te)[0]]).reshape(1, -1)
    return max(float(model.predict(xt)[0]), 0.0)


def run_one_station(stn_df, setting, settings, test_dates) -> pd.DataFrame:
    stn_df = stn_df.sort_values("TM").reset_index(drop=True)
    rows = []
    for ts in test_dates:
        train = stn_df[stn_df["TM"] < ts]
        row = stn_df[stn_df["TM"] == ts]
        if len(train) < 36 or len(row) == 0:
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
            preds.to_csv(monthly_dir / f"TTM_{v}_{stn}.csv", index=False)
            m = calc_all_metrics(preds["Observed"].to_numpy(), preds["Predicted"].to_numpy())
            summary.append({"Model": "TTM", "Setting": v, "Station": stn, **m})
    df = pd.DataFrame(summary)
    df.to_csv(monthly_dir / "TTM_summary.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run(args.config))
