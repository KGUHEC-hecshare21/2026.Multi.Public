"""Step 2 — Build the lag and rolling covariates used by every model.

Three temporal predictors are appended to the cleaned monthly table:

* ``RN_lag1``     – previous-month precipitation
* ``RN_lag12``    – same calendar month one year earlier (seasonality)
* ``RN_rolling3`` – three-month rolling mean of precipitation (trend)

Missing values that appear at the start of the series are filled with the
station-wise mean of the target so that the resulting table never carries
NaNs into the forecasting models.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config_utils import load_config, resolve


TARGET = "SUM_RN"


def add_lag_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add RN_lag1 / RN_lag12 / RN_rolling3 in-place and return the frame."""
    monthly = monthly.sort_values(["STN", "TM"]).copy()

    def _per_station(g: pd.DataFrame) -> pd.DataFrame:
        g["RN_lag1"] = g[TARGET].shift(1)
        g["RN_lag12"] = g[TARGET].shift(12)
        g["RN_rolling3"] = g[TARGET].rolling(3, min_periods=1).mean().shift(1)
        return g

    monthly = monthly.groupby("STN", group_keys=False).apply(_per_station)
    station_means = monthly.groupby("STN")[TARGET].transform("mean")
    for col in ("RN_lag1", "RN_lag12", "RN_rolling3"):
        monthly[col] = monthly[col].fillna(station_means)
    return monthly


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    pre_dir = resolve(cfg, cfg["paths"]["preprocessed_dir"])

    monthly = pd.read_csv(pre_dir / "monthly_all.csv", parse_dates=["TM"])
    monthly = add_lag_features(monthly)

    train_end = pd.Timestamp(cfg["split"]["initial_train_end"])
    test_start = pd.Timestamp(cfg["split"]["test_start"])
    test_end = pd.Timestamp(cfg["split"]["test_end"])

    monthly.to_csv(pre_dir / "monthly_all.csv", index=False)
    monthly[monthly["TM"] <= train_end].to_csv(pre_dir / "train_monthly.csv", index=False)
    monthly[(monthly["TM"] >= test_start) & (monthly["TM"] <= test_end)].to_csv(
        pre_dir / "test_monthly.csv", index=False
    )
    print(f"[covariates] added lag features to {pre_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
