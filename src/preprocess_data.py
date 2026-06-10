"""Step 1 — Load raw KMA ASOS observations and aggregate to monthly.

Reads the daily and monthly CSV files distributed by KMA (cp949 encoding),
joins them per station, fills precipitation NaNs with 0, linearly
interpolates the remaining meteorological variables, and splits the
monthly table into train / test according to ``config.yaml``.

The same logic is used by ``examples/quick_test.py`` after substituting the
sample (synthetic) CSVs in ``data/sample/`` for the raw KMA files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config_utils import load_config, resolve

RAW_DAILY_COLS_KEEP = [
    "STN", "TM", "AVG_TA", "MIN_TA", "MAX_TA", "RN_DAY",
    "AVG_WS", "AVG_RHM", "AVG_PV", "AVG_PA",
    "SUM_SS", "SUM_SR",
]
RAW_MONTHLY_COLS_KEEP = [
    "STN", "TM", "AVG_TA", "MIN_TA", "MAX_TA", "SUM_RN",
    "AVG_WS", "AVG_RHM", "AVG_PV", "AVG_PA",
    "SUM_SS", "SUM_SR",
]


def _read_kma_csv(path: Path) -> pd.DataFrame:
    """Read a single KMA CSV, tolerating cp949 (raw) or utf-8 (sample)."""
    # utf-8-sig first so a BOM-prefixed sample CSV does not pollute the STN column name.
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(f"Could not decode {path}")
    df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)
    return df


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate the daily and monthly raw CSV files.

    When ``data.daily_files`` is empty (e.g. in the quick-test config that
    only ships a synthetic monthly file) the daily frame is returned empty.
    """
    input_dir = resolve(cfg, cfg["paths"]["input_dir"])
    daily_files = cfg["data"].get("daily_files") or []
    daily_frames = [_read_kma_csv(input_dir / f) for f in daily_files]
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    monthly = _read_kma_csv(input_dir / cfg["data"]["monthly_file"])
    return daily, monthly


def _parse_dates(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if granularity == "daily":
        df["TM"] = pd.to_datetime(df["TM"], errors="coerce")
    else:
        df["TM"] = pd.to_datetime(df["TM"], errors="coerce")
        if df["TM"].isna().mean() > 0.5:
            df["TM"] = pd.to_datetime(df["TM"], format="%b-%y", errors="coerce")
    return df


def preprocess(cfg: dict) -> pd.DataFrame:
    """Return the cleaned monthly table for the configured stations."""
    daily, monthly = load_raw(cfg)
    if not daily.empty:
        daily = _parse_dates(daily, "daily")
    monthly = _parse_dates(monthly, "monthly")

    stations = cfg["stations"]
    keep_codes = [int(s["code"]) for s in stations]
    code2name = {int(s["code"]): s["name"] for s in stations}

    if not daily.empty:
        daily = daily[daily["STN"].isin(keep_codes)].copy()
    monthly = monthly[monthly["STN"].isin(keep_codes)].copy()

    if "RN_DAY" in daily.columns and not daily.empty:
        daily["RN_DAY"] = daily["RN_DAY"].fillna(0.0)
    if "SUM_RN" in monthly.columns:
        monthly["SUM_RN"] = monthly["SUM_RN"].fillna(0.0)

    met_vars = [c for c in RAW_MONTHLY_COLS_KEEP if c not in ("STN", "TM", "SUM_RN")]
    for v in met_vars:
        if v in monthly.columns:
            monthly[v] = (
                monthly.groupby("STN")[v]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )

    monthly["station_name"] = monthly["STN"].map(code2name)
    monthly = monthly.dropna(subset=["TM"]).sort_values(["STN", "TM"]).reset_index(drop=True)
    return monthly


def split_train_test(monthly: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the chronological train/test cut defined in the config."""
    train_end = pd.Timestamp(cfg["split"]["initial_train_end"])
    test_start = pd.Timestamp(cfg["split"]["test_start"])
    test_end = pd.Timestamp(cfg["split"]["test_end"])
    train = monthly[monthly["TM"] <= train_end].copy()
    test = monthly[(monthly["TM"] >= test_start) & (monthly["TM"] <= test_end)].copy()
    return train, test


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    out_dir = resolve(cfg, cfg["paths"]["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly = preprocess(cfg)
    train, test = split_train_test(monthly, cfg)

    monthly.to_csv(out_dir / "monthly_all.csv", index=False)
    train.to_csv(out_dir / "train_monthly.csv", index=False)
    test.to_csv(out_dir / "test_monthly.csv", index=False)
    print(f"[preprocess] wrote {len(monthly)} monthly rows to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
