"""Generate synthetic monthly data that mimics KMA ASOS records.

The raw Korea Meteorological Administration observations cannot be
redistributed (see README §9).  This script produces a small, fully
synthetic dataset with the *same column names and shape* as the real
input CSVs so that the entire pipeline — preprocessing, VIF screening,
feature engineering, expanding-window forecasting, evaluation, figure
generation — can be exercised by a reviewer in a few minutes.

The synthetic generator:

* covers 2001-01 → 2025-12, eight stations, monthly resolution;
* injects a 12-month seasonal cycle into precipitation, temperature,
  humidity, and solar radiation, plus station-specific offsets;
* adds Gaussian noise and a mild long-term trend so the resulting series
  are non-trivial for the forecasting models.

The output files are written to ``data/sample/`` and are picked up by
``examples/quick_test.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"

STATIONS = [
    (105, "Gangneung"), (108, "Seoul"), (112, "Incheon"), (133, "Daejeon"),
    (143, "Daegu"), (146, "Jeonju"), (156, "Gwangju"), (159, "Busan"),
]


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2001-01-01", "2025-12-01", freq="MS")
    rows = []
    for code, name in STATIONS:
        offset = (code % 17) * 1.5
        for ts in dates:
            m = ts.month
            seasonal_p = 110 + 95 * np.sin(2 * np.pi * (m - 6) / 12)
            sum_rn = max(0.0, seasonal_p + offset + rng.normal(0, 35))
            max_ta = 18 + 10 * np.sin(2 * np.pi * (m - 4) / 12) + rng.normal(0, 1.6)
            avg_ta = max_ta - 4 - rng.uniform(0.5, 2.0)
            min_ta = max_ta - 8 - rng.uniform(0.5, 2.5)
            avg_rh = 65 + 12 * np.sin(2 * np.pi * (m - 7) / 12) + rng.normal(0, 3.5)
            avg_pa = 1004 + rng.normal(0, 4) - 0.05 * max_ta
            avg_pv = 12 + 0.5 * max_ta + rng.normal(0, 1.5)
            avg_ws = 2.4 + rng.normal(0, 0.4)
            sum_ss = max(0.0, 180 + 60 * np.sin(2 * np.pi * (m - 5) / 12) + rng.normal(0, 25))
            sum_sr = max(0.0, 400 + 180 * np.sin(2 * np.pi * (m - 5) / 12) + rng.normal(0, 60))
            rows.append({
                "STN": code, "STN_NM": name, "TM": ts.strftime("%Y-%m-%d"),
                "SUM_RN": round(sum_rn, 1),
                "AVG_TA": round(avg_ta, 1),
                "MIN_TA": round(min_ta, 1),
                "MAX_TA": round(max_ta, 1),
                "AVG_RHM": round(float(np.clip(avg_rh, 30, 95)), 1),
                "AVG_PA": round(float(avg_pa), 1),
                "AVG_PV": round(float(avg_pv), 1),
                "AVG_WS": round(float(avg_ws), 2),
                "SUM_SS": round(float(sum_ss), 1),
                "SUM_SR": round(float(sum_sr), 1),
            })
    return pd.DataFrame(rows)


def write_sample(seed: int = 42) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    monthly = generate(seed)
    monthly_path = SAMPLE_DIR / "OBS_ASOS_MNH_2001.2025.csv"
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    # Also publish a minimal "human readable" sample pair referenced in §5 of the README.
    monthly[["STN_NM", "TM", "SUM_RN"]].to_csv(
        SAMPLE_DIR / "sample_monthly_precipitation.csv", index=False)
    monthly[["STN_NM", "TM", "MAX_TA", "AVG_PA", "AVG_RHM",
             "AVG_WS", "SUM_SS", "SUM_SR"]].to_csv(
        SAMPLE_DIR / "sample_covariates.csv", index=False)
    print(f"[synthetic] wrote {len(monthly):,} rows to {monthly_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_sample(args.seed)
