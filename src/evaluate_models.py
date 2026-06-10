"""Aggregate per-experiment forecasts into the evaluation tables.

Reads every ``{Model}_{Setting}_{Station}.csv`` produced by the model
modules, computes the six metrics over the full test horizon and the
seasonal / extreme-month subsets defined in the manuscript, and writes
the comparison tables consumed by ``make_figures_tables.py``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config_utils import load_config, resolve
from .metrics import calc_all_metrics

FNAME_RE = re.compile(r"^(?P<model>SARIMAX|LSTM|TimesFM|Chronos|Moirai|TTM)_"
                      r"(?P<setting>V[1-4]_[A-Z]+)_(?P<station>[A-Za-z]+)\.csv$")

SEASONS = {
    "Spring": {3, 4, 5}, "Summer": {6, 7, 8},
    "Autumn": {9, 10, 11}, "Winter": {12, 1, 2},
}


def _evaluate_one(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["TM"] = pd.to_datetime(df["TM"])
    obs, pred = df["Observed"].to_numpy(), df["Predicted"].to_numpy()
    out = {f"Full_{k}": v for k, v in calc_all_metrics(obs, pred).items()}
    for s, months in SEASONS.items():
        sub = df[df["TM"].dt.month.isin(months)]
        if len(sub) >= 2:
            for k, v in calc_all_metrics(sub["Observed"].to_numpy(),
                                          sub["Predicted"].to_numpy()).items():
                out[f"{s}_{k}"] = v
    for top_n in (1, 2, 3):
        sub = df.nlargest(top_n, "Observed")
        if len(sub) >= 1:
            for k, v in calc_all_metrics(sub["Observed"].to_numpy(),
                                          sub["Predicted"].to_numpy()).items():
                out[f"Top{top_n}_{k}"] = v
    return out


def aggregate(monthly_dir: Path) -> pd.DataFrame:
    records = []
    for csv in sorted(monthly_dir.glob("*.csv")):
        m = FNAME_RE.match(csv.name)
        if not m:
            continue
        df = pd.read_csv(csv)
        rec = {"Model": m["model"], "Setting": m["setting"], "Station": m["station"]}
        rec.update(_evaluate_one(df))
        records.append(rec)
    return pd.DataFrame(records)


def compute_average_rank(table: pd.DataFrame, metric_prefix: str = "Full_") -> pd.DataFrame:
    """Rank by the six full-period metrics and average the ranks."""
    metric_cols = [c for c in table.columns if c.startswith(metric_prefix)
                   and c.split("_")[-1] in ("RMSE", "MAE", "PBIAS", "R2", "NSE", "KGE")]
    ranks = table[["Model", "Setting", "Station"]].copy()
    for c in metric_cols:
        if c.endswith(("RMSE", "MAE")):
            ranks[c] = table[c].rank(ascending=True)
        elif c.endswith("PBIAS"):
            ranks[c] = table[c].abs().rank(ascending=True)
        else:
            ranks[c] = table[c].rank(ascending=False)
    ranks["AverageRank"] = ranks[metric_cols].mean(axis=1)
    return ranks.sort_values("AverageRank")


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    monthly_dir = resolve(cfg, cfg["paths"]["monthly_predictions_dir"])
    tables_dir = resolve(cfg, cfg["paths"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    table = aggregate(monthly_dir)
    table.to_csv(tables_dir / "all_metrics.csv", index=False)
    rank = compute_average_rank(table)
    rank.to_csv(tables_dir / "average_rank.csv", index=False)
    print(f"[evaluate] {len(table)} experiments → {tables_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
