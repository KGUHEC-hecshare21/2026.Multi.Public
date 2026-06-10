"""Generate the headline figures and summary tables from the metric tables.

This module produces the lightweight artifacts required to reproduce the
publication's quantitative claims:

* ``best_model_per_station.csv`` — the lowest-average-rank model per
  station, matching the spatial map in Figure 7.
* ``metric_heatmap_<METRIC>.png`` — model × feature-setting heatmaps for
  every metric defined in ``config.yaml``.
* ``hydrograph_<station>.png`` — observed vs. predicted monthly series for
  each station's best model.

Heavy multi-panel figures referenced in the paper (Taylor diagrams,
KGE-decomposition charts, etc.) are deliberately left as the production
of the original ``02.project`` codebase; the manuscript publishes the
final PNGs and the data behind them lives in the tables/CSV files above.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config_utils import load_config, resolve


def best_model_per_station(rank_table: pd.DataFrame) -> pd.DataFrame:
    return (rank_table.sort_values("AverageRank")
            .groupby("Station", as_index=False).first()
            [["Station", "Model", "Setting", "AverageRank"]])


def plot_metric_heatmaps(metrics: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("RMSE", "MAE", "KGE", "NSE"):
        col = f"Full_{metric}"
        if col not in metrics.columns:
            continue
        pivot = metrics.pivot_table(index="Model", columns="Setting", values=col, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
        ax.set_title(f"Mean {metric} (Model × Setting)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / f"metric_heatmap_{metric}.png", dpi=200)
        plt.close(fig)


def plot_hydrographs(monthly_dir: Path, best_rows: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for _, row in best_rows.iterrows():
        path = monthly_dir / f"{row['Model']}_{row['Setting']}_{row['Station']}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["TM"])
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(df["TM"], df["Observed"], "k-", label="Observed", linewidth=1.4)
        ax.plot(df["TM"], df["Predicted"], "r--", label=f"{row['Model']} ({row['Setting']})",
                linewidth=1.2)
        ax.set_title(f"{row['Station']} — best model hydrograph")
        ax.set_ylabel("Monthly precipitation (mm)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"hydrograph_{row['Station']}.png", dpi=200)
        plt.close(fig)


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    monthly_dir = resolve(cfg, cfg["paths"]["monthly_predictions_dir"])
    tables_dir = resolve(cfg, cfg["paths"]["tables_dir"])
    fig_dir = resolve(cfg, cfg["paths"]["figures_dir"])

    metrics = pd.read_csv(tables_dir / "all_metrics.csv")
    rank = pd.read_csv(tables_dir / "average_rank.csv")

    best = best_model_per_station(rank)
    best.to_csv(tables_dir / "best_model_per_station.csv", index=False)

    plot_metric_heatmaps(metrics, fig_dir)
    plot_hydrographs(monthly_dir, best, fig_dir)
    print(f"[figures] wrote heatmaps and hydrographs to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
