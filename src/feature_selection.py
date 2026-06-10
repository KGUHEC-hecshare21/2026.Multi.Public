"""Step 4 — Ensemble feature selection (V2_FS).

Ranks the VIF-kept predictors by an unweighted average of three rankings:
absolute Pearson correlation, Mutual Information, and Random-Forest
importance.  The five top-ranked variables are saved as the V2_FS subset.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from .config_utils import load_config, resolve

TARGET = "SUM_RN"


def rank_features(train: pd.DataFrame, predictors: list[str], seed: int = 42) -> pd.DataFrame:
    x = train[predictors].fillna(train[predictors].median())
    y = train[TARGET].fillna(0.0)

    pearson = x.apply(lambda c: c.corr(y)).abs()
    mi = pd.Series(mutual_info_regression(x, y, random_state=seed), index=predictors)
    rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(x, y)
    rf_imp = pd.Series(rf.feature_importances_, index=predictors)

    table = pd.DataFrame({
        "Pearson": pearson, "MI": mi, "RF": rf_imp,
        "rank_P": pearson.rank(ascending=False),
        "rank_MI": mi.rank(ascending=False),
        "rank_RF": rf_imp.rank(ascending=False),
    })
    table["avg_rank"] = table[["rank_P", "rank_MI", "rank_RF"]].mean(axis=1)
    return table.sort_values("avg_rank")


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    pre_dir = resolve(cfg, cfg["paths"]["preprocessed_dir"])
    out_dir = resolve(cfg, cfg["paths"]["variable_selection_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = (out_dir / "vif_kept.txt").read_text(encoding="utf-8").splitlines()
    kept = [v for v in kept if v]

    train = pd.read_csv(pre_dir / "train_monthly.csv", parse_dates=["TM"])
    table = rank_features(train, kept, cfg.get("random_seed", 42))

    top_k = int(cfg["features"].get("v2_top_k", 5))
    selected = table.head(top_k).index.tolist()

    table.to_csv(out_dir / "feature_selection_results.csv")
    (out_dir / "v2_fs_selected.txt").write_text("\n".join(selected), encoding="utf-8")
    print(f"[feature_selection] V2_FS top {top_k}: {selected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
