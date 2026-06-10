"""Step 3 — VIF-based multicollinearity screening.

Computes the Variance Inflation Factor for every candidate predictor and
removes those above the threshold (default 10) declared in ``config.yaml``.
The variables that survive (``MAX_TA, AVG_PA, AVG_RHM, AVG_WS, SUM_SS,
SUM_SR`` plus three lag features) are persisted as ``vif_kept.txt`` and
consumed by every downstream model.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config_utils import load_config, resolve


def compute_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = df[columns].dropna()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    vifs = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
    return pd.DataFrame({"Variable": columns, "VIF": vifs}).sort_values("VIF", ascending=False)


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    pre_dir = resolve(cfg, cfg["paths"]["preprocessed_dir"])
    out_dir = resolve(cfg, cfg["paths"]["variable_selection_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(pre_dir / "train_monthly.csv", parse_dates=["TM"])
    candidates = cfg["features"]["candidate_variables"]
    threshold = float(cfg["features"]["vif_threshold"])

    vif_df = compute_vif(train, candidates)
    kept = vif_df.loc[vif_df["VIF"] <= threshold, "Variable"].tolist()
    extras = cfg["features"].get("always_keep", [])
    for v in extras:
        if v not in kept and v in train.columns:
            kept.append(v)

    vif_df.to_csv(out_dir / "vif_table.csv", index=False)
    (out_dir / "vif_kept.txt").write_text("\n".join(kept), encoding="utf-8")
    print(f"[vif] kept {len(kept)} predictors: {kept}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
