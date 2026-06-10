"""Step 5 — PCA (V3_PCA) and Varimax-rotated Factor Analysis (V4_FA).

Both methods are fitted on the *training* portion only and persisted as
reusable transformers (joblib) so the forecasting agents can apply them to
the expanding-window slices later without recomputing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from factor_analyzer import FactorAnalyzer
    HAS_FA = True
except Exception:
    HAS_FA = False

from .config_utils import load_config, resolve


def fit_pca(train: pd.DataFrame, kept: list[str], explained: float = 0.85) -> tuple:
    scaler = StandardScaler().fit(train[kept].fillna(0.0))
    x = scaler.transform(train[kept].fillna(0.0))
    full = PCA().fit(x)
    cum = np.cumsum(full.explained_variance_ratio_)
    n = int(np.searchsorted(cum, explained) + 1)
    n = max(2, min(n, len(kept)))
    pca = PCA(n_components=n).fit(x)
    return scaler, pca, n, cum


def fit_factor_analysis(train: pd.DataFrame, kept: list[str], n_factors: int):
    if not HAS_FA:
        return None, None
    scaler = StandardScaler().fit(train[kept].fillna(0.0))
    x = scaler.transform(train[kept].fillna(0.0))
    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
    fa.fit(x)
    return scaler, fa


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    pre_dir = resolve(cfg, cfg["paths"]["preprocessed_dir"])
    out_dir = resolve(cfg, cfg["paths"]["variable_selection_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = (out_dir / "vif_kept.txt").read_text(encoding="utf-8").splitlines()
    kept = [v for v in kept if v]
    train = pd.read_csv(pre_dir / "train_monthly.csv", parse_dates=["TM"])

    scaler, pca, n_pc, cum = fit_pca(
        train, kept, explained=float(cfg["features"].get("pca_explained_variance", 0.85))
    )
    joblib.dump({"scaler": scaler, "pca": pca}, out_dir / "pca_v3.joblib")
    pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(cum))],
                  "cumulative_variance": cum}).to_csv(
        out_dir / "pca_variance_explained.csv", index=False)

    n_fa = int(cfg["features"].get("fa_n_factors", 3))
    sc_fa, fa = fit_factor_analysis(train, kept, n_fa)
    if fa is not None:
        joblib.dump({"scaler": sc_fa, "fa": fa}, out_dir / "fa_v4.joblib")
        loadings = pd.DataFrame(fa.loadings_, index=kept,
                                columns=[f"F{i+1}" for i in range(n_fa)])
        loadings.to_csv(out_dir / "factor_loadings.csv")

    with open(out_dir / "transforms_meta.json", "w", encoding="utf-8") as f:
        json.dump({"pca_n_components": n_pc, "fa_n_factors": n_fa,
                   "has_factor_analyzer": HAS_FA}, f, indent=2)
    print(f"[pca_fa] PCA components={n_pc}, FA factors={n_fa}, FA available={HAS_FA}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
