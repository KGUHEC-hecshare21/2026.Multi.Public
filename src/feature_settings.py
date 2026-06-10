"""Helper that returns the exogenous design matrix for a given setting.

Used by every ``run_*.py`` model so that the four feature configurations
(V1_ALL / V2_FS / V3_PCA / V4_FA) are wired up in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_settings(variable_selection_dir: Path):
    """Read kept variables, V2_FS subset, and fitted transformers."""
    kept = (variable_selection_dir / "vif_kept.txt").read_text(encoding="utf-8").splitlines()
    kept = [v for v in kept if v]

    fs_path = variable_selection_dir / "v2_fs_selected.txt"
    fs = fs_path.read_text(encoding="utf-8").splitlines() if fs_path.exists() else kept[:5]
    fs = [v for v in fs if v]

    pca_obj = joblib.load(variable_selection_dir / "pca_v3.joblib") if (
        variable_selection_dir / "pca_v3.joblib"
    ).exists() else None
    fa_obj = joblib.load(variable_selection_dir / "fa_v4.joblib") if (
        variable_selection_dir / "fa_v4.joblib"
    ).exists() else None
    return {"kept": kept, "fs": fs, "pca": pca_obj, "fa": fa_obj}


def design_matrix(df: pd.DataFrame, setting: str, settings: dict) -> np.ndarray:
    kept = settings["kept"]
    if setting == "V1_ALL":
        return df[kept].fillna(0.0).to_numpy(dtype=float)
    if setting == "V2_FS":
        cols = [c for c in settings["fs"] if c in df.columns] or kept[:5]
        return df[cols].fillna(0.0).to_numpy(dtype=float)
    if setting == "V3_PCA" and settings["pca"] is not None:
        s = settings["pca"]["scaler"].transform(df[kept].fillna(0.0))
        return settings["pca"]["pca"].transform(s)
    if setting == "V4_FA" and settings["fa"] is not None:
        s = settings["fa"]["scaler"].transform(df[kept].fillna(0.0))
        return settings["fa"]["fa"].transform(s)
    return df[kept].fillna(0.0).to_numpy(dtype=float)
