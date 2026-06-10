"""Six evaluation metrics used throughout the manuscript.

RMSE, MAE, PBIAS, R^2, NSE, KGE — implemented exactly as defined in the
methodology section.  ``calc_all_metrics`` returns the standard dict that
downstream code (evaluate_models, make_figures_tables, quick_test) expects.
"""
from __future__ import annotations

import numpy as np


def calc_rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def calc_mae(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(obs - pred)))


def calc_pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    s = float(np.sum(obs))
    if s == 0:
        return 0.0
    return 100.0 * float(np.sum(pred - obs)) / s


def calc_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.std(obs) == 0 or np.std(pred) == 0:
        return 0.0
    r = float(np.corrcoef(obs, pred)[0, 1])
    return r * r


def calc_nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom == 0:
        return 0.0
    return 1.0 - float(np.sum((obs - pred) ** 2)) / denom


def calc_kge(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.std(obs) == 0 or np.mean(obs) == 0:
        return 0.0
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(np.std(pred) / np.std(obs))
    beta = float(np.mean(pred) / np.mean(obs))
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def calc_all_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "RMSE": calc_rmse(obs, pred),
        "MAE": calc_mae(obs, pred),
        "PBIAS": calc_pbias(obs, pred),
        "R2": calc_r2(obs, pred),
        "NSE": calc_nse(obs, pred),
        "KGE": calc_kge(obs, pred),
    }
