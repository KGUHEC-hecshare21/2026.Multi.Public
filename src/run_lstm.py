"""LSTM expanding-window forecasting (deep-learning baseline).

A two-layer LSTM with dropout, trained from scratch at every expanding-
window step.  Sequences of length ``look_back`` (12 months by default) feed
the exogenous variables; the network outputs a single scaled scalar that
is back-transformed to a non-negative precipitation value.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from .config_utils import load_config, resolve
from .feature_settings import design_matrix, load_settings
from .metrics import calc_all_metrics

warnings.filterwarnings("ignore")
TARGET = "SUM_RN"


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _make_sequences(x: np.ndarray, y: np.ndarray, look_back: int):
    xs, ys = [], []
    for i in range(look_back, len(x)):
        xs.append(x[i - look_back:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _train_once(X_seq, y_seq, n_features, device, epochs: int, lr: float, seed: int):
    torch.manual_seed(seed)
    model = LSTMRegressor(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.from_numpy(X_seq).to(device)
    y = torch.from_numpy(y_seq).to(device).unsqueeze(-1)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    return model


def run_one_station(stn_df: pd.DataFrame, setting: str, settings: dict,
                    test_dates: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    look_back = int(cfg["lstm"]["look_back"])
    epochs = int(cfg["lstm"]["epochs"])
    lr = float(cfg["lstm"]["lr"])
    seed = int(cfg.get("random_seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stn_df = stn_df.sort_values("TM").reset_index(drop=True)
    rows = []
    for ts in test_dates:
        train = stn_df[stn_df["TM"] < ts]
        row = stn_df[stn_df["TM"] == ts]
        if len(train) < look_back + 24 or len(row) == 0:
            continue
        X_full = design_matrix(train, setting, settings)
        if X_full.size == 0 or X_full.shape[1] == 0:
            continue
        y_full = train[TARGET].to_numpy(dtype=float).reshape(-1, 1)

        x_scaler = MinMaxScaler().fit(X_full)
        y_scaler = MinMaxScaler().fit(y_full)
        Xs = x_scaler.transform(X_full)
        ys = y_scaler.transform(y_full).flatten()
        X_seq, y_seq = _make_sequences(Xs, ys, look_back)
        if len(X_seq) < 8:
            continue

        model = _train_once(X_seq, y_seq, Xs.shape[1], device, epochs, lr, seed)
        model.eval()
        last_window = torch.from_numpy(Xs[-look_back:].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(last_window).cpu().numpy().reshape(-1, 1)
        pred = float(np.maximum(y_scaler.inverse_transform(pred_scaled)[0, 0], 0.0))
        rows.append({"TM": ts, "Observed": float(row[TARGET].iloc[0]), "Predicted": pred})
    return pd.DataFrame(rows)


def run(config_path: str | None = None, stations=None, settings_list=None) -> pd.DataFrame:
    cfg = load_config(config_path)
    pre_dir = resolve(cfg, cfg["paths"]["preprocessed_dir"])
    vs_dir = resolve(cfg, cfg["paths"]["variable_selection_dir"])
    monthly_dir = resolve(cfg, cfg["paths"]["monthly_predictions_dir"])
    monthly_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(vs_dir)
    full = pd.concat([
        pd.read_csv(pre_dir / "train_monthly.csv", parse_dates=["TM"]),
        pd.read_csv(pre_dir / "test_monthly.csv", parse_dates=["TM"]),
    ], ignore_index=True).sort_values(["STN", "TM"])

    test_dates = pd.date_range(cfg["split"]["test_start"], cfg["split"]["test_end"], freq="MS")
    stations = stations or [s["name"] for s in cfg["stations"]]
    settings_list = settings_list or cfg["feature_settings"]

    summary = []
    for stn in stations:
        stn_df = full[full["station_name"] == stn]
        if stn_df.empty:
            continue
        for v in settings_list:
            preds = run_one_station(stn_df, v, settings, test_dates, cfg)
            if preds.empty:
                continue
            preds.to_csv(monthly_dir / f"LSTM_{v}_{stn}.csv", index=False)
            m = calc_all_metrics(preds["Observed"].to_numpy(), preds["Predicted"].to_numpy())
            summary.append({"Model": "LSTM", "Setting": v, "Station": stn, **m})
    df = pd.DataFrame(summary)
    df.to_csv(monthly_dir / "LSTM_summary.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run(args.config))
