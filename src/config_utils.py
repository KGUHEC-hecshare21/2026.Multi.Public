"""Configuration loader used by all src/ modules.

The single source of truth is ``config/config.yaml`` at the repository root.
All other modules call :func:`load_config` so that paths, station codes,
training/testing periods, and model hyperparameters live in exactly one
place.  This makes the pipeline trivially reusable on other datasets.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load and return the YAML configuration as a plain dict."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve(cfg: dict, *parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return Path(cfg["_repo_root"]).joinpath(*parts)
