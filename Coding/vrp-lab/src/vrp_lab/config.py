"""Config loading + canonical project paths. Deliberately tiny (QL_MAK pattern)."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

# Project root = parents[2] of this file (src/vrp_lab/config.py -> repo root)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load config/settings.yaml (or an explicit path) into a dict."""
    path = Path(path) if path else CONFIG_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create all data/report directories if missing. Idempotent."""
    for d in (RAW, PROCESSED, CACHE, REPORTS):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Project-standard logger: INFO to stderr, no duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
