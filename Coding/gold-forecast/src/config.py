"""Config loading + canonical project paths. Deliberately tiny."""
from __future__ import annotations

from pathlib import Path
import yaml

# Project root = parent of src/
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml. Defaults to the one at project root."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (RAW, PROCESSED, REPORTS, REPORTS / "data_quality"):
        d.mkdir(parents=True, exist_ok=True)
