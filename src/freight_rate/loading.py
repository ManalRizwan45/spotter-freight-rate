"""Reading the supplied CSVs, with schema enforcement.

Loading is kept separate from cleaning so that a schema failure (a renamed column, a
truncated file) surfaces as its own error rather than as a confusing downstream NaN.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CONFIG

FEATURE_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal",
]
TARGET_COLUMN = "posted_rate"
DECEMBER_COLUMNS = [
    "pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate",
]


class SchemaError(ValueError):
    """Raised when a supplied file does not match its expected columns."""


def _read(path: Path, expected: list[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found at {path}. The data/ directory ships with the assessment."
        )
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SchemaError(f"{label} is missing expected columns: {missing}")
    return frame


def load_train(path: Path | None = None) -> pd.DataFrame:
    """48,000 labelled loads, 2025-01-01 to 2025-10-31."""
    return _read(path or CONFIG.paths.train, FEATURE_COLUMNS + [TARGET_COLUMN], "train_test.csv")


def load_validation(path: Path | None = None) -> pd.DataFrame:
    """12,000 unlabelled loads, 2025-11-01 to 2025-12-31."""
    return _read(path or CONFIG.paths.validation, FEATURE_COLUMNS, "validation.csv")


def load_december_inputs(path: Path | None = None) -> pd.DataFrame:
    """31 fixed-input rows for the chart.

    Ships without market_index, quote_signal or coordinates; see december.py for how
    each is reconstructed.
    """
    return _read(
        path or CONFIG.paths.december_inputs, DECEMBER_COLUMNS, "december_chart_inputs.csv"
    )
