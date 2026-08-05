"""Composition of the feature blocks into a model matrix.

Only the date block varies between configurations. Everything else is shared, so a
comparison between two encodings isolates the date handling and nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import geography, temporal
from .temporal import DateEncoding

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


def build_load(frame: pd.DataFrame) -> pd.DataFrame:
    """Load-level block: size, price signal, cargo."""
    out = pd.DataFrame(index=frame.index)
    out["distance"] = frame["distance"]
    out["log_distance"] = np.log1p(frame["distance"])
    # quote_signal is a per-load dollars-per-mile quote and the strongest single
    # feature: distance * quote_signal lands within 2% on half the training rows.
    out["quote_signal"] = frame["quote_signal"]
    out["market_index"] = frame["market_index"]
    out["equipment_code"] = frame["equipment"].map(EQUIPMENT_CODES).astype(float)
    out["weight"] = frame["weight"]
    out["weight_missing"] = frame.get("weight_missing", 0)
    out["weight_was_negative"] = frame.get("weight_was_negative", 0)
    return out


def build(frame: pd.DataFrame, encoding: DateEncoding,
          market_levels: pd.Series | None = None) -> pd.DataFrame:
    """Full feature matrix for `frame` under the given date encoding."""
    unknown = set(frame["equipment"].dropna()) - set(EQUIPMENT_CODES)
    if unknown:
        raise ValueError(f"unmapped equipment types: {sorted(unknown)}")

    return pd.concat(
        [
            build_load(frame),
            geography.build(frame),
            temporal.build(frame["date"], encoding, market_levels),
        ],
        axis=1,
    )
