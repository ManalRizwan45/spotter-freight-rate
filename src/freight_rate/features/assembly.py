"""Composition of the feature blocks into a model matrix.

Only the date block varies between configurations. Everything else is shared, so a
comparison between two encodings isolates the date handling and nothing else.
"""
from __future__ import annotations

import pandas as pd

from . import geography, temporal
from .temporal import DateEncoding

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


def build_load(frame: pd.DataFrame) -> pd.DataFrame:
    """Load-level block: size, price signal, cargo.

    No log of distance: gradient-boosted trees bin by quantile and split on
    thresholds, so any monotone transform of a feature yields identical splits.
    log(distance) can add nothing distance does not already carry.

    No missing-weight flag either: HistGradientBoosting routes NaN to whichever
    branch fits better, so an explicit flag duplicates work the model already does.
    """
    out = pd.DataFrame(index=frame.index)
    out["distance"] = frame["distance"]
    # quote_signal is a per-load dollars-per-mile quote and the strongest single
    # feature by a wide margin: 79% of permutation importance.
    out["quote_signal"] = frame["quote_signal"]
    out["market_index"] = frame["market_index"]
    out["equipment_code"] = frame["equipment"].map(EQUIPMENT_CODES).astype(float)
    out["weight"] = frame["weight"]
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
