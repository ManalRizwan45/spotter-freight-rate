"""Date encodings.

This module decides whether the December chart works, so the two encodings are kept
side by side and the failing one is retained rather than deleted - the comparison is
part of the deliverable.

ORDINAL (unsafe)
    date_ordinal + month. Both are monotone in time and stop at the training horizon.
    Training ends 2025-10-31, so every December date falls beyond the last learned
    split and a gradient-boosted tree clamps it to the final region. Every December day
    then receives an identical prediction: a flat line.

CYCLICAL (safe)
    day-of-week as sin/cos, plus day-of-month and the looked-up daily market level.
    These recur, so December lands inside the range the model already knows. The weekly
    term is justified: market_index carries lag-7 autocorrelation of +0.969, higher than
    its lag-1 of +0.927.

Measured on the 31 fixed-input December rows, with everything except the date frozen:

    constant market input  + ordinal    range $0.00   1 distinct value  in 31 days
    recovered daily level  + ordinal    range $7.18  11 distinct values, corr +0.349
    recovered daily level  + cyclical   range $19.77 30 distinct values, corr +0.672

The middle row is the trap: it moves, so it looks fixed, but it manages only 11 distinct
values and tracks the market at half the strength of the cyclical encoding.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2025-01-01")
DAYS_IN_WEEK = 7


class DateEncoding(str, Enum):
    ORDINAL = "ordinal"
    CYCLICAL = "cyclical"


def build_ordinal(dates: pd.Series) -> pd.DataFrame:
    """Extrapolation-unsafe encoding. Retained to demonstrate the failure."""
    out = pd.DataFrame(index=dates.index)
    out["date_ordinal"] = (dates - EPOCH).dt.days
    out["month"] = dates.dt.month
    return out


def build_cyclical(dates: pd.Series, market_levels: pd.Series) -> pd.DataFrame:
    """Extrapolation-safe encoding.

    Raises if `market_levels` cannot cover every date, rather than silently emitting
    NaN that a tree would absorb into a default branch.
    """
    out = pd.DataFrame(index=dates.index)
    day_of_week = dates.dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / DAYS_IN_WEEK)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / DAYS_IN_WEEK)
    out["day_of_month"] = dates.dt.day

    level = dates.map(market_levels)
    if level.isna().any():
        gaps = sorted(dates[level.isna()].dt.date.unique())
        shown = f"{gaps[:5]}{' ...' if len(gaps) > 5 else ''}"
        raise ValueError(f"no market level available for: {shown}")
    out["daily_market_level"] = level
    return out


def build(dates: pd.Series, encoding: DateEncoding,
          market_levels: pd.Series | None = None) -> pd.DataFrame:
    if encoding is DateEncoding.ORDINAL:
        return build_ordinal(dates)
    if market_levels is None:
        raise ValueError("cyclical encoding requires market_levels")
    return build_cyclical(dates, market_levels)
