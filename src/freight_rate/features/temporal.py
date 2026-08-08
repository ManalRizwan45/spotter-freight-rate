"""Date encodings.

This module decides whether the December chart works, so both encodings are kept side by
side and the failing one is retained: the comparison is part of the deliverable.

ORDINAL (unsafe)
    date_ordinal + month, both monotone in time. Training ends 2025-10-31, so every
    December date falls beyond the last learned split and a tree clamps it to the final
    region: one identical prediction for all 31 days. No tree can extrapolate past a
    threshold it never saw.

RECURRING (safe)
    day-of-week, day-of-month, and the looked-up daily market level. These recur, so
    December lands inside the range the model knows. The weekly term is justified:
    market_index carries lag-7 autocorrelation of +0.969 against lag-1 of +0.927.

    Day-of-week is a plain integer, not sin/cos. Cyclical encoding matters for models
    where distance in feature space is the mechanism (linear, kNN, neural nets). A tree
    splits on thresholds, so isolating one weekday takes two splits on one feature where
    sin/cos needs a region in two dimensions. Measured cost of sin/cos: +0.29 MAE
    (+/-0.08).

Range and distinct-value counts for the three December scenarios are tabulated in
README.md.
"""
from __future__ import annotations

from enum import Enum

import pandas as pd

EPOCH = pd.Timestamp("2025-01-01")


class DateEncoding(str, Enum):
    ORDINAL = "ordinal"
    RECURRING = "recurring"


def build_ordinal(dates: pd.Series) -> pd.DataFrame:
    """Extrapolation-unsafe encoding. Retained to demonstrate the failure."""
    out = pd.DataFrame(index=dates.index)
    out["date_ordinal"] = (dates - EPOCH).dt.days
    out["month"] = dates.dt.month
    return out


def build_recurring(dates: pd.Series, market_levels: pd.Series) -> pd.DataFrame:
    """Extrapolation-safe encoding.

    Raises if `market_levels` cannot cover every date, rather than silently emitting
    NaN that a tree would absorb into a default branch.
    """
    out = pd.DataFrame(index=dates.index)
    out["day_of_week"] = dates.dt.dayofweek
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
        raise ValueError("recurring encoding requires market_levels")
    return build_recurring(dates, market_levels)
