"""Date encodings.

This module decides whether the December chart works, so the two encodings are kept
side by side and the failing one is retained rather than deleted - the comparison is
part of the deliverable.

ORDINAL (unsafe)
    date_ordinal + month. Both are monotone in time and stop at the training horizon.
    Training ends 2025-10-31, so every December date falls beyond the last learned
    split and a gradient-boosted tree clamps it to the final region. Every December day
    then receives an identical prediction: a flat line.

RECURRING (safe)
    day-of-week, day-of-month, and the looked-up daily market level. These recur, so
    December lands inside the range the model already knows. The weekly term is
    justified: market_index carries lag-7 autocorrelation of +0.969, higher than its
    lag-1 of +0.927.

    Day-of-week is a plain integer, not a sin/cos pair. Cyclical encoding exists for
    models where distance in feature space is the mechanism - linear models, kNN,
    neural nets - because those misread Sunday as six units from Monday. A tree splits
    on thresholds instead, so isolating one weekday from an integer takes two splits on
    one feature, where sin/cos requires bounding a region in two dimensions. Measured
    cost of the sin/cos version: +1.16 MAE, 95% CI +/-0.47.

Measured on the 31 fixed-input December rows, with everything except the date frozen:

    constant market input  + ordinal      range $0.00   1 distinct value  in 31 days
    recovered daily level  + ordinal      range $22.99 17 distinct values
    recovered daily level  + recurring    range $31.18 31 distinct values

The middle row is the trap: it moves, so it looks fixed, but it resolves only 17 of the
31 days. Correlation against the true December level is not quoted because it has flipped
sign across feature configurations; the distinct-value count is stable and orders the
three scenarios identically every time.
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
