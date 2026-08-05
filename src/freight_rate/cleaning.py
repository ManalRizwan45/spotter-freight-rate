"""Data-quality repairs, split into a fit step and an apply step.

The split matters for leakage. Anything *learned* from data (a fallback mean, a median)
is fitted on the training fold only and then applied unchanged elsewhere. Repairs that
need no learning (taking abs of a sign error, filling from a row's own date group) are
pure transforms and carry no such risk.

Issues addressed, with counts from the supplied files:

  negative weight     292 train / 145 validation. Magnitudes are plausible and the
                      distribution of their absolute values matches the positive rows,
                      so these read as sign errors rather than corrupt records.
  missing weight      300 / 165. Left as NaN: HistGradientBoosting splits on missingness
                      natively, which beats imputing a value the model then cannot
                      distinguish from a real one. A flag column is added regardless.
  missing market_index 374 / 249. Filled from the same date's mean, because 97.8% of the
                      column's variance is explained by date alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CleaningStats:
    """Quantities learned from the training fold, applied unchanged to any other frame."""
    market_index_fallback: float


def fit(train: pd.DataFrame) -> CleaningStats:
    return CleaningStats(market_index_fallback=float(train["market_index"].mean()))


def apply(frame: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    out = frame.copy()

    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)

    # A row's own date group is available at prediction time, so this is not leakage.
    same_day_mean = out.groupby("date")["market_index"].transform("mean")
    out["market_index"] = out["market_index"].fillna(same_day_mean)
    # Only reached if an entire date is missing the column.
    out["market_index"] = out["market_index"].fillna(stats.market_index_fallback)

    return out


def fit_apply(train: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Convenience for the common case of cleaning the training frame itself."""
    stats = fit(train)
    return apply(train, stats), stats
