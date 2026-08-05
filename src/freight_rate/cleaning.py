"""Data-quality repairs.

Issues addressed, with counts from the supplied files:

  negative weight      292 train / 145 validation. Magnitudes are plausible and the
                       distribution of their absolute values matches the positive rows,
                       so these read as sign errors rather than corrupt records.
  missing weight       300 / 165. Left as NaN: HistGradientBoosting splits on
                       missingness natively, which beats imputing a value the model
                       then cannot distinguish from a real one. A flag is added anyway.
  missing market_index 374 / 249. Filled from the same date's mean, because 97.8% of
                       the column's variance is explained by date alone.

Nothing is learned from the data, so there is no fit/apply pair - every repair is a
pure transform. A date with no market_index at all stays NaN and fails in
features.temporal.build_cyclical, which names it.
"""
from __future__ import annotations

import pandas as pd


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a repaired copy. The input frame is never modified."""
    out = frame.copy()

    # Flag before fixing: once the sign is corrected the evidence is gone.
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)

    # A row's own date group is available at prediction time, so this is not leakage.
    same_day_mean = out.groupby("date")["market_index"].transform("mean")
    out["market_index"] = out["market_index"].fillna(same_day_mean)

    return out
