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

There is no fit/apply split here, because nothing is learned from the training data.
Every repair is a pure transform: abs() on a sign error, and a fill computed from the
row's own date group - information the assessment supplies for the prediction window
just as it does for training.

An earlier version carried a training-derived fallback for dates with no market_index
at all. It never fired: all 249 validation gaps are covered by their own date's mean,
and every one of the 61 validation dates has at least 159 observed values. Had it
fired it would have injected the training mean of 1.083 into a window whose mean is
0.927 - a spring-peak value in a soft-market month, on the column with the worst
train/validation shift in the dataset.

A date that genuinely has no market_index now stays NaN and fails loudly in
features.temporal.build_cyclical, which names the offending dates. That is a better
place to find out than silently inheriting a number from another market regime.
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
