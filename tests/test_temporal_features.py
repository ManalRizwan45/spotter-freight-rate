"""Tests for the date encodings.

The central property is extrapolation safety: features generated for December must fall
inside the range the model saw during training. Ordinal encoding violates this by
construction, which is why the December chart flattens.
"""
from __future__ import annotations

import pandas as pd
import pytest

from freight_rate.features import temporal
from freight_rate.features.temporal import DateEncoding


def test_ordinal_encoding_leaves_the_training_range(sample_loads, december_rows):
    train = temporal.build_ordinal(sample_loads["date"])
    december = temporal.build_ordinal(december_rows["date"])

    # Every December value sits beyond every training value - nothing to interpolate
    # between, so a tree clamps the lot to its final split.
    assert december["date_ordinal"].min() > train["date_ordinal"].max()
    assert december["month"].min() > train["month"].max()


def test_recurring_encoding_stays_inside_the_training_range(
    sample_loads, december_rows, market_levels
):
    levels = pd.concat([
        market_levels,
        pd.Series(1.0, index=pd.DatetimeIndex(december_rows["date"])),
    ])
    train = temporal.build_recurring(sample_loads["date"], levels)
    december = temporal.build_recurring(december_rows["date"], levels)

    for column in ["day_of_week", "day_of_month"]:
        assert december[column].min() >= train[column].min()
        assert december[column].max() <= train[column].max()


def test_dates_seven_days_apart_encode_identically():
    """The weekly term must repeat, which is what carries it into December."""
    dates = pd.Series(pd.to_datetime(["2025-12-01", "2025-12-08"]))
    levels = pd.Series(1.0, index=pd.DatetimeIndex(dates))
    encoded = temporal.build_recurring(dates, levels)
    assert encoded["day_of_week"].iloc[0] == encoded["day_of_week"].iloc[1]


def test_missing_market_level_raises_rather_than_emitting_nan(december_rows):
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="no market level"):
        temporal.build_recurring(december_rows["date"], empty)


def test_recurring_requires_market_levels(sample_loads):
    with pytest.raises(ValueError, match="requires market_levels"):
        temporal.build(sample_loads["date"], DateEncoding.RECURRING, None)
