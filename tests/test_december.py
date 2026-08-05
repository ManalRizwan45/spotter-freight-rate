from __future__ import annotations

import pandas as pd
import pytest

from freight_rate import december, market


def _levels(sample_loads, december_rows, constant=1.0):
    """Market levels covering both the training dates and December."""
    return pd.concat([
        market.daily_levels(sample_loads),
        pd.Series(constant, index=pd.DatetimeIndex(december_rows["date"])),
    ])


def test_prepare_resolves_every_required_column(sample_loads, december_rows):
    levels = _levels(sample_loads, december_rows)
    prepared = december.prepare(december_rows, sample_loads, sample_loads, levels)

    required = [
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "quote_signal", "market_index", "weight_missing", "weight_was_negative",
    ]
    for column in required:
        assert column in prepared, f"missing {column}"
        assert prepared[column].notna().all(), f"{column} contains nulls"


def test_all_thirty_one_days_survive(sample_loads, december_rows):
    levels = _levels(sample_loads, december_rows)
    prepared = december.prepare(december_rows, sample_loads, sample_loads, levels)
    assert len(prepared) == 31
    assert prepared["date"].nunique() == 31


def test_unknown_city_raises(sample_loads, december_rows):
    levels = _levels(sample_loads, december_rows)
    stranger = december_rows.assign(pickup="Atlantis")
    with pytest.raises(ValueError, match="no coordinates"):
        december.prepare(stranger, sample_loads, sample_loads, levels)


def test_uncovered_december_date_raises(sample_loads, december_rows):
    partial = market.daily_levels(sample_loads)  # training dates only
    with pytest.raises(ValueError, match="no market level"):
        december.prepare(december_rows, sample_loads, sample_loads, partial)


def test_constant_market_flattens_the_input(sample_loads, december_rows):
    levels = _levels(sample_loads, december_rows)
    naive = december.prepare(
        december_rows, sample_loads, sample_loads, levels, constant_market=True
    )
    assert naive["market_index"].nunique() == 1


def test_quote_signal_is_constant_across_the_month(sample_loads, december_rows):
    """The column has no daily structure, so a single conditional estimate is correct."""
    levels = _levels(sample_loads, december_rows)
    prepared = december.prepare(december_rows, sample_loads, sample_loads, levels)
    assert prepared["quote_signal"].nunique() == 1


def test_prepare_does_not_mutate_input(sample_loads, december_rows):
    levels = _levels(sample_loads, december_rows)
    before = december_rows.copy()
    december.prepare(december_rows, sample_loads, sample_loads, levels)
    pd.testing.assert_frame_equal(december_rows, before)
