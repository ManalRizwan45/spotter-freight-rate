from __future__ import annotations

import numpy as np
import pandas as pd

from freight_rate import cleaning


def test_negative_weights_are_corrected_and_flagged(sample_loads):
    cleaned, _ = cleaning.fit_apply(sample_loads)

    assert (cleaned["weight"].dropna() >= 0).all()
    assert cleaned["weight_was_negative"].sum() == 3
    # The magnitude must survive the sign fix.
    original = sample_loads["weight"].abs()
    pd.testing.assert_series_equal(
        cleaned["weight"], original, check_names=False
    )


def test_missing_weight_is_flagged_but_left_missing(sample_loads):
    cleaned, _ = cleaning.fit_apply(sample_loads)

    assert cleaned["weight_missing"].sum() == 3
    # Deliberately not imputed: the model splits on missingness natively.
    assert cleaned["weight"].isna().sum() == 3


def test_market_index_filled_from_same_day(sample_loads):
    cleaned, _ = cleaning.fit_apply(sample_loads)
    assert cleaned["market_index"].notna().all()

    # The filled rows should equal their own date's mean of the observed values.
    gap_rows = sample_loads.index[10:13]
    for row in gap_rows:
        date = sample_loads.loc[row, "date"]
        expected = sample_loads.loc[sample_loads.date == date, "market_index"].mean()
        assert np.isclose(cleaned.loc[row, "market_index"], expected)


def test_stats_are_fitted_on_train_and_reused(sample_loads):
    train = sample_loads[sample_loads.date < "2025-02-01"]
    test = sample_loads[sample_loads.date >= "2025-02-01"].copy()
    # Blank an entire date so the fallback is the only path available.
    blanked = test["date"].iloc[0]
    test.loc[test.date == blanked, "market_index"] = np.nan

    stats = cleaning.fit(train)
    cleaned = cleaning.apply(test, stats)

    filled = cleaned.loc[cleaned.date == blanked, "market_index"]
    assert np.allclose(filled, stats.market_index_fallback)
    # The fallback must come from train, never from the frame being cleaned.
    assert np.isclose(stats.market_index_fallback, train["market_index"].mean())


def test_apply_does_not_mutate_input(sample_loads):
    before = sample_loads.copy()
    cleaning.fit_apply(sample_loads)
    pd.testing.assert_frame_equal(sample_loads, before)
