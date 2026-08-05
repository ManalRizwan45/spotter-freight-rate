from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate import cleaning
from freight_rate.features import temporal


def test_negative_weights_are_corrected_and_flagged(sample_loads):
    cleaned = cleaning.clean(sample_loads)

    assert (cleaned["weight"].dropna() >= 0).all()
    assert cleaned["weight_was_negative"].sum() == 3
    # The magnitude must survive the sign fix.
    pd.testing.assert_series_equal(
        cleaned["weight"], sample_loads["weight"].abs(), check_names=False
    )


def test_missing_weight_is_flagged_but_left_missing(sample_loads):
    cleaned = cleaning.clean(sample_loads)

    assert cleaned["weight_missing"].sum() == 3
    # Deliberately not imputed: the model splits on missingness natively.
    assert cleaned["weight"].isna().sum() == 3


def test_market_index_filled_from_its_own_date(sample_loads):
    cleaned = cleaning.clean(sample_loads)
    assert cleaned["market_index"].notna().all()

    for row in sample_loads.index[10:13]:
        date = sample_loads.loc[row, "date"]
        expected = sample_loads.loc[sample_loads.date == date, "market_index"].mean()
        assert np.isclose(cleaned.loc[row, "market_index"], expected)


def test_a_date_with_no_market_index_stays_missing(sample_loads):
    """No training-derived fallback: nothing is learned here, so nothing is imputed
    from another period. See the module docstring."""
    frame = sample_loads.copy()
    blanked = frame["date"].iloc[0]
    frame.loc[frame.date == blanked, "market_index"] = np.nan

    cleaned = cleaning.clean(frame)
    assert cleaned.loc[cleaned.date == blanked, "market_index"].isna().all()
    # Other dates are unaffected.
    assert cleaned.loc[cleaned.date != blanked, "market_index"].notna().all()


def test_that_gap_fails_loudly_downstream(sample_loads):
    """The empty date surfaces where it can be diagnosed, naming the offending dates."""
    from freight_rate import market

    frame = sample_loads.copy()
    blanked = frame["date"].iloc[0]
    frame.loc[frame.date == blanked, "market_index"] = np.nan
    cleaned = cleaning.clean(frame)

    levels = market.daily_levels(cleaned)
    with pytest.raises(ValueError, match="no market level"):
        temporal.build_cyclical(cleaned["date"], levels)


def test_clean_does_not_mutate_input(sample_loads):
    before = sample_loads.copy()
    cleaning.clean(sample_loads)
    pd.testing.assert_frame_equal(sample_loads, before)
