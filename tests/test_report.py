"""Tests for the evaluation panel.

A metric that is wrong does not crash - it prints a plausible number and the whole
report becomes fiction. So the formulas are pinned against hand-computed values chosen
so that a wrong implementation gives a different answer: WAPE is checked on loads of
unequal size, where a mean-of-percentages would disagree with it, and bias is checked
for sign rather than magnitude.

The other failure mode is alignment. `by_segment` indexes a positional prediction array
with group positions taken from a frame that arrives pooled from several folds. Getting
that wrong scores each group against another group's predictions and still returns a
full table, so it is tested directly with an index that does not match position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from freight_rate.config import Fold
from freight_rate.evaluation import report
from freight_rate.features import DateEncoding
from freight_rate.modeling import RateModel, splits

FOLDS = (
    Fold("2025-01-01", "2025-01-20", "2025-01-21", "2025-01-31"),
    Fold("2025-01-01", "2025-01-31", "2025-02-01", "2025-02-20"),
)


def test_dollar_metrics_match_hand_computed_values():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 300.0])  # errors +10, -20, 0
    scores = report.panel(actual, predicted)

    assert scores["MAE $"] == pytest.approx(10.0)
    assert scores["RMSE $"] == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert scores["bias $"] == pytest.approx(-10 / 3)


def test_bias_is_signed_and_positive_when_the_model_quotes_high():
    actual = np.array([100.0, 200.0])
    assert report.panel(actual, actual * 1.05)["bias $"] > 0
    assert report.panel(actual, actual * 0.95)["bias $"] < 0
    assert report.panel(actual, actual * 1.05)["bias %"] == pytest.approx(5.0)


def test_wape_weights_by_dollars_where_mape_does_not():
    """One 10%-wrong cheap load and one 1%-wrong dear load.

    MAPE averages the percentages and reads 5.5%. WAPE divides total error by total
    dollars, so the dear load dominates. A WAPE computed as a mean of percentages would
    return the MAPE and this test would fail.
    """
    actual = np.array([100.0, 10_000.0])
    predicted = np.array([110.0, 10_100.0])
    scores = report.panel(actual, predicted)

    assert scores["MAPE %"] == pytest.approx(5.5)
    assert scores["WAPE %"] == pytest.approx((10 + 100) / 10_100 * 100)


def test_r2_matches_scikit_learn():
    rng = np.random.default_rng(0)
    actual = rng.uniform(500, 5000, 400)
    predicted = actual + rng.normal(0, 120, 400)
    assert report.panel(actual, predicted)["R2"] == pytest.approx(r2_score(actual, predicted))


def test_r2_is_undefined_rather_than_infinite_on_constant_actuals():
    """A one-row segment has no spread to explain. Dividing by it would emit inf and a
    warning, and the rest of the panel would still be worth having."""
    scores = report.panel(np.array([1000.0]), np.array([1100.0]))
    assert np.isnan(scores["R2"])
    assert scores["MAE $"] == pytest.approx(100.0)


def test_percentile_and_hit_rate_bands():
    """Nine loads at 1% error and one at 50%: the median is untouched, P90 is not."""
    actual = np.full(10, 1000.0)
    predicted = np.concatenate([np.full(9, 1010.0), [1500.0]])
    scores = report.panel(actual, predicted)

    assert scores["median APE %"] == pytest.approx(1.0)
    assert scores["P90 APE %"] > 1.0
    assert scores["within 5%"] == pytest.approx(90.0)
    assert scores["within 25%"] == pytest.approx(90.0)


def test_hit_rate_band_is_inclusive_at_its_edge():
    """A load exactly 10% out counts as within 10%, so the band is not one row narrower
    than it reads."""
    scores = report.panel(np.array([1000.0]), np.array([1100.0]))
    assert scores["within 10%"] == pytest.approx(100.0)
    assert scores["within 5%"] == pytest.approx(0.0)


@pytest.mark.parametrize("broken", [np.array([0.0, 100.0]), np.array([-50.0, 100.0])])
def test_non_positive_actuals_are_rejected(broken):
    with pytest.raises(ValueError, match="positive"):
        report.panel(broken, np.array([10.0, 100.0]))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        report.panel(np.array([100.0, 200.0]), np.array([100.0]))


def test_compare_scores_each_candidate_against_the_reference():
    actual = np.array([100.0, 200.0, 300.0])
    table = report.compare(
        actual,
        {"half the error": actual + 10, "reference": actual + 20},
        reference="reference",
    )
    assert table.loc["MAE $", "reference"] == pytest.approx(20.0)
    assert table.loc["MAE cut vs reference %", "half the error"] == pytest.approx(50.0)
    assert table.loc["MAE cut vs reference %", "reference"] == pytest.approx(0.0)


def test_compare_rejects_an_unknown_reference():
    actual = np.array([100.0, 200.0])
    with pytest.raises(KeyError):
        report.compare(actual, {"model": actual}, reference="quote only")


def test_baselines_are_the_documented_two(sample_loads):
    values = report.baselines(sample_loads)
    assert set(values) == {"mean rate", "quote only"}
    assert values["mean rate"] == pytest.approx(sample_loads.posted_rate.mean())
    assert values["quote only"] == pytest.approx(
        (sample_loads.distance * sample_loads.quote_signal).to_numpy()
    )


def test_by_segment_uses_position_not_index_labels():
    """The frame's index is scrambled and duplicated, as a pooled multi-fold frame can
    be. Only the Reefer rows may reach the Reefer score."""
    frame = pd.DataFrame(
        {
            "posted_rate": [100.0, 100.0, 100.0, 100.0],
            "equipment": ["Dry Van"] * 2 + ["Reefer"] * 2,
        },
        index=[7, 7, 0, 3],
    )
    predicted = np.array([100.0, 100.0, 150.0, 150.0])
    table = report.by_segment(frame, predicted, "equipment")

    assert table.loc["Dry Van", "MAE $"] == pytest.approx(0.0)
    assert table.loc["Reefer", "MAE $"] == pytest.approx(50.0)


def test_by_segment_partitions_every_row(sample_loads):
    predicted = sample_loads.posted_rate.to_numpy() * 1.02
    table = report.by_segment(sample_loads, predicted, "equipment")
    assert table["n"].sum() == len(sample_loads)


def test_distance_bands_are_left_closed():
    """250 belongs to the 250-500 band, not to the one below it."""
    banded = report.distance_band(pd.Series([0.0, 249.9, 250.0, 999.9, 1000.0, 5000.0]))
    assert list(banded) == [
        "under 250 mi", "under 250 mi", "250-500 mi", "500-1000 mi", "1000+ mi", "1000+ mi",
    ]


def test_segments_returns_the_three_cuts(sample_loads):
    predicted = sample_loads.posted_rate.to_numpy() * 1.02
    cuts = report.segments(sample_loads, predicted)
    assert set(cuts) == {"equipment", "distance", "month"}
    assert list(cuts["month"].index) == ["2025-01", "2025-02"]


def test_out_of_fold_returns_one_prediction_per_pooled_row(sample_loads, market_levels):
    def factory() -> RateModel:
        return RateModel(DateEncoding.RECURRING, market_levels)

    pooled, predicted = report.out_of_fold(splits.forward_folds(sample_loads, FOLDS), factory)

    expected = sum(len(split.test) for split in splits.forward_folds(sample_loads, FOLDS))
    assert len(pooled) == len(predicted) == expected
    assert not pooled.load_id.duplicated().any()
    assert set(pooled.fold) == {fold.label for fold in FOLDS}


def test_out_of_fold_rejects_overlapping_test_blocks(sample_loads, market_levels):
    """Two folds testing the same window would double-count those loads in every metric.

    Reusing one fold twice is the cheapest way to produce that overlap; the guard has to
    fire on it rather than silently score those loads at twice their weight.
    """
    def factory() -> RateModel:
        return RateModel(DateEncoding.RECURRING, market_levels)

    repeated = list(splits.forward_folds(sample_loads, FOLDS[:1])) * 2
    with pytest.raises(ValueError, match="overlap"):
        report.out_of_fold(repeated, factory)
