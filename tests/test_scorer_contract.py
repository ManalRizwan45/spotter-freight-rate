"""Guards against a submission that fails format validation.

score.py rejects a file outright for a wrong column order, a missing id, or a
non-positive rate - and a rejected submission scores nothing regardless of model
quality. These rules are copied from score.py so a drift in either surfaces here.

The tests skip when the deliverables have not been generated yet; run
`python -m freight_rate.cli all` first.
"""
from __future__ import annotations

import pandas as pd
import pytest

from freight_rate.config import CONFIG

EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}
DECEMBER_COLUMNS = [
    "pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate",
]
DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")


@pytest.fixture
def predictions() -> pd.DataFrame:
    path = CONFIG.paths.predictions
    if not path.is_file():
        pytest.skip(f"{path.name} not generated yet - run `python -m freight_rate.cli predict`")
    return pd.read_csv(path)


@pytest.fixture
def december_predictions() -> pd.DataFrame:
    path = CONFIG.paths.december_predictions
    if not path.is_file():
        pytest.skip(f"{path.name} not generated yet - run `python -m freight_rate.cli december`")
    return pd.read_csv(path)


def test_predictions_columns_exact_and_ordered(predictions):
    assert list(predictions.columns) == ["load_id", "predicted_rate"]


def test_predictions_row_count(predictions):
    assert len(predictions) == EXPECTED_ROWS


def test_prediction_ids_match_the_validation_set(predictions):
    submitted = set(predictions["load_id"].astype(str))
    assert not submitted ^ EXPECTED_IDS, "ids do not match validation.csv exactly"
    assert not predictions["load_id"].duplicated().any()


def test_predicted_rates_are_positive_and_finite(predictions):
    rates = pd.to_numeric(predictions["predicted_rate"], errors="coerce")
    assert rates.notna().all()
    assert (rates > 0).all()


def test_december_keeps_original_columns_and_order(december_predictions):
    assert list(december_predictions.columns) == DECEMBER_COLUMNS


def test_december_covers_every_day_once(december_predictions):
    dates = pd.to_datetime(december_predictions["date"])
    assert len(december_predictions) == 31
    assert set(dates) == set(DECEMBER_DATES)
    assert not dates.duplicated().any()


def test_december_fixed_inputs_are_untouched(december_predictions):
    assert (december_predictions["pickup"] == "Lexington").all()
    assert (december_predictions["delivery"] == "Fort Wayne").all()
    assert (december_predictions["equipment"] == "Dry Van").all()
    assert (december_predictions["distance"] == 360).all()
    assert (december_predictions["weight"] == 32_000).all()


def test_december_rates_are_positive(december_predictions):
    rates = pd.to_numeric(december_predictions["predicted_rate"], errors="coerce")
    assert rates.notna().all()
    assert (rates > 0).all()
