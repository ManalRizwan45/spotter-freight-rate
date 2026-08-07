"""Tests for reading and schema enforcement.

The contract worth pinning: every missing column raises SchemaError, uniformly. An
earlier version passed parse_dates= to read_csv, which made a missing 'date' column
raise pandas' own ValueError from inside read_csv - so it never reached the schema
check, and a caller writing `except SchemaError` would have missed it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from freight_rate import loading
from freight_rate.loading import SchemaError


def _valid_train_csv(path):
    """A minimal but complete train_test.csv."""
    frame = pd.DataFrame({
        "load_id": ["TR-000001", "TR-000002"],
        "pickup": ["Lexington", "Dallas"],
        "delivery": ["Fort Wayne", "Boston"],
        "pickup_lat": [38.0, 31.8], "pickup_lon": [-84.5, -94.4],
        "delivery_lat": [41.1, 42.4], "delivery_lon": [-85.1, -71.1],
        "distance": [360.0, 1800.0],
        "equipment": ["Dry Van", "Reefer"],
        "weight": [32000.0, 28000.0],
        "date": ["2025-01-01", "2025-01-02"],
        "market_index": [0.96, 0.98],
        "quote_signal": [2.10, 2.20],
        "posted_rate": [756.0, 3960.0],
    })
    frame.to_csv(path, index=False)
    return frame


def test_valid_file_loads(tmp_path):
    path = tmp_path / "train.csv"
    _valid_train_csv(path)
    frame = loading.load_train(path)
    assert len(frame) == 2


def test_date_is_parsed_to_datetime(tmp_path):
    path = tmp_path / "train.csv"
    _valid_train_csv(path)
    frame = loading.load_train(path)
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame["date"].iloc[0] == pd.Timestamp("2025-01-01")


def test_missing_ordinary_column_raises_schema_error(tmp_path):
    path = tmp_path / "train.csv"
    _valid_train_csv(path).drop(columns=["equipment"]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="equipment"):
        loading.load_train(path)


def test_missing_date_column_raises_schema_error(tmp_path):
    """Regression: 'date' must not be special-cased into pandas' own ValueError."""
    path = tmp_path / "train.csv"
    _valid_train_csv(path).drop(columns=["date"]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="date"):
        loading.load_train(path)


def test_missing_target_raises_for_train_but_not_validation(tmp_path):
    """validation.csv is train_test.csv minus posted_rate, so only train requires it."""
    path = tmp_path / "frame.csv"
    _valid_train_csv(path).drop(columns=["posted_rate"]).to_csv(path, index=False)

    with pytest.raises(SchemaError, match="posted_rate"):
        loading.load_train(path)
    assert len(loading.load_validation(path)) == 2


def test_every_missing_column_is_reported_at_once(tmp_path):
    path = tmp_path / "train.csv"
    frame = _valid_train_csv(path).drop(columns=["equipment", "weight", "distance"])
    frame.to_csv(path, index=False)

    with pytest.raises(SchemaError) as caught:
        loading.load_train(path)
    message = str(caught.value)
    assert all(column in message for column in ["equipment", "weight", "distance"])


def test_extra_columns_are_accepted(tmp_path):
    """The check is permissive by design: required columns present, not an exact match."""
    path = tmp_path / "train.csv"
    _valid_train_csv(path).assign(unexpected=1).to_csv(path, index=False)
    assert "unexpected" in loading.load_train(path).columns


def test_column_order_does_not_matter(tmp_path):
    path = tmp_path / "train.csv"
    frame = _valid_train_csv(path)
    frame[list(reversed(frame.columns))].to_csv(path, index=False)
    assert len(loading.load_train(path)) == 2


def test_missing_file_raises_with_a_useful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_test.csv not found"):
        loading.load_train(tmp_path / "absent.csv")
