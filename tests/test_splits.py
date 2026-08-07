"""Tests for validation splits.

The forward-chaining guarantee - test always begins after train ends - is the property
that keeps reported error honest, so it is asserted directly rather than assumed.
"""
from __future__ import annotations

import pytest

from freight_rate.config import Fold
from freight_rate.modeling import splits

FOLDS = (
    Fold("2025-01-01", "2025-01-20", "2025-01-21", "2025-01-31"),
    Fold("2025-01-01", "2025-01-31", "2025-02-01", "2025-02-20"),
)


def test_forward_folds_never_leak_across_time(sample_loads):
    for split in splits.forward_folds(sample_loads, FOLDS):
        assert split.train["date"].max() < split.test["date"].min()


def test_forward_folds_share_no_rows(sample_loads):
    for split in splits.forward_folds(sample_loads, FOLDS):
        assert set(split.train.load_id).isdisjoint(split.test.load_id)


def test_empty_fold_is_rejected(sample_loads):
    impossible = (Fold("2025-06-01", "2025-06-30", "2025-07-01", "2025-07-31"),)
    with pytest.raises(ValueError, match="empty side"):
        list(splits.forward_folds(sample_loads, impossible))
