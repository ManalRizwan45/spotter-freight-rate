"""Tests for the configuration itself.

Config is data, so it can carry bugs that no amount of testing the pipeline would
catch: a hyperparameter that never reaches the model, or a fold whose test block starts
before training ends. These pin those invariants.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from freight_rate.config import CONFIG, ModelParams


def test_as_kwargs_covers_every_field():
    """Regression: a hand-written dict here once risked dropping new hyperparameters."""
    declared = {f.name for f in dataclasses.fields(ModelParams)}
    assert set(CONFIG.model.as_kwargs()) == declared


def test_as_kwargs_are_all_real_sklearn_parameters():
    """Catches a renamed or removed parameter on a scikit-learn upgrade."""
    accepted = set(HistGradientBoostingRegressor().get_params())
    unknown = set(CONFIG.model.as_kwargs()) - accepted
    assert not unknown, f"not accepted by HistGradientBoostingRegressor: {sorted(unknown)}"


def test_early_stopping_is_off():
    """sklearn's 'auto' enables early stopping above 10,000 rows, and the training set
    has 48,000. That would carve out an internal RANDOM validation split - exactly the
    split strategy this project argues is invalid here."""
    assert CONFIG.model.early_stopping is False


def test_forward_folds_never_overlap_in_time():
    for fold in CONFIG.folds:
        train_end = pd.Timestamp(fold.train_end)
        test_start = pd.Timestamp(fold.test_start)
        test_end = pd.Timestamp(fold.test_end)
        assert train_end < test_start, f"{fold} tests before training ends"
        assert test_start <= test_end, f"{fold} has an inverted test window"


def test_forward_folds_use_an_expanding_window():
    ends = [pd.Timestamp(fold.train_end) for fold in CONFIG.folds]
    assert ends == sorted(ends)
    assert len({fold.train_start for fold in CONFIG.folds}) == 1, "all folds start together"


def test_folds_stay_inside_the_labelled_period():
    """Training data runs 2025-01-01 to 2025-10-31; no fold may reach beyond it."""
    for fold in CONFIG.folds:
        assert pd.Timestamp(fold.train_start) >= pd.Timestamp("2025-01-01")
        assert pd.Timestamp(fold.test_end) <= pd.Timestamp("2025-10-31")


