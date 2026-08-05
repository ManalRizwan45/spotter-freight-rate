"""Scoring.

Four metrics are reported because they disagree in a way that is informative here.
The rate distribution has a long right tail, so MAE and median APE describe the typical
load while RMSE and mean MAPE are pulled by the ~0.7% of rows above $4/mile. A change
that improves RMSE while worsening median APE has traded typical accuracy for tail
accuracy, which is a decision worth seeing rather than averaging away.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..modeling.estimator import RateModel
from ..modeling.splits import Split


@dataclass(frozen=True)
class Scores:
    mae: float
    rmse: float
    mape: float
    median_ape: float

    def as_row(self) -> dict:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE %": self.mape,
            "median APE %": self.median_ape,
        }


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> Scores:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    error = predicted - actual
    absolute_percentage = np.abs(error) / actual * 100
    return Scores(
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt((error ** 2).mean())),
        mape=float(absolute_percentage.mean()),
        median_ape=float(np.median(absolute_percentage)),
    )


def evaluate_splits(splits: Iterable[Split], model_factory) -> pd.DataFrame:
    """Fit a fresh model per split and score it. `model_factory` returns a RateModel."""
    rows = []
    for split in splits:
        model: RateModel = model_factory()
        predicted = model.fit_predict(split.train, split.test)
        scores = evaluate(split.test["posted_rate"].to_numpy(), predicted)
        rows.append({
            "fold": split.label,
            "n_train": len(split.train),
            "n_test": len(split.test),
            **scores.as_row(),
        })
    return pd.DataFrame(rows)


METRIC_COLUMNS = ["MAE", "RMSE", "MAPE %", "median APE %"]


def summarise(table: pd.DataFrame, label: str) -> pd.DataFrame:
    """Collapse a per-fold table to one row of mean metrics."""
    return pd.DataFrame([{"strategy": label, **table[METRIC_COLUMNS].mean().to_dict()}])
