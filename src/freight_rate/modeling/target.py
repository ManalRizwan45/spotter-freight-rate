"""Target transform.

The model predicts log(posted_rate / (distance * quote_signal)) rather than the rate
itself. Two reasons:

  The quote is already a strong estimate. distance * quote_signal lands within 2% of
  the actual rate on half the training rows and gives a median absolute error of 2.0%
  with no model at all. Predicting the ratio means the model only has to learn the
  deviation, which is where the remaining signal lives - equipment, geography, and
  market conditions.

  The rate has a long right tail. Median rate-per-mile is $2.15 but roughly 0.7% of
  rows exceed $4/mile, reaching $14.1. Under squared error on raw dollars those rows
  would dominate the loss. In log space they are ordinary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseline(frame: pd.DataFrame) -> pd.Series:
    """The quoted rate for the whole load: the anchor the model corrects."""
    return frame["distance"] * frame["quote_signal"]


def encode(frame: pd.DataFrame) -> pd.Series:
    """Actual rate -> log ratio against the quote."""
    if "posted_rate" not in frame:
        raise KeyError("encode() requires the labelled posted_rate column")
    return np.log(frame["posted_rate"] / baseline(frame))


def decode(predicted_log_ratio: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Log ratio -> dollars."""
    return np.exp(predicted_log_ratio) * baseline(frame).to_numpy()
