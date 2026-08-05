"""The estimator: feature building, fitting, and prediction in dollars.

HistGradientBoostingRegressor is the choice because it handles NaN natively (weight has
300 training nulls that carry signal as missingness), trains on 48,000 rows in seconds,
and needs no scaling or one-hot expansion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import CONFIG, ModelParams
from ..features import DateEncoding
from ..features import build as build_features
from . import target


class RateModel:
    """Fit on labelled loads, predict rates in dollars.

    Holds the date encoding and market levels so that train and predict cannot
    accidentally disagree about how a date is represented.
    """

    def __init__(self, encoding: DateEncoding, market_levels: pd.Series | None = None,
                 params: ModelParams | None = None) -> None:
        self.encoding = encoding
        self.market_levels = market_levels
        self.params = params or CONFIG.model
        self._model = HistGradientBoostingRegressor(**self.params.as_kwargs())
        self._feature_names: list[str] | None = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return build_features(frame, self.encoding, self.market_levels)

    def fit(self, train: pd.DataFrame) -> RateModel:
        matrix = self._matrix(train)
        self._feature_names = list(matrix.columns)
        self._model.fit(matrix, target.encode(train))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._feature_names is None:
            raise RuntimeError("fit() must be called before predict()")
        matrix = self._matrix(frame)
        if list(matrix.columns) != self._feature_names:
            raise RuntimeError(
                "feature mismatch between fit and predict: "
                f"expected {self._feature_names}, got {list(matrix.columns)}"
            )
        return target.decode(self._model.predict(matrix), frame)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return self.fit(train).predict(test)
