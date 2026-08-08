"""The estimator: feature building, fitting, and prediction in dollars.

RandomForestRegressor is the choice on the evidence: under forward chaining it beats
HistGradientBoosting by $10.85 +/- 0.86 MAE, with the same sign in all three folds.

It cannot read NaN, so nulls are filled with medians taken from the TRAINING matrix only
and reused unchanged at predict time, meaning a validation row can never influence its
own imputed value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import CONFIG
from ..features import DateEncoding
from ..features import build as build_features
from . import target


class RateModel:
    """Fit on labelled loads, predict rates in dollars.

    Holds the date encoding, the market levels and the imputation medians, so train and
    predict cannot disagree about how a row is represented.
    """

    def __init__(self, encoding: DateEncoding, market_levels: pd.Series | None = None) -> None:
        self.encoding = encoding
        self.market_levels = market_levels
        self._model = RandomForestRegressor(**CONFIG.model.as_kwargs())
        self._feature_names: list[str] | None = None
        self._medians: pd.Series | None = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return build_features(frame, self.encoding, self.market_levels)

    def fit(self, train: pd.DataFrame) -> RateModel:
        matrix = self._matrix(train)
        self._feature_names = list(matrix.columns)
        self._medians = matrix.median()
        unusable = self._medians.index[self._medians.isna()].tolist()
        if unusable:
            raise ValueError(f"no median available to impute (all-null in train): {unusable}")
        self._model.fit(matrix.fillna(self._medians), target.encode(train))
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
        return target.decode(self._model.predict(matrix.fillna(self._medians)), frame)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return self.fit(train).predict(test)
