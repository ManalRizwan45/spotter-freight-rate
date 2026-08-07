"""Reconstructing the columns december_chart_inputs.csv does not ship with.

The file carries seven columns - pickup, delivery, distance, equipment, weight, date,
predicted_rate - and the model needs coordinates, market_index and quote_signal. Each
is recovered differently, and the reasoning differs in kind:

  coordinates    Exact. Fixed per city across the dataset, so a name lookup recovers
                 them with no estimation at all.

  market_index   Estimated, precisely. Take each date's daily level from validation.csv,
                 which covers all 31 December days. The averaging cancels per-load noise
                 to within about +/-0.002.

  quote_signal   Estimated, coarsely, and held constant across the month. The column has
                 almost no daily structure - day-to-day std of 0.016 against a within-day
                 spread of 0.219 - so there is no daily value to recover. A conditional
                 median over comparable loads (same equipment, similar length) is the
                 honest estimate.

Holding quote_signal constant has a useful side effect: every bit of movement in the
resulting December curve comes from the date handling, which is exactly what the chart
is built to expose.
"""
from __future__ import annotations

import pandas as pd

from .config import CONFIG
from .features import geography


def _resolve_coordinates(frame: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    unknown = (set(frame["pickup"]) | set(frame["delivery"])) - set(coordinates.index)
    if unknown:
        raise ValueError(f"no coordinates available for: {sorted(unknown)}")
    out = frame.copy()
    out["pickup_lat"] = out["pickup"].map(coordinates["lat"])
    out["pickup_lon"] = out["pickup"].map(coordinates["lon"])
    out["delivery_lat"] = out["delivery"].map(coordinates["lat"])
    out["delivery_lon"] = out["delivery"].map(coordinates["lon"])
    return out


def estimate_quote_signal(train: pd.DataFrame, equipment: str,
                          distance_band: tuple[int, int]) -> float:
    """Median quote_signal over comparable training loads."""
    low, high = distance_band
    comparable = train[
        (train["equipment"] == equipment) & (train["distance"].between(low, high))
    ]
    if comparable.empty:
        raise ValueError(f"no comparable loads for {equipment} in {distance_band} miles")
    return float(comparable["quote_signal"].median())


def prepare(december: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame,
            market_levels: pd.Series, constant_market: bool = False) -> pd.DataFrame:
    """Return the December rows with every column the model requires.

    `constant_market` substitutes the global training mean for the daily level. That is
    the naive path - it is retained because it is what happens when the recoverability
    of the daily level goes unnoticed, and it produces a perfectly flat chart.
    """
    settings = CONFIG.december

    frame = _resolve_coordinates(
        december, geography.city_coordinates(train, validation)
    )

    frame["quote_signal"] = estimate_quote_signal(
        train, settings.reference_equipment, settings.distance_band
    )

    if constant_market:
        frame["market_index"] = float(train["market_index"].mean())
    else:
        level = frame["date"].map(market_levels)
        if level.isna().any():
            gaps = sorted(frame.loc[level.isna(), "date"].dt.date.unique())
            raise ValueError(f"no market level for December dates: {gaps}")
        frame["market_index"] = level

    return frame
