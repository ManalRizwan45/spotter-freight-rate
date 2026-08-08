"""Distance and direction features, plus the city coordinate lookup.

City names are deliberately never encoded as categories. Eight cities - Allentown,
Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego - appear only in
validation.csv, so any name-based encoding meets unseen categories at prediction time.
Coordinates cover those cities without special handling.

The coordinates are not real geography (Los Angeles sits at 28.6N, 116.7W) but they are
internally consistent: fixed per city, and identical whether the city is an origin or a
destination. That is all these features need.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.8


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Build a city -> (lat, lon) lookup by stacking both ends of every load.

    Needed because december_chart_inputs.csv ships city names without coordinates.
    """
    parts = []
    for frame in frames:
        parts.append(
            frame[["pickup", "pickup_lat", "pickup_lon"]]
            .rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"})
        )
        parts.append(
            frame[["delivery", "delivery_lat", "delivery_lon"]]
            .rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})
        )
    stacked = pd.concat(parts, ignore_index=True).dropna()
    return stacked.groupby("city")[["lat", "lon"]].first()


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """Geography block of the feature matrix.

    Direction of travel is deliberately absent. Bearing and the lat/lon deltas were
    tried and measured, paired per load across the forward folds:

        adding bearing and both deltas    +0.27 MAE (+/-0.17)
        bearing on top of the deltas      -0.05 MAE (+/-0.09)

    Adding direction is slightly worse, consistently enough to sit outside the noise,
    and bearing contributes nothing either way. Coordinates and haversine are a
    different matter and stay: removing them costs +2.61 MAE (+/-0.29).
    """
    out = pd.DataFrame(index=frame.index)
    out["pickup_lat"] = frame["pickup_lat"]
    out["pickup_lon"] = frame["pickup_lon"]
    out["delivery_lat"] = frame["delivery_lat"]
    out["delivery_lon"] = frame["delivery_lon"]
    out["haversine"] = haversine(
        frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"]
    )
    return out
