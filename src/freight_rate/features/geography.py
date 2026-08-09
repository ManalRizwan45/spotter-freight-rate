"""Where a load runs, plus the city coordinate lookup.

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
    """Geography block: the four raw coordinates, plus a haversine distance.

    The coordinates answer WHERE, which `distance` cannot. Among Dry Van loads of 300 to
    500 miles, lane median rate per mile spans 2.105 to 2.535, a 20% spread at
    effectively the same length, and it replicates (split-half +0.737 across 193 lanes).
    Northern lanes sit at the cheap end, Gulf lanes at the dear end. Dropping the
    coordinates while keeping distance costs +3.86 MAE (+/-0.34), the same sign in all
    three folds.

    The haversine correlates 0.9995 with the supplied `distance` column, so it is very
    nearly a duplicate. Whether that duplication pays depends on a hyperparameter, which
    is worth stating because the two decisions were made in the wrong order once already:

        with every feature offered at each split   worth -0.09 MAE (+/-0.13), nothing
        with max_features restricted to 8 of 15    worth +2.08 MAE (+/-0.27) to drop

    Under feature subsampling only some columns are offered at each node, so a second
    distance-like column raises the odds that one is available. Redundancy stops being
    waste and becomes insurance. The column was removed while the forest still saw
    everything, and restored once it did not.

    Direction of travel is still absent. Bearing and the lat/lon deltas cost +0.51 MAE
    (+/-0.23) with the sign varying by fold.

    Note that permutation importance disagrees, scoring every coordinate at 0.0%. That is
    the known failure of the measure under correlated features: permuting one coordinate
    leaves three others that still pin the lane down, so each looks worthless alone while
    the four together are worth 3.86. The ablations above are what the decision rests on.
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
