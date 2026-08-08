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

import pandas as pd


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
    """Geography block: the four raw coordinates, and nothing derived from them.

    The coordinates answer WHERE, which `distance` cannot. Among Dry Van loads of 300 to
    500 miles, lane median rate per mile spans 2.105 to 2.535, a 20% spread at
    effectively the same length, and it replicates (split-half +0.737 across 193 lanes).
    Northern lanes sit at the cheap end, Gulf lanes at the dear end. Dropping the
    coordinates while keeping distance costs +2.52 MAE (+/-0.27), the same sign in all
    three folds.

    Two derived features were tried and rejected, both measured paired per load against
    this build:

        adding a haversine great-circle distance   -0.09 MAE (+/-0.13), sign varies
        adding bearing and the lat/lon deltas      +0.43 MAE (+/-0.17), sign varies

    The haversine is not wrong, it is redundant: it correlates 0.9995 with the supplied
    `distance` column, which is already a feature, so it restates a question the matrix
    has answered. Neither difference holds its sign across all three folds.

    Note that permutation importance disagrees, scoring every coordinate at 0.0%. That is
    the known failure of the measure under correlated features: permuting one coordinate
    leaves three others that still pin the lane down, so each looks worthless alone while
    the four together are worth 2.52. The ablations above are what the decision rests on.
    """
    out = pd.DataFrame(index=frame.index)
    out["pickup_lat"] = frame["pickup_lat"]
    out["pickup_lon"] = frame["pickup_lon"]
    out["delivery_lat"] = frame["delivery_lat"]
    out["delivery_lon"] = frame["delivery_lon"]
    return out
