"""Shared fixtures.

Tests run on synthetic frames rather than the supplied CSVs so they stay fast and pass
in CI whether or not data/ is present. The one test that needs real output skips
explicitly when it is missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

CITIES = {
    "Lexington": (38.0, -84.5),
    "Fort Wayne": (41.1, -85.1),
    "Dallas": (31.8, -94.4),
    "Boston": (42.4, -71.1),
}


@pytest.fixture
def sample_loads() -> pd.DataFrame:
    """A small labelled frame spanning two months, with the real data's quirks."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", "2025-02-28", freq="D").repeat(4)
    size = len(dates)
    pickups = rng.choice(list(CITIES), size)
    deliveries = [rng.choice([c for c in CITIES if c != p]) for p in pickups]

    frame = pd.DataFrame({
        "load_id": [f"TR-{i:06d}" for i in range(1, size + 1)],
        "pickup": pickups,
        "delivery": deliveries,
        "pickup_lat": [CITIES[c][0] for c in pickups],
        "pickup_lon": [CITIES[c][1] for c in pickups],
        "delivery_lat": [CITIES[c][0] for c in deliveries],
        "delivery_lon": [CITIES[c][1] for c in deliveries],
        "distance": rng.uniform(200, 2000, size).round(1),
        "equipment": rng.choice(["Dry Van", "Reefer", "Flatbed"], size),
        "weight": rng.uniform(20000, 40000, size).round(),
        "date": dates,
        "market_index": rng.normal(1.0, 0.02, size).round(5),
        "quote_signal": rng.normal(2.1, 0.2, size).round(5),
    })
    frame["posted_rate"] = (frame.distance * frame.quote_signal).round(2)

    # Inject the two data-quality issues the real files carry.
    frame.loc[frame.index[:3], "weight"] *= -1
    frame.loc[frame.index[5:8], "weight"] = np.nan
    frame.loc[frame.index[10:13], "market_index"] = np.nan
    return frame


@pytest.fixture
def market_levels(sample_loads: pd.DataFrame) -> pd.Series:
    from freight_rate import market
    return market.daily_levels(sample_loads)


@pytest.fixture
def december_rows() -> pd.DataFrame:
    """Mirrors december_chart_inputs.csv: seven columns, fixed lane, only date varies."""
    dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    return pd.DataFrame({
        "pickup": "Lexington",
        "delivery": "Fort Wayne",
        "distance": 360,
        "equipment": "Dry Van",
        "weight": 32000,
        "date": dates,
        "predicted_rate": np.nan,
    })
