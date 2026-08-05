"""Recovering the daily market level from per-load market_index values.

market_index looks per-load - on a day with ~157 loads there are ~155 distinct values -
but 97.8% of its variance is explained by the date alone. The structure is a daily
market level plus per-load noise of std 0.025, and that noise correlates with nothing:
+0.002 with the target, and 0.00001 or less by equipment type.

Averaging a date's loads cancels the noise and recovers the level to about +/-0.002,
against a level that ranges 0.767 to 1.402. Two random halves of the same day's loads
agree at correlation 0.9997.

This matters because december_chart_inputs.csv ships without market_index, and
validation.csv covers all 31 December days.

Leakage note: this reads the market_index feature column only, never posted_rate.
Computing it across train and validation together uses information the assessment
supplies for the prediction window, so it is legitimate - but it is a judgment call
and is called out in the report.
"""
from __future__ import annotations

import pandas as pd


def daily_levels(*frames: pd.DataFrame) -> pd.Series:
    """Return a date -> market level Series covering every date in the inputs."""
    if not frames:
        raise ValueError("at least one frame is required")
    stacked = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    levels = stacked.dropna().groupby("date")["market_index"].mean()
    levels.name = "daily_market_level"
    return levels
