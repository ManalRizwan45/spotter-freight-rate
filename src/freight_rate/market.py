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

Why it takes several frames, and why that is not leakage: it reads the market_index
feature column and never posted_rate, and each date's level is averaged over that date's
own loads. Train covers January to October and validation November and December, so the
two never contribute to the same date. No training row's feature draws on the prediction
window, and no validation row's draws on training. market_index is supplied for the
validation rows, so using it is not using information that would be absent at prediction
time.
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
