"""Regenerate the figures REPORT.md embeds.

    python docs/make_figures.py

Two of the four already come from the pipeline: `cli december` writes the encoding
comparison and `cli evaluate` writes the residual diagnostics. This script adds the two
the report needs and the CLI has no reason to produce, so every figure in the report has
a command behind it rather than a manual export.

It composes the same modules the pipeline does. Nothing here recomputes a number that
`src/freight_rate/` already defines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from freight_rate import cleaning, loading, market
from freight_rate.config import CONFIG
from freight_rate.evaluation import charts, report
from freight_rate.features import DateEncoding
from freight_rate.modeling import RateModel, forward_folds, target


def main() -> int:
    train = cleaning.clean(loading.load_train())
    validation = cleaning.clean(loading.load_validation())
    market_levels = market.daily_levels(train, validation)

    quote = charts.quote_baseline(
        target.baseline(train), train["posted_rate"],
        CONFIG.paths.figures / "quote_baseline.png",
    )
    print(f"wrote {quote}")

    pooled, predicted = report.out_of_fold(
        forward_folds(train), lambda: RateModel(DateEncoding.RECURRING, market_levels)
    )
    pooled_month = pooled["date"].dt.strftime("%Y-%m")
    quote_only = report.baselines(pooled)["quote only"]
    by_month = pd.DataFrame({
        "model MAE": np.abs(predicted - pooled["posted_rate"]).groupby(pooled_month).mean(),
        "quote-only MAE": np.abs(quote_only - pooled["posted_rate"]).groupby(pooled_month).mean(),
    })

    ratio = train["posted_rate"] / target.baseline(train)
    daily_share = (ratio > 1.2).groupby(train["date"]).mean() * 100

    regime = charts.regime_split(
        by_month, daily_share, CONFIG.paths.figures / "regime_split.png",
    )
    print(f"wrote {regime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
