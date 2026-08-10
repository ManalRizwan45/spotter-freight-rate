"""Regenerate the figures REPORT.md embeds.

    python docs/make_figures.py

Three of the five already exist before this runs: `cli december` writes the encoding
comparison, `cli evaluate` writes the residual diagnostics, and `score.py` writes
candidate_december.png, which the assessment requires the report to contain. This script
adds the two the report needs that no command has reason to produce, and copies the
scorer's chart out of the gitignored scorer_results/ so the report can be rebuilt from a
clean checkout.

It composes the same modules the pipeline does. Nothing here recomputes a number that
`src/freight_rate/` already defines.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from freight_rate import cleaning, loading, market
from freight_rate.config import CONFIG, PROJECT_ROOT
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

    # The assessment asks for candidate_december.png in the report, and only score.py
    # can produce it. Copied rather than redrawn: a lookalike of the scorer's chart is
    # not the scorer's chart, and the difference is the whole point of including it.
    scorer_chart = PROJECT_ROOT / "scorer_results" / "candidate_december.png"
    if not scorer_chart.is_file():
        raise FileNotFoundError(
            f"{scorer_chart} not found. Run the pipeline and then the scorer:\n"
            "    python -m freight_rate.cli all\n"
            "    python score.py --predictions validation_predictions.csv "
            "--december-predictions data/december_chart_inputs.csv"
        )
    copied = CONFIG.paths.figures / "candidate_december.png"
    shutil.copyfile(scorer_chart, copied)
    print(f"wrote {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
