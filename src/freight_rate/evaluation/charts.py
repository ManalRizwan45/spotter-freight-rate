"""Figures for the report.

Matplotlib's Agg backend is selected at import: these run headless in CI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIXED_INPUTS_CAPTION = (
    "Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only the date changes"
)

TEAL, RUST, SLATE = "#064A56", "#B3261E", "#607478"


def december_comparison(dates: pd.Series, curves: list[np.ndarray],
                        titles: list[str], colours: list[str], path: Path) -> Path:
    """Side-by-side December curves under different date/market handling."""
    if not (len(curves) == len(titles) == len(colours)):
        raise ValueError("curves, titles and colours must be the same length")

    figure, axes = plt.subplots(1, len(curves), figsize=(5.7 * len(curves), 5.0),
                                dpi=170, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, values, title, colour in zip(axes, curves, titles, colours, strict=True):
        axis.plot(dates, values, color=colour, linewidth=2.6, marker="o", markersize=3.4)
        axis.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=10)
        axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9DAFB3")
        axis.tick_params(axis="x", rotation=35, labelsize=8)
        axis.annotate(
            f"moves ${values.max() - values.min():,.2f} across the month\n"
            f"{len(np.unique(values.round(2)))} distinct values in 31 days",
            xy=(0.04, 0.95), xycoords="axes fraction", va="top",
            fontsize=9.5, color=colour, fontweight="bold",
        )
    axes[0].set_ylabel("Predicted rate ($)")
    figure.suptitle("Identical model, identical training data, identical non-date features",
                    x=0.008, ha="left", fontsize=14.5, fontweight="bold")
    figure.text(0.008, -0.015, FIXED_INPUTS_CAPTION, fontsize=9.5, color="#455A60")
    figure.tight_layout(rect=(0, 0.02, 1, 0.93))

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def error_diagnostics(actual, predicted, path: Path, sample: int = 6000) -> Path:
    """Three residual views of one set of predictions.

    A metric table says how large the error is. These say what shape it has, which is
    what decides whether the number can be trusted across the range:

      fit          predicted against actual. Scatter that hugs the diagonal at every
                   rate level, rather than fanning out or bending, means one MAE
                   describes the whole book.
      distribution signed percentage error. Centred is unbiased; the width of the bulk
                   against the reach of the tails is what MAE and RMSE disagree about.
      calibration  bias by decile of PREDICTED rate. Binning on the prediction rather
                   than on the truth reverses the usual shrinkage artifact: a load the
                   model calls cheap is on average dearer than the call, so the first
                   decile reads low and the last reads high. The question is how much,
                   and whether it stays flat in between.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    signed_percentage = (predicted - actual) / actual * 100

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), dpi=150)

    shown = np.random.default_rng(0).choice(
        actual.size, size=min(sample, actual.size), replace=False
    )
    limit = float(np.percentile(actual, 99.5))
    axes[0].scatter(actual[shown], predicted[shown], s=5, alpha=0.25, color=TEAL)
    axes[0].plot([0, limit], [0, limit], color=RUST, linestyle="--", linewidth=1.4)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("actual posted_rate ($)")
    axes[0].set_ylabel("predicted rate ($)")
    axes[0].set_title("Fit across the range", loc="left", fontweight="bold")

    axes[1].hist(np.clip(signed_percentage, -40, 40), bins=90, color=TEAL)
    axes[1].axvline(0, color=RUST, linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("signed % error, clipped at +/-40 for display")
    axes[1].set_title(
        f"Error distribution: median {np.median(signed_percentage):+.2f}%",
        loc="left", fontweight="bold",
    )

    deciles = pd.qcut(predicted, 10, labels=False, duplicates="drop")
    bias = pd.Series(signed_percentage).groupby(deciles).mean()
    axes[2].bar(bias.index + 1, bias.to_numpy(),
                color=[RUST if value < 0 else TEAL for value in bias])
    axes[2].axhline(0, color=SLATE, linewidth=0.9)
    axes[2].set_xlabel("decile of predicted rate (1 = cheapest)")
    axes[2].set_ylabel("mean signed % error")
    axes[2].set_title("Calibration by rate level", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#9DAFB3")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path
