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
