"""The evaluation panel: what the model is like to price with, not what selected it.

`metrics.py` has one job - ranking configurations under forward chaining - and four
metrics is the right width for that. A selection metric wants to be stable, comparable
and few. This module answers the other question: is the model good enough to quote with,
and where does it fail? Four things the selection table cannot show.

  Direction. MAE is symmetric, so a model that quotes 3% high on every load scores the
  same as one that scatters either side. On a rate desk those are not the same mistake,
  so `bias` is reported signed: positive means the model quotes above what the load paid.

  Dollar weight. MAPE gives a $500 load and a $9,000 load an equal vote. WAPE weights by
  dollars - total error over total dollars - which is how the error is actually paid.
  With this target the two disagree by construction: `posted_rate` scales with distance,
  so percentage error and dollar error do not rank loads the same way.

  The tail. Median APE describes the load in the middle. P90 APE names what a bad one
  looks like, and the gap between them is the whole story on a distribution this skewed.

  A reference. R2 against `posted_rate` reads well here for a reason that has nothing to
  do with the model: the target scales with trip length and `distance` alone explains
  most of its variance, so a high R2 is mostly a restatement of that. The honest
  reference is `distance x quote_signal`, which is what a broker already has before any
  model runs. `compare()` puts it in the table and scores the model against it.

Predictions come from `out_of_fold`, which pools the forward-chaining test blocks. Every
row is scored by a model trained only on earlier months, so the panel measures the same
extrapolation the real task demands. The alternative - fitting on all of train and
scoring the rows it was fitted on - would report a forest's memory rather than its
accuracy.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..modeling import target
from ..modeling.estimator import RateModel
from ..modeling.splits import Split

# Share of loads landing within each band of the actual rate. 10% is the headline: it is
# roughly where a quote stops being negotiable and starts being wrong.
ACCURACY_BANDS = (5.0, 10.0, 25.0)

# Length classes a dispatcher would recognise, rather than equal-count quantiles: the
# question is whether the model holds up on short hauls, not whether each cell is equal.
DISTANCE_EDGES = (0.0, 250.0, 500.0, 1000.0, np.inf)
DISTANCE_LABELS = ("under 250 mi", "250-500 mi", "500-1000 mi", "1000+ mi")

# The panel is too wide to repeat for every group, so segment tables carry the six that
# answer "is this segment worse, and in which direction".
SEGMENT_COLUMNS = ["MAE $", "bias $", "WAPE %", "median APE %", "P90 APE %", "within 10%"]


def panel(actual, predicted) -> dict[str, float]:
    """Every metric for one set of predictions, keyed by its display name.

    `within N%` is the share of loads whose prediction lands within N percent of the
    actual rate, in percent - a hit rate, not an error.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")
    if actual.size == 0:
        raise ValueError("no rows to score")
    # Every percentage metric below divides by the actual rate. A zero would return inf
    # and a negative would silently flip the sign of an error, so neither is allowed
    # through: posted_rate is dollars paid and is positive on every supplied row.
    if (actual <= 0).any():
        raise ValueError("actual rates must be positive to score percentage error")

    error = predicted - actual
    absolute = np.abs(error)
    absolute_percentage = absolute / actual * 100
    # R2 divides by the spread of the actuals, which is zero for a single row or a group
    # whose loads all paid the same. It is undefined there rather than infinite, so it
    # reads NaN and every other metric in the panel still returns.
    spread = float(((actual - actual.mean()) ** 2).sum())

    scores = {
        "MAE $": float(absolute.mean()),
        "RMSE $": float(np.sqrt((error ** 2).mean())),
        "bias $": float(error.mean()),
        "MAPE %": float(absolute_percentage.mean()),
        "median APE %": float(np.median(absolute_percentage)),
        "P90 APE %": float(np.percentile(absolute_percentage, 90)),
        "WAPE %": float(absolute.sum() / actual.sum() * 100),
        "bias %": float(error.sum() / actual.sum() * 100),
        "R2": float(1 - (error ** 2).sum() / spread) if spread > 0 else float("nan"),
    }
    for band in ACCURACY_BANDS:
        scores[f"within {band:g}%"] = float((absolute_percentage <= band).mean() * 100)
    return scores


def baselines(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """The two references the panel should be read against.

    `mean rate` is deliberately generous: it uses the mean of the very rows being scored,
    which no real predictor would know. A baseline that cannot be beaten decisively even
    when handed the answer's average is the point of including it.

    `quote only` is `distance x quote_signal` - the number the assessment already
    supplies, and the anchor the model is trained to correct rather than replace.
    """
    return {
        "mean rate": np.full(len(frame), float(frame["posted_rate"].mean())),
        "quote only": target.baseline(frame).to_numpy(),
    }


def compare(actual, candidates: dict[str, np.ndarray],
            reference: str | None = None) -> pd.DataFrame:
    """One column per candidate, metrics down the rows.

    With `reference` named, a final row reports the MAE reduction against it, which is
    the number that says what the model added over what was already on hand.
    """
    table = pd.DataFrame({name: panel(actual, values) for name, values in candidates.items()})
    if reference is not None:
        if reference not in table.columns:
            raise KeyError(f"reference {reference!r} is not among {list(table.columns)}")
        anchor = table.loc["MAE $", reference]
        table.loc[f"MAE cut vs {reference} %"] = (1 - table.loc["MAE $"] / anchor) * 100
    return table


def out_of_fold(splits: Iterable[Split], model_factory) -> tuple[pd.DataFrame, np.ndarray]:
    """Pool the test blocks of every split, each scored by a model that never saw it.

    Returns the pooled rows and their predictions, positionally aligned. `model_factory`
    returns a RateModel, matching `metrics.evaluate_splits`.
    """
    frames, predictions = [], []
    for split in splits:
        model: RateModel = model_factory()
        predictions.append(model.fit_predict(split.train, split.test))
        frames.append(split.test.assign(fold=split.label))

    if not frames:
        raise ValueError("no splits to evaluate")
    pooled = pd.concat(frames, ignore_index=True)
    # Forward-chaining test blocks are disjoint in time, so pooling them cannot repeat a
    # load. If it ever does, some rows would be double-counted in every metric below.
    if pooled["load_id"].duplicated().any():
        raise ValueError("splits overlap: a load appears in more than one test block")
    return pooled, np.concatenate(predictions)


def by_segment(frame: pd.DataFrame, predicted: np.ndarray, column: str) -> pd.DataFrame:
    """`SEGMENT_COLUMNS` per group of `column`, plus the group size.

    `predicted` is positional, so it is indexed with the group's positions rather than
    with its labels: `frame` arrives pooled from several folds and its index is not
    assumed to mean anything.
    """
    actual = frame["posted_rate"].to_numpy(dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {predicted.shape}")

    rows = {}
    for name, positions in frame.groupby(column, observed=True).indices.items():
        scores = panel(actual[positions], predicted[positions])
        rows[name] = {"n": len(positions), **{key: scores[key] for key in SEGMENT_COLUMNS}}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(column)


def distance_band(distance: pd.Series) -> pd.Series:
    return pd.cut(distance, bins=list(DISTANCE_EDGES), labels=list(DISTANCE_LABELS), right=False)


def segments(frame: pd.DataFrame, predicted: np.ndarray) -> dict[str, pd.DataFrame]:
    """The three cuts worth reading: what is being hauled, how far, and when.

    Equipment and distance test whether one kind of load is carrying the error. Month
    tests something else - each fold predicts a period beyond its own training data, so
    the month column shows whether accuracy decays with distance from the training
    horizon, which is what November and December will ask of it.
    """
    working = frame.copy()
    working["distance band"] = distance_band(working["distance"])
    working["month"] = working["date"].dt.strftime("%Y-%m")
    return {
        "equipment": by_segment(working, predicted, "equipment"),
        "distance": by_segment(working, predicted, "distance band"),
        "month": by_segment(working, predicted, "month"),
    }
