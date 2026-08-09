"""Composition of the feature blocks into a model matrix.

Only the date block varies between configurations. Everything else is shared, so a
comparison between two encodings isolates the date handling and nothing else.
"""
from __future__ import annotations

import pandas as pd

from . import geography, temporal
from .temporal import DateEncoding

EQUIPMENT_TYPES = ("Dry Van", "Reefer", "Flatbed")
EQUIPMENT_COLUMNS = {kind: f"is_{kind.replace(' ', '_').lower()}" for kind in EQUIPMENT_TYPES}


def build_load(frame: pd.DataFrame) -> pd.DataFrame:
    """Load-level block: size, price signal, cargo.

    No log of distance: trees split on thresholds, so any monotone transform of a
    feature yields identical splits. log(distance) can add nothing distance does
    not already carry.

    No missing-weight flag either. Imputing erases the fact that a value was missing, so
    a flag could carry it, but there is nothing to carry: the 300 missing-weight rows
    price like the rest (median rate/mile 2.1176 against 2.1456, Mann-Whitney p = 0.234)
    and match on equipment mix to within 2 points.

    Equipment is one-hot rather than an integer code. A tree can isolate a category from
    an integer, but only as a contiguous run, so the coding's ORDER decides which single
    splits exist, and an arbitrary order is not free. Codes of Dry Van=0, Reefer=1,
    Flatbed=2 cost +1.74 MAE (+/-0.26) against these indicators, the same sign in all
    three folds. Reordering those same codes by median rate per mile recovers most but
    not all of it (+0.80, and the sign varies by fold), so the ordering was most of the
    problem and the representation is the rest. Indicators need no ordering to justify
    at all, which is why they are kept.
    """
    out = pd.DataFrame(index=frame.index)
    out["distance"] = frame["distance"]
    # quote_signal is a per-load dollars-per-mile quote and the strongest single
    # feature by a wide margin: 81% of permutation importance, against 8% for distance.
    out["quote_signal"] = frame["quote_signal"]
    out["market_index"] = frame["market_index"]
    for kind, column in EQUIPMENT_COLUMNS.items():
        out[column] = (frame["equipment"] == kind).astype(float)
    out["weight"] = frame["weight"]
    return out


def build(frame: pd.DataFrame, encoding: DateEncoding,
          market_levels: pd.Series | None = None) -> pd.DataFrame:
    """Full feature matrix for `frame` under the given date encoding."""
    # One-hot would silently encode an unseen type as all-zeros, indistinguishable from
    # a type the model knows. Raising keeps that from reaching a prediction.
    unknown = set(frame["equipment"].dropna()) - set(EQUIPMENT_TYPES)
    if unknown:
        raise ValueError(f"unmapped equipment types: {sorted(unknown)}")

    return pd.concat(
        [
            build_load(frame),
            geography.build(frame),
            temporal.build(frame["date"], encoding, market_levels),
        ],
        axis=1,
    )
