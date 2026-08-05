"""Command line entry point.

    python -m freight_rate.cli validate     forward chaining, and the random-fold contrast
    python -m freight_rate.cli predict      write validation_predictions.csv
    python -m freight_rate.cli december     December curves + december_predictions.csv
    python -m freight_rate.cli all          all three

One entry point rather than a set of run_*.py scripts: argument parsing stays in one
place and every pipeline stage remains importable and testable.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cleaning, loading, market
from . import december as december_module
from .config import CONFIG
from .evaluation import charts, metrics
from .features import DateEncoding
from .modeling import RateModel, forward_folds, random_folds

# (constant_market, encoding, colour, title) for the December comparison.
DECEMBER_SCENARIOS = [
    (True, DateEncoding.ORDINAL, "#B3261E",
     "1. Constant market input\n     + ordinal date encoding"),
    (False, DateEncoding.ORDINAL, "#B26A00",
     "2. Recovered daily market level\n     + ordinal date encoding"),
    (False, DateEncoding.CYCLICAL, "#064A56",
     "3. Recovered daily market level\n     + cyclical date encoding"),
]


@dataclass
class Pipeline:
    """Loaded, cleaned data plus the recovered market levels."""
    train: pd.DataFrame
    validation: pd.DataFrame
    market_levels: pd.Series

    @classmethod
    def build(cls) -> Pipeline:
        raw_train = loading.load_train()
        raw_validation = loading.load_validation()
        train, stats = cleaning.fit_apply(raw_train)
        validation = cleaning.apply(raw_validation, stats)
        return cls(train, validation, market.daily_levels(train, validation))

    def model(self, encoding: DateEncoding = DateEncoding.CYCLICAL) -> RateModel:
        return RateModel(encoding, self.market_levels)


def _configure_display() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda value: f"{value:,.2f}")


def _heading(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def run_validate(pipeline: Pipeline) -> None:
    _heading("FORWARD-CHAINING VALIDATION  (expanding window, 2-month test blocks)")
    forward = metrics.evaluate_splits(forward_folds(pipeline.train), pipeline.model)
    print(forward.to_string(index=False))

    _heading("RANDOM K-FOLD  (invalid here - shown to quantify the optimism)")
    shuffled = metrics.evaluate_splits(random_folds(pipeline.train), pipeline.model)
    print(shuffled.to_string(index=False))

    _heading("SUMMARY")
    summary = pd.concat([
        metrics.summarise(forward, "forward chaining (honest)"),
        metrics.summarise(shuffled, "random k-fold (optimistic)"),
    ], ignore_index=True)
    print(summary.to_string(index=False))
    honest, optimistic = summary.loc[0, "MAE"], summary.loc[1, "MAE"]
    print(f"\nRandom k-fold understates MAE by ${honest - optimistic:,.2f} "
          f"({(honest / optimistic - 1) * 100:.1f}% optimistic).")

    _heading("DATE ENCODING under forward chaining")
    ordinal = metrics.evaluate_splits(
        forward_folds(pipeline.train),
        lambda: pipeline.model(DateEncoding.ORDINAL),
    )
    comparison = pd.concat([
        metrics.summarise(ordinal, "ordinal (date_ordinal + month)"),
        metrics.summarise(forward, "cyclical (day-of-week + market level)"),
    ], ignore_index=True)
    print(comparison.to_string(index=False))
    penalty = comparison.loc[0, "MAE"] / comparison.loc[1, "MAE"] - 1
    print(f"\nOrdinal encoding costs {penalty * 100:.0f}% MAE here, because every forward fold")
    print("tests beyond its own training horizon - the same extrapolation December demands.")


def run_predict(pipeline: Pipeline) -> None:
    _heading("PREDICTING validation.csv")
    model = pipeline.model().fit(pipeline.train)
    predicted = model.predict(pipeline.validation)

    output = pd.DataFrame({
        "load_id": pipeline.validation["load_id"],
        "predicted_rate": np.round(predicted, 2),
    })
    destination = CONFIG.paths.predictions
    output.to_csv(destination, index=False)
    print(f"rows {len(output):,} | min ${output.predicted_rate.min():,.2f} "
          f"| max ${output.predicted_rate.max():,.2f}")
    print(f"wrote {destination}")


def run_december(pipeline: Pipeline) -> None:
    _heading("DECEMBER CURVE: 31 days, every input frozen except the date")
    raw = loading.load_december_inputs()

    curves, titles, colours = [], [], []
    for constant_market, encoding, colour, title in DECEMBER_SCENARIOS:
        frame = december_module.prepare(
            raw, pipeline.train, pipeline.validation,
            pipeline.market_levels, constant_market=constant_market,
        )
        model = RateModel(encoding, pipeline.market_levels).fit(pipeline.train)
        values = model.predict(frame)
        curves.append(values)
        titles.append(title)
        colours.append(colour)

        print(f"\n{title.splitlines()[0].strip()}")
        print(f"    market input : {'constant' if constant_market else 'recovered':<10}"
              f"   date encoding : {encoding.value}")
        print(f"    min ${values.min():,.2f}   max ${values.max():,.2f}   "
              f"range ${values.max() - values.min():,.2f}   "
              f"distinct {len(np.unique(values.round(2)))}/31")

    figure_path = charts.december_comparison(
        raw["date"], curves, titles, colours,
        CONFIG.paths.figures / "december_encoding_comparison.png",
    )

    filled = raw.copy()
    filled["predicted_rate"] = np.round(curves[-1], 2)
    filled["date"] = filled["date"].dt.strftime("%Y-%m-%d")
    filled.to_csv(CONFIG.paths.december_predictions, index=False)

    print(f"\nwrote {CONFIG.paths.december_predictions}")
    print(f"wrote {figure_path}")


COMMANDS = {"validate": run_validate, "predict": run_predict, "december": run_december}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="freight_rate",
        description="Freight rate prediction for the Spotter ML assessment.",
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    arguments = parser.parse_args(argv)

    _configure_display()
    pipeline = Pipeline.build()
    selected = COMMANDS.values() if arguments.command == "all" else [COMMANDS[arguments.command]]
    for command in selected:
        command(pipeline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
