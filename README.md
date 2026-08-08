# Freight Rate Prediction: Spotter ML Assessment

Predicts `posted_rate` for 12,000 held-out freight loads, and produces the fixed
December chart.

## Data

The Spotter-supplied datasets and the assessment PDF are **not committed**: they are
gitignored so this repository does not redistribute assessment materials. To run the
pipeline, place the supplied files as:

```
data/train_test.csv
data/validation.csv
data/december_chart_inputs.csv
data/validation_predictions_template.csv
```

`score.py` and the instructions at [`docs/assessment_readme.md`](docs/assessment_readme.md)
are included so the documented workflow is reproducible.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Run

```bash
python -m freight_rate.cli all
```

| Command | Does |
|---|---|
| `validate` | Forward-chaining validation, plus the random k-fold contrast |
| `predict` | Writes `validation_predictions.csv` |
| `december` | December curves + `december_predictions.csv` |
| `all` | All three |

Then the official scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

> **Note on the scorer invocation.** The supplied instructions point
> `--december-predictions` at `data/december_chart_inputs.csv`, i.e. that file filled in
> place. This pipeline writes a filled copy to `december_predictions.csv` at the repo
> root instead, leaving everything under `data/` exactly as supplied. Both files are
> byte-identical in structure and the scorer accepts either.

Tests and lint:

```bash
python -m pytest && python -m ruff check src tests
```

## Exploratory analysis

[`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb) is committed with all
outputs and figures embedded, so it reads without executing. It reads from
`src/freight_rate/` rather than reimplementing anything, so it cannot drift from the
pipeline. To re-run it:

```bash
python -m pip install -e ".[notebooks]"
```

## Approach

**The target is a correction, not a rate.** `distance × quote_signal` already lands
within 2% of the actual rate on half the training rows, a 2.0% median error with no
model at all. The model predicts `log(posted_rate / (distance × quote_signal))`, so it
only learns the deviation. The log keeps the ~0.7% of rows above $4/mile from
dominating a squared-error loss.

**Validation is forward-chained, not shuffled.** Daily median rate-per-mile carries
lag-1 autocorrelation of +0.844, so a random split puts one day's loads in train and the
adjacent day's in test while both share a market level. Measured cost of getting this
wrong:

| Strategy | MAE | RMSE | MAPE | median APE |
|---|---|---|---|---|
| Forward chaining (honest) | $146.27 | $640.28 | 6.48% | 3.13% |
| Random k-fold (optimistic) | $109.40 | $594.02 | 4.96% | 1.98% |

Random k-fold understates MAE by **33.7%**.

**Dates are encoded so they extrapolate.** Training ends 2025-10-31; the chart asks for
December. A `date_ordinal` or `month` feature falls beyond every learned split, so a
tree clamps it and the chart flatlines. Day-of-week, day-of-month and the looked-up daily
market level both exist in December.

| Encoding | MAE under forward chaining |
|---|---|
| Ordinal (`date_ordinal` + `month`) | $204.12 |
| Recurring (day-of-week + market level) | **$146.27** |

**Cities are never encoded by name.** Eight of them (Allentown, Charlotte, Chicago,
Jackson, Knoxville, Laredo, Norfolk, San Diego) appear only in `validation.csv`.
Geography comes from coordinates and haversine, which cover unseen cities.

## The December chart

`december_chart_inputs.csv` ships without `market_index`, `quote_signal` or coordinates,
so each is reconstructed. See [`src/freight_rate/december.py`](src/freight_rate/december.py).
The key one: `market_index` looks per-load, but **97.8% of its variance is explained by
date alone**. Averaging a date's ~157 loads cancels the per-load noise and recovers the
daily level to ±0.002. `validation.csv` covers all 31 December days.

Holding everything except the date frozen:

| Market input | Date encoding | Range across the month | Distinct values in 31 days |
|---|---|---|---|
| Global mean | Ordinal | **$0.00** | **1** |
| Recovered daily level | Ordinal | $22.99 | 17 |
| Recovered daily level | Recurring | $31.18 | **31** |

The top row is the failure the chart is built to expose: one value repeated across the
whole month. The middle row moves, but resolves only 17 of 31 days, because an ordinal
date encoding cannot separate days that fall beyond the training horizon. The bottom
row gives all 31 days a distinct value. Recovering
the daily market level is necessary but not sufficient; the date encoding has to be
able to use it.

Correlation against the true December market level is deliberately not quoted here.
For the middle row it has flipped sign across feature configurations, so it is not a
stable quantity. The distinct-value counts are, and they order the three scenarios
the same way every time. Figure: `reports/figures/december_encoding_comparison.png`.

## Data-quality issues addressed

| Issue | Counts (train / validation) | Handling |
|---|---|---|
| Negative `weight` | 292 / 145 | Sign errors: magnitudes match the positive rows at every quantile, so `abs()` |
| Missing `weight` | 300 / 165 | Left as NaN; the model splits on missingness natively |
| Missing `market_index` | 374 / 249 | Filled from the same date's mean |
| Rate outliers | ~0.7% above $4/mile | Kept; the log-ratio target contains them |
| `market_index` shift | train mean 1.08 → validation 0.93 | Noted; the model does not lean on its upper range |

## Layout

```
src/freight_rate/
  config.py       frozen dataclasses: paths, folds, model params, seed
  loading.py      CSV reads with schema enforcement
  cleaning.py     sign errors and gaps, as pure transforms with nothing learned
  market.py       daily market level recovery
  features/
    geography.py  haversine and the city coordinate lookup
    temporal.py   the two date encodings, where the chart is won or lost
    assembly.py   composition into the model matrix
  modeling/
    target.py     log-ratio transform and its inverse
    estimator.py  fit / predict
    splits.py     forward chaining, and the random-fold contrast
  evaluation/     metrics and figures
  december.py     reconstructing the chart file's missing columns
  cli.py          entry point
tests/            44 tests, including a scorer-contract guard
```

`score.py`, `data/` and `requirements.txt` are supplied by the assessment and unmodified.

## Notes on choices

- **Config in Python, not YAML.** At ~15 parameters a config file costs a parser and a
  dependency while giving up type checking; frozen dataclasses keep it typed and
  navigable.
- **`HistGradientBoostingRegressor`, chosen because nothing dominates it on both
  deliverables.** Ten models were benchmarked on in-range MAE under forward chaining.
  Chart quality was judged separately, by rehearsing December on five held-out months:
  train on everything prior, build the fixed-lane 31-day chart the same way, and
  correlate its shape against what comparable loads actually cost that month.

  | Model | MAE | Chart shape vs real prices |
  |---|---|---|
  | ExtraTrees | **129.57** | +0.04 |
  | RandomForest | 135.50 | +0.30 |
  | **HistGradientBoosting** | 146.27 | +0.24 |
  | LightGBM | 146.63 | +0.26 |
  | ElasticNet | 175.84 | **+0.44** |

  ElasticNet tracks the chart best and pays 20% on MAE. ExtraTrees wins MAE and has
  effectively no chart signal. The middle three are not separable on chart shape: across
  the five months each model's score swings by 0.5 or more, so the choice among them
  rests on MAE and on needing no extra dependency.

  Chart shape is scored against real prices rather than against the recovered market
  level. Those two rank models differently, and only the first is ground truth. Months
  are weighted by their own split-half reliability, since a month whose daily medians do
  not replicate (October scores 0.045) offers nothing to match. Also tested: XGBoost,
  classic GradientBoosting, KNeighbors, Ridge, a mean predictor, and the quote alone.
- **Leakage boundary.** The daily market level reads the `market_index` feature column
  across both train and validation, never `posted_rate`, because the assessment supplies
  that column for the prediction window. That is a judgment call, so it is stated rather
  than assumed.
- **`tests/test_scorer_contract.py`** asserts the output satisfies every rule in
  `score.py`. A format rejection scores zero regardless of model quality.
