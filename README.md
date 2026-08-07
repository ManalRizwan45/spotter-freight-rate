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
| Forward chaining (honest) | $147.59 | $641.24 | 6.54% | 3.19% |
| Random k-fold (optimistic) | $110.25 | $594.16 | 5.00% | 2.01% |

Random k-fold understates MAE by **33.9%**.

**Dates are encoded so they extrapolate.** Training ends 2025-10-31; the chart asks for
December. A `date_ordinal` or `month` feature falls beyond every learned split, so a
tree clamps it and the chart flatlines. Day-of-week (cyclical) and the looked-up daily
market level both exist in December.

| Encoding | MAE under forward chaining |
|---|---|
| Ordinal (`date_ordinal` + `month`) | $199.51 |
| Cyclical (day-of-week + market level) | **$147.59** |

**Cities are never encoded by name.** Eight of them (Allentown, Charlotte, Chicago,
Jackson, Knoxville, Laredo, Norfolk, San Diego) appear only in `validation.csv`.
Geography comes from coordinates, haversine and bearing, which cover unseen cities.

## The December chart

`december_chart_inputs.csv` ships without `market_index`, `quote_signal` or coordinates,
so each is reconstructed. See [`src/freight_rate/december.py`](src/freight_rate/december.py).
The key one: `market_index` looks per-load, but **97.8% of its variance is explained by
date alone**. Averaging a date's ~157 loads cancels the per-load noise and recovers the
daily level to ±0.002. `validation.csv` covers all 31 December days.

Holding everything except the date frozen:

| Market input | Date encoding | Range across the month | Distinct values | Correlation with actual market level |
|---|---|---|---|---|
| Global mean | Ordinal | **$0.00** | 1 / 31 | n/a |
| Recovered daily level | Ordinal | $9.03 | 12 / 31 | **−0.345** |
| Recovered daily level | Cyclical | $26.00 | 30 / 31 | **+0.687** |

The middle row is the trap: it moves, so it passes a glance, but it is *anti-correlated*
with the December market. Figure: `reports/figures/december_encoding_comparison.png`.

## Data-quality issues addressed

| Issue | Counts (train / validation) | Handling |
|---|---|---|
| Negative `weight` | 292 / 145 | Sign errors: magnitudes are plausible, so `abs()` and flag |
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
    geography.py  haversine, bearing, city coordinate lookup
    temporal.py   the two date encodings, where the chart is won or lost
    assembly.py   composition into the model matrix
  modeling/
    target.py     log-ratio transform and its inverse
    estimator.py  fit / predict
    splits.py     forward chaining, and the random-fold contrast
  evaluation/     metrics and figures
  december.py     reconstructing the chart file's missing columns
  cli.py          entry point
tests/            52 tests, including a scorer-contract guard
```

`score.py`, `data/` and `requirements.txt` are supplied by the assessment and unmodified.

## Notes on choices

- **Config in Python, not YAML.** At ~15 parameters a config file costs a parser and a
  dependency while giving up type checking; frozen dataclasses keep it typed and
  navigable.
- **`HistGradientBoostingRegressor`.** Handles NaN natively (missing `weight` carries
  signal), trains 48,000 rows in seconds, needs no scaling or one-hot expansion.
- **Leakage boundary.** Cleaning learns nothing from the data: every repair is a pure
  transform, so there is no fitted quantity that could carry training information into
  the prediction window. The daily market level does read the `market_index` *feature*
  column (never `posted_rate`) across both train and validation, because the
  assessment supplies it for the prediction window. That judgment call is called out
  here and in the report.
- **`tests/test_scorer_contract.py`** asserts the output satisfies every rule in
  `score.py`. A format rejection scores zero regardless of model quality.
