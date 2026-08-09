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

Runtime dependencies are pinned to exact versions, because every figure quoted below is
a measurement and scikit-learn's tree construction can shift across minor releases.
Produced on Python 3.13.14 with pandas 2.3.3, numpy 2.5.1, scikit-learn 1.9.0 and
matplotlib 3.11.1.

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

> **Scorer invocation.** The supplied instructions fill
> `data/december_chart_inputs.csv` in place. This pipeline writes the filled copy to
> `december_predictions.csv` instead, leaving `data/` exactly as supplied. Same
> structure, and the scorer accepts either.

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
| Forward chaining (honest) | $128.93 | $636.27 | 5.70% | 2.39% |
| Random k-fold (optimistic) | $94.13 | $591.54 | 4.28% | 1.37% |

Random k-fold understates MAE by **37.0%**. It is reported here as a measured
counterexample and is never used to select or score anything.

**Dates are encoded so they extrapolate.** Training ends 2025-10-31; the chart asks for
December. A `date_ordinal` or `month` feature falls beyond every learned split, so a tree
clamps it and the chart flatlines. Day-of-week, day-of-month and the looked-up daily
market level all recur, so December lands inside the learned range.

| Encoding | MAE under forward chaining |
|---|---|
| Ordinal (`date_ordinal` + `month`) | $173.18 |
| Recurring (day-of-week + market level) | **$128.93** |

**Cities are never encoded by name.** Eight of them (Allentown, Charlotte, Chicago,
Jackson, Knoxville, Laredo, Norfolk, San Diego) appear only in `validation.csv`.
Geography comes from the four coordinates and a haversine distance, which cover unseen cities.

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
| Recovered daily level | Ordinal | $4.84 | 22 |
| Recovered daily level | Recurring | $9.68 | **30** |

Recovering the daily market level is necessary but not sufficient. The top row is the
failure the chart exposes: one value all month, because a constant market input leaves
an ordinal date encoding nothing that varies. The other two rows both move.

The dollar ranges are small because the chart freezes `quote_signal`, which carries 81%
of permutation importance. What moves here is the date response with the dominant
feature pinned, which is what the chart isolates and is not a measure of accuracy.

**Read the counts as an ordering, not as magnitudes.** They have ordered the three
scenarios the same way under every model this repo has run, but the values move a lot:
the middle row has read 10, 11, 18 and 22 distinct days as the estimator changed, and
the bottom row 30 or 31. The load-bearing evidence for the encoding is the MAE above,
$173.18 against $128.93, not this table. Correlation against the December market level
is not quoted at all, because for the middle row it flips sign across configurations.
Figure: `reports/figures/december_encoding_comparison.png`.

## Data-quality issues addressed

| Issue | Counts (train / validation) | Handling |
|---|---|---|
| Negative `weight` | 292 / 145 | Sign errors: magnitudes match the positive rows at every quantile, so `abs()` |
| Missing `weight` | 300 / 165 | Imputed in `RateModel.fit` from training medians (31,496 lb), reused unchanged at predict time. Not imputed in cleaning, which would compute the median across both splits |
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
    geography.py  coordinates, haversine, and the city lookup
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

- **Config in Python, not YAML.** At 17 fields a config file costs a parser and a
  dependency while giving up type checking; frozen dataclasses keep it typed and
  navigable.
- **`RandomForestRegressor`, chosen on forward-chained MAE.** Thirteen configurations
  were benchmarked at sensible defaults, then the three leading families were tuned over
  comparable grids, roughly 130 fits in all. Every column below uses training-set median
  imputation, including for the models that can read NaN natively, so the comparison is
  like for like. The five the decision turned on, each at its best known setting:

  | Model | MAE | Chart shape vs real prices |
  |---|---|---|
  | ExtraTrees (tuned) | **128.28** | +0.153 |
  | **RandomForest (tuned)** | 128.93 | +0.233 |
  | LightGBM (tuned) | 131.06 | +0.237 |
  | HistGradientBoosting (tuned) | 135.57 | +0.204 |
  | ElasticNet (tuned) | 148.23 | **+0.382** |

  **Against boosting the margin is decisive.** Paired per load, RandomForest beats
  HistGradientBoosting by $6.64 +/-0.61, the same sign in all three folds.

  **ExtraTrees has the lower pooled MAE**, by $0.64 +/-0.43, but the sign varies by fold
  (-3.41, +4.43, -2.97): it is much better on folds 1 and 3 and clearly worse on fold 2,
  the hardest block. An edge that reverses on one of three time windows will not carry
  into an unseen month, so RandomForest takes it on stability rather than on average.

  **Every one of the five was tuned**, over comparable grids on the same folds, so no
  ranking here rests on one model having been searched and another guessed at. Boosting
  gained heavily from it: XGBoost 150.85 to 133.43, LightGBM 146.48 to 131.06,
  HistGradientBoosting 145.29 to 135.57. Each search was extended until its optimum sat
  interior on every axis rather than against a grid boundary.

  Feature subsampling was the largest single lever for every tree family, the one tuning
  result that generalised. Nothing else did: RandomForest's settings applied to ExtraTrees
  make it worse (135.23), because ExtraTrees already randomises split thresholds and its
  own best setting uses every feature.

  **The folds disagree about which model is better, and what separates them is model
  capacity.** Fold 2, which tests July and August, is the block every model finds hardest,
  and the ranking there is close to reversed. Against RandomForest on fold 2: XGBoost at
  depth 3 is 14.52 better, ElasticNet (a linear model) 13.89 better, LightGBM at four
  leaves 7.28 better. All three pay for it on folds 1 and 3, ElasticNet by 44.41.

  Every model that wins fold 2 is a heavily constrained one, and every search in this
  repo tuned toward more constraint: `max_features` 8 of 15 for the forest, 0.4 for HGB,
  0.3 for XGBoost, `reg_lambda` 50 for both boosters, and ElasticNet's lambda ten times
  its hand-picked value. Five independent searches agreeing is worth more than any single
  one of them. The reading is that July-August departs from what the earlier months teach,
  so capacity spent fitting them is capacity misspent.

  RandomForest is therefore the best single model here, not the best available answer. An
  ensemble weighted toward constrained models on the hardest block is the obvious next
  step and was not attempted. Stated as an open direction rather than left for a reader
  to notice.

  **ElasticNet leads the chart column** and pays 15% on MAE. It is the only candidate
  that extrapolates past the training horizon, so both the lead and the cost are real.

  **The chart column is unstable and is not what decides this.** Edits with nothing to
  do with dates have moved HistGradientBoosting's score across +0.295, +0.074, +0.137 and
  +0.204. Per-month scores swing by half a point or more across the five rehearsal
  months. A column that reorders under unrelated edits cannot carry a model decision, so
  MAE does.

  Chart shape is measured by rehearsing December on five held-out months: train on
  everything prior, build the fixed-lane chart the same way, correlate its shape against
  what comparable loads actually cost. Months are weighted by split-half reliability,
  since October's daily medians replicate at only 0.045. Part of the instability is the
  fixed lane freezing `quote_signal`, which carries 81% of permutation importance.

  Also benchmarked, none of them competitive: XGBoost, which tunes to 133.43 and is
  discussed above; classic GradientBoosting (149.52), Ridge (152.85), KNeighbors k=25
  (187.45), two superseded HGB hand-tunings (144.18 and 152.26), the quote alone with no
  model (286.20), and a mean predictor (329.24). ElasticNet is in the table for its chart
  column rather than its MAE, and its lambda and L1 mix were searched like everything
  else.
- **Hyperparameters were searched, and one of them mattered.** sklearn defaults
  `max_features` to every column, so with `quote_signal` at 81% of permutation importance
  each tree splits on it first and the forest ends up correlated, which is the one thing
  averaging cannot repair. Restricting to 8 of 15 is the single largest win in the search.
  `min_samples_leaf=20` survived its own sweep (10 and 40 are both worse), 400 trees is
  where more stops helping, and `absolute_error`, `friedman_mse`, `ccp_alpha`, `max_depth`
  and `max_samples` were all tried without a gain that held across folds.

  Selection and scoring share the same three folds, so the reported MAE is optimistic as
  a *generalisation* estimate by an amount this design cannot measure. The comparison
  between configurations is unaffected, since every one is scored identically, and the
  random-k-fold contrast above is a separate point about split strategy rather than about
  this number's absolute level.
- **Imputation lives in the estimator, not in cleaning.** RandomForest cannot read NaN.
  Medians are taken from the training matrix and reused unchanged at predict time, so a
  validation row cannot influence its own imputed value. Cleaning stays a pure transform.
- **Leakage boundary.** The daily market level reads the `market_index` feature column
  across both train and validation, never `posted_rate`, because the assessment supplies
  that column for the prediction window. That is a judgment call, so it is stated rather
  than assumed.
- **`tests/test_scorer_contract.py`** asserts the output satisfies every rule in
  `score.py`. A format rejection scores zero regardless of model quality.
