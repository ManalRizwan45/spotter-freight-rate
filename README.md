# Freight Rate Prediction: Spotter ML Assessment

Predicts `posted_rate` for 12,000 held-out freight loads, and produces the fixed
December chart.

## Data

The four supplied files are committed, so the pipeline runs on a fresh clone with nothing
to place by hand:

```
data/train_test.csv                       48,000 labelled loads, Jan-Oct
data/validation.csv                       12,000 unlabelled loads, Nov-Dec
data/december_chart_inputs.csv            31 fixed-input rows; `december` fills these
data/validation_predictions_template.csv  the submission template
```

`december_chart_inputs.csv` with its `predicted_rate` column filled is itself a
deliverable, which is why the directory is committed rather than ignored. The assessment
PDF is not committed, since nothing here reads it and
[`docs/assessment_readme.md`](docs/assessment_readme.md) carries the instructions that
matter. `score.py` is included and unmodified.

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
| `evaluate` | The full metric panel, segment breakdowns and residual diagnostics |
| `predict` | Writes `validation_predictions.csv` |
| `december` | December curves + `december_predictions.csv` |
| `all` | All four |

Then the official scorer, exactly as the assessment documents it:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

> **Where the deliverables land.** `december` fills `data/december_chart_inputs.csv`'s
> `predicted_rate` column in place, which is what the instructions ask for and what the
> command above reads. Filling in place is idempotent: the file is read before it is
> written, and only `predicted_rate` changes. An identical copy is also written to
> `december_predictions.csv` at the repo root, which is the path
> `tests/test_scorer_contract.py` checks and lets the scorer run without reaching into
> `data/`.

Tests and lint:

```bash
python -m pytest && python -m ruff check src tests
```

## Report

[`reports/REPORT.pdf`](reports/REPORT.pdf) is the deliverable: seven pages covering the
pipeline stage by stage, the findings that shaped it, the model comparison and the
results. This README is the long version, and the notebook below is the working one.

`reports/` holds only output. The source is [`docs/REPORT.md`](docs/REPORT.md) and the PDF
is built from it, so a correction is made in one place:

```bash
python docs/make_figures.py && python docs/make_pdf.py
```

`make_figures.py` produces the two figures no pipeline command has reason to write; the
other two come from `cli december` and `cli evaluate`. `make_pdf.py` needs headless Chrome
or Edge, and `pip install -e ".[report]"` for the rest.

## Notebook walkthrough

[`notebooks/01_walkthrough.ipynb`](notebooks/01_walkthrough.ipynb) walks the whole
pipeline in the order it runs, with the evidence for each decision beside the code that
implements it. It imports from `src/freight_rate/` rather than reimplementing anything,
so it cannot drift from what `cli.py` does, and it is committed with all outputs and
figures embedded so it reads without executing.

| Part | Sections | Covers |
|---|---|---|
| I. Exploratory analysis | 1-10 | What the data says, and what each finding forces |
| II. Feature engineering | 11-12 | The 15 columns, why each is there, what each is worth |
| III. Encoding | 13-15 | Equipment, dates, geography, each choice measured |
| IV. Missing values | 16 | Where every gap is filled, and why the placement is the point |
| V. Training and testing | 17-19 | Target transform, the estimator, the forward-chaining harness |
| VI. Evaluation | 20-23 | The panel, what MAE hides, residual diagnostics |
| VII. Deliverables | 24-25 | The December chart and `validation_predictions.csv` |

To re-run it:

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

## Evaluation

`python -m freight_rate.cli evaluate` scores **pooled out-of-fold predictions**: the three
forward folds concatenated, 28,890 loads from May to October, each predicted by a model
that was trained only on earlier months. Scoring the rows the forest was fitted on would
measure its memory instead.

| | model | quote only | mean rate |
|---|---|---|---|
| MAE $ | **128.97** | 286.37 | 1,183.74 |
| RMSE $ | 636.26 | 726.69 | 1,521.83 |
| bias $ | -64.91 | -59.71 | 0.00 |
| MAPE % | 5.70 | 13.97 | 83.51 |
| median APE % | 2.30 | 7.04 | 43.55 |
| P90 APE % | 8.13 | 33.21 | 197.55 |
| WAPE % | 5.35 | 11.88 | 49.12 |
| bias % | -2.69 | -2.48 | 0.00 |
| R² | 0.83 | 0.77 | 0.00 |
| within 5% | 78.2 | 44.9 | 5.5 |
| within 10% | 93.0 | 56.7 | 10.9 |
| within 25% | 98.3 | 82.5 | 26.7 |

MAE reads $128.97 here against $128.93 in the Approach section. Same predictions: pooled
by load rather than averaged across the three folds, which weights folds by their slightly
different test sizes.

The comparison against `quote only` is what says the model earned its place: a **55.0%
MAE cut** against `distance × quote_signal`, the estimate available before any model runs.
`mean rate` is the conventional null and is set as generously as possible, using the mean
of the very rows being scored, which no real predictor would know.

`bias` is signed on purpose. MAE cannot tell a model that quotes 3% high on everything
from one that scatters, and a rate desk can. WAPE is total error over total dollars, so a
$9,000 load counts for more than a $500 one, which MAPE does not do. R² is reported
because it is expected, and it decides nothing: `posted_rate` scales with trip length, so
`distance` alone explains most of its variance and 0.83 largely restates that.

Two findings the pooled MAE hides, both with their measurements in the notebook.

**The model quotes low, by 2.69% of dollars.** It is mechanical rather than a data
problem: the forest fits the mean of the log ratio, and exponentiating a mean lands below
the mean of the exponentials by about `exp(σ²/2)`, which measures 1.30 of the 3 points.
Duan's smearing estimator removes the bias and costs **$8.29 MAE**, because exponentiating
an unbiased log-space fit gives a conditional *median*, and the median is what minimises
absolute error. It ships uncorrected because the assessment scores MAE. Feeding a revenue
forecast rather than a per-load quote, the correction belongs back in.

**The training year holds two regimes, and in one of them the model is worse than no
model at all.**

| Month | model MAE | quote-only MAE | loads paying >20% over quote |
|---|---|---|---|
| 2025-05 | 91.88 | 403.40 | 39.8% |
| 2025-06 | **144.56** | **82.85** | 0.8% |
| 2025-07 | 107.96 | 415.77 | 39.2% |
| 2025-08 | 194.00 | 317.26 | 20.9% |
| 2025-09 | **107.03** | **71.75** | 0.7% |
| 2025-10 | 129.76 | 413.72 | 36.0% |

In January, February, March, June and September the quote is essentially exact, the
median load pays 1.000× and there is no deviation to predict. The model predicts one
anyway, and loses to submitting the raw quote by 49% and 74%. In April, May, July and
October a third of loads run past the quote and the model cuts MAE by 69 to 77%. August is
a third case, active on 21 of its 31 days. The switch is by calendar month and it is
total: every day of an active month is active, every day of a quiet month is quiet.

Nothing available at prediction time separates the two. The daily market level does not:
June is quiet at 1.280 while October is active at 0.957, and the correlation between the
daily market level and the daily share running over quote is +0.42. A `month` feature
would carry it for the ten months in the training file and be useless for the two that
matter, for the same reason the date encoding rejects `date_ordinal`.

So the model hedges, predicting the unconditional mixture: 16.6% of November and December
loads above 1.2 × quote, against 17.5% across the training year. **That is the largest
single risk in this submission and no pooled metric shows it.** If November and December
are quiet, `distance × quote_signal` with no model at all would score better; if they are
active, the model wins by a wide margin. Stated as an open risk rather than left for a
reader to find.

Error shape, rather than error size, is in `reports/figures/error_diagnostics.png`. The
fit holds across the whole rate range, so a single MAE describes it; the error
distribution sits slightly left of zero, which is the bias above; and calibration is flat
through the middle eight deciles of predicted rate, drifting only at the ends, where a
load the model calls cheapest turns out 2.1% dearer than the call and one it calls dearest
1.2% cheaper.

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
  evaluation/
    metrics.py    the four numbers every configuration was selected on
    report.py     the full panel, segment breakdowns and the baselines
    charts.py     figures
  december.py     reconstructing the chart file's missing columns
  cli.py          entry point
tests/            63 tests, including a scorer-contract guard
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
  and the ranking there is close to reversed. The month breakdown in the Evaluation
  section localises that to August specifically, at MAE 194.00 against July's 107.96. Against RandomForest on fold 2: XGBoost at
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
  column rather than its MAE. Its lambda and L1 mix were searched jointly, coarsely then
  finely: the optimum is a ridge rather than a peak, since the model responds to the
  products lambda*mix and lambda*(1-mix) rather than to either number alone, and
  everything along it lands within 0.15 MAE.
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
- **The daily market level is built from both files, and that is not leakage.** It reads
  the `market_index` feature column and never `posted_rate`, and each date's level is
  averaged over that date's own loads. Train is January to October and validation is
  November and December, so the two never contribute to the same date, and `market_index`
  is supplied for the validation rows. Nothing here uses information that would be absent
  at prediction time.
- **`tests/test_scorer_contract.py`** asserts the output satisfies every rule in
  `score.py`. A format rejection scores zero regardless of model quality.
