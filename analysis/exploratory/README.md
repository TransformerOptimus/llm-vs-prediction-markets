# Exploratory analyses

Each script here asks one question of the data and prints the numbers that
answer it. The paper's appendix uses them, and Table IV in the paper names the
script behind each result. Every script was run over all eight models, and its
printed output is saved in [`outputs/`](outputs/), so you can read the results
without running anything.

## Running a script

Build the databases first (`analysis/build_db.py`; see
[`../README.md`](../README.md)), then run from the repository root:

```sh
python3 analysis/exploratory/<script>.py                # prints its numbers
python3 analysis/exploratory/<script>.py --test-split   # only the held-out test rows (where supported)
python3 analysis/exploratory/<script>.py --figure       # also saves a figure into figures/ (where supported; needs matplotlib)
```

Every script opens both databases read-only. Unless a script says otherwise,
it uses the primary set (see [`../README.md`](../README.md)) on both the
calibration and the test split, and its 95% intervals come from re-drawing
whole events at random 1,000 times with seed 20260902.

Three scripts build simulated comparison forecasters (`correlated-errors-shrinkage-null`,
`correlated-errors-shrinkage-null-news-rows` and `correlated-errors-covariate-adjusted`).
Their simulated values can differ in the third decimal from one run to the next,
because Python varies the order in which it walks through a set; the models' own
numbers never change.

Window C lines need the Kalshi data that `build_db.py --kalshi` merges. Window C
reply text is not released, so the two reply-text scripts cannot reproduce their
Window C lines from the release.

## The scripts, by topic

Script names start with their topic. A few scripts have a saved output for the
test split as well (`outputs/<script>-test-split.txt`).

**Benchmark and baselines**

| Script | Question |
|---|---|
| `benchmark-primary-set-counts` | How many rows each window's primary set has, with news and in sports (Table I). |
| `baselines-whole-window-edge` | How far each model trails the market over a whole window, with an interval. |
| `baselines-absolute-and-horizon` | How good the models are in absolute terms and against naive forecasters, and whether that depends on how far ahead the forecast was made. |
| `event-definition-and-cluster-counts` | What an "event" is, and how many events the bootstrap re-draws from. |
| `split-leakage-event-grouped` | Does splitting rows one market at a time, rather than one event at a time, flatter the recalibration check? |

**Calibration and sorting** (how honest the probabilities are, and how well they separate likely from unlikely outcomes)

| Script | Question |
|---|---|
| `calibration-shape-resolution-gap` | Is the models' shortfall a sorting problem rather than one that rescaling could fix? |
| `calibration-shape-yes-no-asymmetry` | Do the models fall shorter when the market favours Yes than when it favours No? |
| `calibration-shape-model-spread` | How the pattern above differs between models on the same markets. |
| `calibration-shape-sports-split` | Does sorting, not calibration, also explain the gap on non-sports rows? |
| `calibration-shape-topic` | How sports and non-sports rows differ in price, horizon and depth. |
| `decomposition-binning-bias` | Does the calibration/sorting split hold without the usual ten bins? |
| `decomposition-why-reliability-is-small` | Why the calibration-error term comes out small. |
| `recalibration-crossfit-and-isotonic` | How much of the gap rescaling the models' probabilities can recover, without new information. |

**Direction of disagreement and money**

| Script | Question |
|---|---|
| `direction-and-money-against-crowd-carries-the-loss` | Does the loss come from rows where the model leans against the market? |
| `direction-and-money-against-means-buying-the-longshot` | Does leaning against the market mean buying cheap long shots, and how do those bets do? |
| `direction-and-money-same-gap-different-cost` | Does the same distance from the price cost more against the crowd than with it? |
| `direction-same-side-share` | How much of "leaning against the crowd" is real disagreement about which side wins? |
| `fees-stake-sensitivity` | Are the trading losses due to the forecasts, or to trading one dollar at a time? |

**Correlated errors** (models making the same mistakes)

| Script | Question |
|---|---|
| `correlated-errors-outcome-inflates-r` | How much of the error correlation is just the shared outcome? |
| `correlated-errors-pair-ranking` | Which model pairs agree most closely? |
| `correlated-errors-both-wrong-beyond-market` | On rows the market got right, how often are two models wrong together, compared with chance? |
| `correlated-errors-shrinkage-null` | How much agreement comes from every model pulling toward the same middle? |
| `correlated-errors-shrinkage-null-news-rows` | The same check on rows with news only. |
| `correlated-errors-covariate-adjusted` | How much agreement is left after accounting for what the models were shown? |
| `correlated-errors-with-and-without-news` | Does agreement fall when the models get news? |
| `correlated-errors-news-covariance-or-variance` | If it falls, is that because the models share less, or because each gets noisier? |

**Combining models**

| Script | Question |
|---|---|
| `ensembles-and-extremizing` | Does averaging the models, or pushing the average away from 0.5, close the gap to the market? |
| `ensembles-learned-stacking` | Do learned weights do better than equal weights? |

**News**

| Script | Question |
|---|---|
| `probe-news-value-and-reask-rate` | What the news was worth to each model, and how often a model needed the stricter re-ask. |
| `information-gap-news-and-recency` | Is the gap about the models knowing less, or judging worse? |
| `news-effect-toward-market` | Does news help only when it moves the model toward the market price? |
| `news-toward-away-placebo` | How much of that result is built into the way "toward" is defined? |
| `news-effect-coin-flip` | Row by row, how often does news help and how often does it hurt? |
| `news-effect-high-price-gap` | On markets priced high for Yes, how far does news move the models? |
| `news-effect-nonsports-moves` | Does news move forecasts more on non-sports rows than on sports rows? |

**Sports and topic**

| Script | Question |
|---|---|
| `topic-and-sports-brier-gap` | Is the gap to the market wider on non-sports rows? |
| `topic-and-sports-divergence` | Do the models stray further from the price on non-sports rows? |
| `topic-and-sports-news-shift` | Does news move forecasts more on non-sports rows? |

**Venue and window**

| Script | Question |
|---|---|
| `venue-and-window-kalshi-gap-is-question-mix` | Is the wider gap on Kalshi due to the kind of questions, not the exchange? |
| `venue-and-window-kalshi-hedging-on-obscure-sports` | Do the models stay near 0.5 on obscure Kalshi sports markets? |
| `venue-and-window-no-lean-fades-by-window` | Does the models' lean toward No on non-sports rows differ by window? |

**Time to close**

| Script | Question |
|---|---|
| `horizon-news-value-final-day` | Is news worth more on the last day before a market closes? |
| `horizon-kalshi-sports-same-day-deficit` | Is the gap on Kalshi sports widest for same-day markets? |

**Price paths after the forecast**

| Script | Question |
|---|---|
| `convergence-toward-outcome` | When the price later moves toward the model, is that just the market moving toward the outcome? |
| `convergence-settled-in-window` | The same question for markets that closed within 72 hours. |
| `convergence-open-market-move` | The same question for markets still open after 72 hours. |
| `convergence-baseline-choice` | How much does the result depend on how the random comparison is built? |

**Ladders and extreme prices**

| Script | Question |
|---|---|
| `ladders-and-extremes-exclusion-hides-gap` | How much wider is the gap once ladders and extreme prices are put back? |
| `ladders-and-extremes-ladder-overweight` | Do the models overrate long shots more inside ladders? |
| `ladders-and-extremes-longshot-floor` | How much probability do the models give events the market prices near zero? |
| `ladders-and-extremes-news-hurts-lo-tail` | Does news make forecasts worse on events priced near zero? |

**Thinking and reply length**

| Script | Question |
|---|---|
| `thinking-and-length-more-thinking-more-disagreement` | Are the rows a model thinks longest about the ones where it strays furthest from the price? |
| `thinking-and-length-budget-cap-harmless` | Did the 1,000-token thinking cap hurt accuracy? |
| `thinking-and-length-latency-buys-nothing` | Do slower answers do any better? |

**Failures**

| Script | Question |
|---|---|
| `failures-imputation-sensitivity` | Does leaving out the rows where a model gave no usable number change the results? |
| `failures-and-cutoffs-news-triggers-overrun` | Do long news blocks make replies run past the length limit? |
| `failures-and-cutoffs-hard-rows-shared` | Do different models fail on the same rows? |

**Reply text**

| Script | Question |
|---|---|
| `reply-text-hedging` | How often do replies hedge ("hard to say", "limited information")? |
| `reply-text-news-cited` | How often do replies cite the news, and does citing it go with better forecasts? |

## Helper modules

These are shared code, not analyses:

- `_paths.py`: where the databases are; puts `analysis/` on the import path.
- `_roster.py`: the list of models, read from `harness/models.json`.
- The shared loaders and helpers used by each topic: `calibration_shape_common.py`,
  `convergence_common.py`, `correlated_errors_common.py`, `failures_common.py`,
  `horizon_common.py`, `ladders_common.py`, `news_effect_common.py`,
  `reply_text_common.py`, `common_ins.py`, `common_root.py`, `common_scripts.py`,
  `common_topic.py`, `_common_thinking.py`, `_dm_common.py`, `_lib.py`,
  `_report.py`, `shared.py` and `vw_base.py`.
