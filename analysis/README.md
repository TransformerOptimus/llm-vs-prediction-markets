# Analysis: scoring the forecasts

This folder turns the recorded model forecasts into the paper's numbers. It
scores every forecast against what actually happened, compares it with the
market price at the same moment, and writes the result tables.

It reads two local SQLite databases and never changes them:

- `benchmark/benchmark.db`: the market-moments (questions, prices, order books,
  news, outcomes).
- `harness/runs/runs.db`: every model call and its answer.

Neither database is in this repository. `build_db.py` rebuilds both from the
released datasets (see [`../data/README.md`](../data/README.md)):

```sh
python3 analysis/build_db.py --main PATH_TO_MAIN_DATASET --news PATH_TO_NEWS_DATASET
python3 analysis/results.py
```

`build_db.py` rebuilds Windows A and B. Window C (Kalshi) also needs Kalshi's own
market data, which we cannot redistribute; the header of `build_db.py` and the
main dataset's card explain what to supply.

## What each file does

| File | What it does |
|---|---|
| `build_db.py` | Rebuilds the two databases, and the fixed inputs in `out/`, from the released Parquet files, in exactly the layout the rest of this code reads. |
| `scoring.py` | The core. Takes one forecast and returns one scored row: the Brier score for the model and for the market, the trade the model's forecast implies and its return, which way the model disagreed with the market, the price band and the depth bucket. Also holds the bootstrap and the depth tests. |
| `harness_loader.py` | Reads the model calls out of `runs.db` and joins each one to its market-moment. Counts skipped, failed and flagged calls per run. |
| `results.py` | Produces the paper's result tables in `out/results/`, each as a CSV file and a readable Markdown copy. `--n-boot` sets fewer bootstrap draws for a quick run; `--no-figure` skips the figure. |
| `gates.py` | Two checks: the memory-probe quarantine rule (flags rows where a model seems to know the outcome without being told) and the calibration check. |
| `correlated_errors.py` | Measures how often models make the same mistakes: the correlation of their errors, and how often two models are wrong together compared with chance. |
| `no_skill.py` | Five strategies that need no forecasting skill (always buy Yes, always buy No, buy the cheaper side, buy the side the market favours, flip a coin), scored with the same trading rule and fees, as a floor for comparison. |
| `price_paths.py` | How each market's price moved after the forecast time. |
| `pair_compare.py` | Compares two models on the rows both forecast. Used for GPT-5.2 against GPT-5.5; works for any two models. |
| `window_shape.py` | Describes a window before any scoring: rows, events, news coverage, order-book state and price range per depth bucket. |
| `make_split.py` | Draws the split of rows into a calibration part and a test part (seed 20260902). The released data already carries the split, so you do not need to run it. |
| `make_validation_sample.py` | Draws the fixed 500-row Window A sample used to check the memory probe. |
| `first_runs_report.py` | Accounting for the runs: memory-probe flags per run, the validation comparison, the calibration check and failure counts. |
| `test_fees.py`, `test_gates.py`, `test_results.py` | Self-checks: the fee formulas against hand-worked examples, both checks in `gates.py`, and `results.py` on a small made-up dataset. Run each with `python3 analysis/<name>.py`. |
| [`exploratory/`](exploratory/) | The exploratory analyses, one script per question, each with its saved output. See its README. |
| `out/` | Not in the repository. Created by `build_db.py` (fixed inputs) and `results.py` (result tables). |

The result tables fall into three groups by file name: `expA_` files are about
order-book depth, `expB_` files about the memory probe, calibration, disagreement
with the market and later price moves, and `expC_` files about correlated errors.

## Terms used in the code, in plain words

- **Row**: one model's probability of Yes on one market-moment.
- **Market probability** (`q` in the code): the midpoint between the best price
  to buy Yes and the best price to sell it. On Window A the stored price in the
  source data is a stale copy, so the midpoint is taken from the order book; on
  Windows B and C the stored price already equals the midpoint.
- **Brier score**: the squared gap between a probability and what happened (1 if
  Yes, 0 if No). Lower is better. **Brier edge** is the market's Brier score
  minus the model's; positive means the model was more accurate.
- **Return**: what the model's forecast would have earned as a trade. If the
  forecast is above the best price to buy Yes, buy Yes at that price; if it is
  below the best price to sell Yes, buy No; otherwise, no trade. Reported as
  profit per dollar staked, after the exchange's fee. Books with only one side,
  or with a gap between buy and sell prices above 0.50, are never traded, but
  those rows still count for accuracy.
- **Depth**: dollars resting on the order book at the five best prices on each
  side (`depth_usd`). Rows are split into five equal-sized **depth buckets** per
  window, with the cut points stored in `out/depth_cuts.json`.
- **Primary set**: the rows the main results use: not a ladder (see below), no
  news article dated after the forecast time, end date not revised, and market
  probability between 0.05 and 0.95.
- **Ladder**: a group of questions in one event that differ only by a number
  (for example "above 80°F", "above 85°F"). Scored separately.
- **Sports**: a row whose topic tags name a sport (`scoring.is_sports`).
- **One window at a time**: depth buckets mean different dollar amounts in each
  window, so any function that groups by bucket refuses rows from more than one
  window. A window is identified by exchange and data source together.
- **Two arms**: most tables are produced twice, on rows that came with news and
  on all rows.
- **With or against the crowd**: *with* means the model leaned further toward the
  side the market already favoured; *against* means the other way.
- **Intervals**: 95% bootstrap intervals. The bootstrap re-draws whole events
  (the exchange's grouping of related markets) at random, 2,000 times, with seed
  20260902, and reads the spread of the results.
- **`moment_key`**: a row's permanent name (exchange, market id, forecast time).
  Use it, not `row_id`, to join tables.
- **`db_sha256_matches_live`** in `reporting_line.csv` compares the database
  checksum recorded at run time with the file on your disk. It reads False on
  any rebuilt database, because a rebuilt SQLite file never has the original's
  exact bytes; the contents are the same (the datasets' `MANIFEST.sha256` files
  check the released files).
