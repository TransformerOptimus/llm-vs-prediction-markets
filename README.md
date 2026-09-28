# Sorting, Not Calibration: code and paper

This repository holds the code and the paper for *Sorting, Not Calibration: A
Price-Blind Comparison of Language Models and Prediction Markets*
(<<FILL: paper link>>).

## What the study does, in short

A **prediction market** sells a contract that pays $1 if an event happens, so
its trading price works as the crowd's probability for that event. We asked
eight language models for the same probability, at the same moments, without
ever showing them the price ("price-blind"). Then we scored models and market
against what actually happened.

The study covers 10,536 **market-moments** (one market at one forecast time)
from three stretches of time, called **windows**:

| Window | Exchange | Dates | Market-moments |
|---|---|---|---|
| A | Polymarket, taken from the published PolyBench dataset | 6 to 12 February 2026 | 2,997 |
| B | Polymarket | 15 July to 9 August 2026 | 3,619 |
| C | Kalshi | 14 May to 10 June 2026 | 3,920 |

The main finding: the market was more accurate than every model, and the gap is
in *sorting*, not *calibration*. The models' probabilities were roughly honest
(when a model said 70%, the event happened close to 70% of the time), but the
models separated likely outcomes from unlikely ones less well than the crowd
did. The models were also often wrong together. The paper has the full results;
all of them are exploratory.

## What is in this repository

| Folder or file | What it holds |
|---|---|
| [`harness/`](harness/) | The code that asks each model for its forecast and records every call. |
| [`analysis/`](analysis/) | The code that scores the forecasts and produces every table and figure in the paper, plus the exploratory analyses and their saved outputs. |
| [`data/`](data/) | Where the data lives (two datasets on Hugging Face), how to get it, and its licences. |
| [`paper/`](paper/) | The paper as a PDF and its LaTeX source. |
| `requirements.txt` | The Python packages the code needs. |
| `LICENSE` | Apache License 2.0, for the code. |
| `CITATION.cff` | How to cite this work (GitHub shows it as "Cite this repository"). |

Every folder has its own README explaining what is in it.

## The data

The data is not stored here. It is in two datasets on Hugging Face:

- **Main dataset** (<<FILL: main dataset URL>>, about 58 MB): the market-moments,
  every model call, the fixed analysis inputs and the result tables.
- **News dataset** (<<FILL: news dataset URL>>, about 10 MB): the full text of the
  news each model was shown.

See [`data/README.md`](data/README.md) for what each holds and how it is licensed.

## Reproducing the results

You need Python 3.9 or newer.

```sh
pip install -r requirements.txt

# 1. Download the two datasets (paths below are wherever you saved them).
# 2. Rebuild the local databases the code reads:
python3 analysis/build_db.py --main PATH_TO_MAIN_DATASET --news PATH_TO_NEWS_DATASET

# 3. Produce the paper's result tables (written to analysis/out/results/):
python3 analysis/results.py

# 4. Run any exploratory analysis; it prints its numbers:
python3 analysis/exploratory/calibration-shape-resolution-gap.py
```

Windows A and B rebuild in full from the release. Window C (Kalshi) ships only
market identifiers and our own labels, because we cannot redistribute Kalshi's
market data. To rebuild Window C you need that data yourself; the header of
`analysis/build_db.py` and the main dataset's card explain what is needed.

Running the models again is not needed to reproduce anything: every model reply
is in the main dataset. If you do want to run the harness, it needs your own API
keys and costs money; see [`harness/README.md`](harness/README.md).

## Where each number in the paper comes from

Table IV in the paper's appendix lists, for each result, the script and the
released file it comes from.

## Licences

- Code: Apache License 2.0 (`LICENSE`).
- Data: CC BY 4.0, except Window A, which comes from PolyBench and keeps its
  CC BY-NC-SA 4.0 licence. Details in [`data/README.md`](data/README.md).
- The paper: <<FILL: paper licence>>.

## Citation

<<FILL: citation block>>
