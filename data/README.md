# Data

The data is not stored in this repository. It is published as two datasets on
Hugging Face, each with its own card (a README describing every file and
column):

| Dataset | Size | What it holds |
|---|---|---|
| Main dataset (<<FILL: main dataset URL>>) | about 58 MB | The market-moments of all three windows, every model call (prompt, reply, probability, tokens, host, settings), the fixed analysis inputs, and the result tables. |
| News dataset (<<FILL: news dataset URL>>) | about 10 MB | The full text of the news each model was shown, stored once per distinct text and linked to the main dataset by checksum. |

Download both, then rebuild the local databases the code reads:

```sh
python3 analysis/build_db.py --main PATH_TO_MAIN_DATASET --news PATH_TO_NEWS_DATASET
```

This creates `benchmark/benchmark.db`, `harness/runs/runs.db` and
`analysis/out/` at the repository root (none of them is tracked by git). The
news dataset is optional: without it the news text is left empty, and no result
table needs it.

## Window C (Kalshi)

We cannot redistribute Kalshi's market data. For Window C the main dataset
holds only market identifiers, forecast times and our own labels, never
Kalshi's prices, order books, outcomes, questions or rules. Model replies on
Window C are also left out, since they quote the questions. Result tables keep
Window C counts but blank the scores in any cell built on fewer than 10 markets.
To rebuild Window C in full, collect the Kalshi data yourself and pass it with
`build_db.py --kalshi DIR`. The header of `analysis/build_db.py` and the main
dataset's card list exactly what is needed.

## Personal data

Every email address and phone number found in the news text and model replies
was replaced by `[email removed]` or `[phone removed]` before release. To ask for
a correction or removal, open an issue: <<FILL: takedown issues URL>>.

## Licences

| Files | Licence | Full text |
|---|---|---|
| Everything except Window A | CC BY 4.0: free to use, share and adapt, with credit | `LICENSE-DATA` |
| Window A (market rows and news, taken from the published PolyBench dataset) | CC BY-NC-SA 4.0: non-commercial use only, with credit, and adaptations shared under the same licence | `LICENSE-WINDOW-A` |

The news articles were written by their publishers; each article row in the
main dataset carries its URL, site and capture time. The code in this
repository is under the Apache License 2.0 (`../LICENSE`).
