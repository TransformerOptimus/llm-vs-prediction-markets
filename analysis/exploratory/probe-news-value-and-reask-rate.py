import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""probe-news-value-and-reask-rate: what the news was worth to each model, and how often a model
needed the stricter re-ask.

1. Value of news. Every news-bearing row was asked twice, with the news (mode normal) and
   without it (mode memory_probe). Per model and window, on the rows where both forecasts have a
   parsed probability, the value of news is the mean Brier score without the news minus the mean
   Brier score with it, so positive means the news helped. Primary set; the test split (the rows
   of the paper's tables) and both splits.

2. Re-ask rate. An unparseable reply is re-asked once with a stricter instruction. Per scored
   run (mode normal), the share of the rows the model was actually asked (final records, rows
   skipped by benchmark flag, eligibility or the leak check left out) on which the stricter
   re-ask was sent.

Read-only on both databases. Window C needs the Kalshi data that build_db.py --kalshi merges.
"""
import sqlite3
from collections import defaultdict

from calibration_shape_common import SHORT, load

with_news, _ = load("normal")
without, _ = load("memory_probe")
wo = {(r["forecaster"], r["row_id"]): r["brier_model"] for r in without}

print("1. value of news = mean Brier without news minus mean Brier with news, on news-bearing primary "
      "rows where both forecasts parsed; positive = the news helped")
for label, keep in (("test split", lambda r: r["split"] == "test"), ("both splits", lambda r: True)):
    cells = defaultdict(list)
    for r in with_news:
        k = (r["forecaster"], r["row_id"])
        if r["news_available"] and k in wo and keep(r):
            cells[(r["window"], r["forecaster"])].append(wo[k] - r["brier_model"])
    vals = []
    print("=== %s" % label)
    for (w, mdl) in sorted(cells):
        v = cells[(w, mdl)]
        vals.append(sum(v) / len(v))
        print("  window %s %-16s n=%-5d value of news %+.4f" % (w, SHORT[mdl], len(v), vals[-1]))
    print("  RANGE over %d model-window cells: %+.4f to %+.4f" % (len(vals), min(vals), max(vals)))

print()
print("2. stricter re-ask: share of asked rows (final records, skipped rows left out) that needed it, "
      "per scored run")
rc = sqlite3.connect("file:%s?mode=ro" % _paths.RUNS_DB, uri=True)
runs = rc.execute("SELECT run_id, model, window FROM runs WHERE mode='normal' AND dry_run=0 AND smoke=0 "
                  "AND void=0 AND state='finished' ORDER BY model, window").fetchall()
for run_id, model, w in runs:
    if model not in SHORT:
        continue
    finals = rc.execute("SELECT row_id, status FROM calls WHERE run_id=? AND final=1", (run_id,)).fetchall()
    asked = {rid for rid, st in finals if not st.startswith(("excluded", "ineligible", "leak_check"))}
    strict = {rid for (rid,) in rc.execute("SELECT DISTINCT row_id FROM calls WHERE run_id=? AND strict=1",
                                           (run_id,))} & asked
    print("  window %s %-16s asked %-5d re-asked %-4d share %.1f%%"
          % (w, SHORT[model], len(asked), len(strict), 100.0 * len(strict) / len(asked)))
