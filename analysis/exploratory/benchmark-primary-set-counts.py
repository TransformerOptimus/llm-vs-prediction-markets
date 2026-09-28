import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""benchmark-primary-set-counts: the size of each window's primary analysis set.

Table I of the paper gives, per window, the number of primary rows, how many of them carry
news, and how many are sports. This script counts them from the benchmark alone.

Primary set, news flag and sports flag are exactly those of analysis/scoring.py
(load_moments): a row is primary when its market price is between 0.05 and 0.95, it is not a
ladder, its news has no late dateline and its end date was not revised; "with news" is
news_available = 1; sports is any topic tag naming a sport (scoring.is_sports). Both splits.
Read-only on the benchmark. Window C needs the Kalshi data that build_db.py --kalshi merges.
"""
import scoring

con = scoring.connect()
moments = scoring.load_moments(con)
con.close()

print("rows: every market-moment in the benchmark; primary set, news flag and sports flag as in "
      "analysis/scoring.py; both splits")
for w in "ABC":
    rows = [m for m in moments.values() if m["window"] == w]
    prim = [m for m in rows if m["primary"]]
    news = sum(1 for m in prim if m["news_available"])
    sports = sum(1 for m in prim if m["sports"])
    print("window %s: drawn %d | primary %d | primary with news %d | primary sports %d (%.0f%%)"
          % (w, len(rows), len(prim), news, sports, 100.0 * sports / len(prim) if prim else 0.0))
