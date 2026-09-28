import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""event-definition-and-cluster-counts: what an "event" is in this benchmark, and how many clusters
the bootstrap actually draws from.

Every interval in the paper comes from resampling events rather than rows. This script shows what an
event is and how many there are.

An event here is the venue's own grouping of markets that belong to one thing, carried into the
benchmark as moment_outcomes.source_event_id: on Polymarket the event id behind a group of markets
(for example the twelve "will Warsh say X" markets of one press conference), on Kalshi the event
ticker that the individual market tickers hang off (for example one golf tournament's "will player X
make the cut" markets). scoring.load_moments turns it into the bootstrap cluster event_id, and a row
whose source_event_id is missing becomes its own cluster ("row:<row_id>"). Example events are printed
so the grouping can be seen rather than taken on trust.

Per window, on the test split of the primary analysis set, and on the full primary set for reference:

  - the number of distinct events;
  - markets per event: the smallest, the median, the largest, and the share of events holding exactly
    one market;
  - the number of clusters the bootstrap therefore draws from -- which is just that event count, since
    every bootstrap function in scoring.py groups the rows it is given by event_id and resamples those
    groups with replacement, the same number of draws as there are groups.

The common rows used by the correlated-errors scripts (rows every model in the window's roster
forecast) are counted separately, because those scripts bootstrap over a smaller set of events.

One market-moment is one market at one forecast time; the count of distinct markets is printed beside
the count of moments so it is clear whether a market ever appears twice. Read-only on both databases.
"""
import statistics as st
from collections import Counter, defaultdict

import correlated_errors_common as hc
import scoring


def describe(label, ms):
    """Print the event counts and the markets-per-event distribution for a set of market-moments."""
    ev = Counter(m["event_id"] for m in ms)
    if not ev:
        print("     %-34s (no rows)" % label)
        return
    sizes = sorted(ev.values())
    ones = sum(1 for c in sizes if c == 1)
    missing = sum(1 for m in ms if m["source_event_id"] is None)
    print("     %-34s %6d moments (%d distinct markets) in %d events; per event min %d, median %.1f, "
          "max %d; exactly one market: %d events (%.1f%%); moments with no source_event_id (own "
          "cluster): %d"
          % (label, len(ms), len({m["venue_market_id"] for m in ms}), len(ev), sizes[0],
             st.median(sizes), sizes[-1], ones, 100.0 * ones / len(ev), missing))
    print("     %-34s bootstrap draws %d clusters with replacement per resample" % ("", len(ev)))


con = scoring.connect()
MOMENTS = scoring.load_moments(con)
con.close()

print("rows: benchmark.db market_moments joined to moment_outcomes; primary analysis set unless a "
      "line says otherwise; an event is moment_outcomes.source_event_id, and a moment with none is "
      "its own cluster. Bootstrap settings: scoring.py resamples events, seed %d (%d resamples in "
      "the result tables, %d in the exploratory scripts)." % (scoring.SEED, scoring.N_BOOT, hc.N_BOOT))

print("")
print("=== what source_event_id groups: the three largest events of each window, primary set")
for w in "ABC":
    per = defaultdict(list)
    for m in MOMENTS.values():
        if m["window"] == w and m["primary"]:
            per[m["event_id"]].append(m)
    print("  window %s" % w)
    for ev, ms in sorted(per.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))[:3]:
        print("    event %s: %d markets, event title %r" % (ev, len(ms), ms[0]["event_title"]))
        for m in sorted(ms, key=lambda m: m["venue_market_id"])[:3]:
            print("      market %s: %s" % (m["venue_market_id"], m["question"][:90]))
        if len(ms) > 3:
            print("      ... and %d more markets in the same event" % (len(ms) - 3))

print("")
print("=== events and bootstrap clusters per window")
for w in "ABC":
    print("  window %s" % w)
    prim = [m for m in MOMENTS.values() if m["window"] == w and m["primary"]]
    describe("primary set, test split:", [m for m in prim if m["split"] == "test"])
    describe("primary set, calibration split:", [m for m in prim if m["split"] == "calibration"])
    describe("primary set, both splits:", prim)

print("")
print("=== the same for the common rows the correlated-errors scripts use")
print("    (one row per market-moment that EVERY model in the window's roster forecast)")
data, _ = hc.load("normal")
for w in hc.WINDOWS:
    rows = data.get(w, [])
    print("  window %s" % w)
    for label, sel in (("common rows, test split:", [r for r in rows if r["split"] == "test"]),
                       ("common rows, both splits:", rows)):
        ev = Counter(r["event_id"] for r in sel)
        if not ev:
            print("     %-34s (no rows)" % label)
            continue
        sizes = sorted(ev.values())
        ones = sum(1 for c in sizes if c == 1)
        print("     %-34s %6d rows in %d events; per event min %d, median %.1f, max %d; exactly one "
              "row: %d events (%.1f%%); bootstrap draws %d clusters"
              % (label, len(sel), len(ev), sizes[0], st.median(sizes), sizes[-1], ones,
                 100.0 * ones / len(ev), len(ev)))
