import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""split-leakage-event-grouped: the calibration/test split was drawn one market at a time, so two
markets about the same event can land on opposite sides. Does that flatter the recalibration control?

Background. analysis/make_split.py draws the calibration split per MARKET-MOMENT: inside each cell
(window x depth bucket x sports flag) it sorts the moment keys and picks 20 per cent of them at
random, seed 20260902. An "event" is the exchange's own grouping of markets about one thing (stored
as source_event_id: for example the twelve "will player X make the cut" markets of one golf
tournament). Nothing in the draw keeps an event together, so one market of an event can be in the
calibration split while its sibling is in the test split.

Why that could matter. The recalibration control fits a two-parameter logistic map (a "Platt map":
new probability = sigmoid(intercept + slope x log-odds of the forecast)) on the calibration split and
scores it on the test split. If the two splits share events, the map is fitted partly on rows that
are near-copies of the rows it is then scored on, which would make recalibration look better than it
is -- and the paper's claim is that recalibration recovers only a small share of the gap.

This script does two things.

(a) EXPOSURE. Per window, on the test split: how many events hold at least one calibration market and
    at least one test market ("straddling events"), and what share of test markets and of test rows
    (one row = one model on one market-moment) sit in such an event. Counted on the primary analysis
    set, and also on every moment the split covers, so neither framing is hidden.

(b) AN EVENT-GROUPED SPLIT. A fresh calibration/test partition of the same size that never breaks an
    event apart. Each event is assigned to the cell where most of its moments sit; inside a cell the
    events are shuffled with seed 20260902 and whole events are moved into the calibration side while
    that brings the cell closer to its 20 per cent target. Every market of an event therefore lands on
    the same side, and the cells keep the sizes make_split.py gave them.

    The single-split Platt recovery is then reported per model and window under three column sets,
    side by side: the split the paper uses, the same split scored only on test rows whose event does
    not straddle (same fitted map, leakage-free rows), and the event-grouped split (fitted and scored
    with no shared events at all). "Recovery" is the share of the model's Brier gap to the market that
    the rescaling removes, exactly as in recalibration-crossfit-and-isotonic.py.

    The cross-fitted Platt and isotonic figures in that script are NOT affected and are not repeated
    here: they already build their folds by event, so no fold ever shares an event with its training
    data.

Read-only on both databases. Nothing here rewrites the split stored in the benchmark.
"""
import math
import random
from collections import Counter, defaultdict

import numpy as np
from calibration_shape_common import MODELS, SHORT, load

import scoring

SEED = 20260902
SHARE = 0.2
EPS = 1e-4


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def platt_fit(ps, ys):
    X = np.array([[1.0, logit(p)] for p in ps])
    y = np.array(ys, dtype=float)
    w = np.zeros(2)
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (pr - y)
        H = (X * (pr * (1 - pr))[:, None]).T @ X + 1e-6 * np.eye(2)
        w -= np.linalg.solve(H, g)
    return w


def platt_apply(w, ps):
    return [1 / (1 + math.exp(-(w[0] + w[1] * logit(p)))) for p in ps]


def recovery(rows, newp):
    """(model Brier, rescaled Brier, market Brier, share of the gap removed) or None if too few rows."""
    if len(rows) < 50:
        return None
    bm = sum(r["brier_model"] for r in rows) / len(rows)
    bq = sum(r["brier_market"] for r in rows) / len(rows)
    bn = sum((n - r["outcome_yes"]) ** 2 for n, r in zip(newp, rows)) / len(rows)
    gap = bm - bq
    return bm, bn, bq, ((bm - bn) / gap if gap > 0 else float("nan"))


def event_grouped_split(moments, cuts):
    """moment_key -> 'calibration' / 'test', with every market of an event on the same side.

    Same population and same cells as make_split.py (every moment with news_dateline_after_capture = 0,
    cells of window x depth bucket x sports), but the unit drawn is the event, not the market.
    """
    inplay = [m for m in moments.values() if not m["news_dateline_after_capture"]]
    cell_of = {}
    per_event = defaultdict(list)
    for m in inplay:
        per_event[m["event_id"]].append(m)
    for ev, ms in per_event.items():
        counts = Counter((m["window"], scoring.bucket_of(m, cuts), m["sports"]) for m in ms)
        cell_of[ev] = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]
    cells = defaultdict(list)
    for ev in per_event:
        cells[cell_of[ev]].append(ev)
    out = {}
    for cell in sorted(cells, key=str):
        evs = sorted(cells[cell])
        n = sum(len(per_event[e]) for e in evs)
        target = max(1, int(round(SHARE * n)))
        random.Random(SEED).shuffle(evs)
        chosen, got = set(), 0
        for e in evs:
            k = len(per_event[e])
            if abs(got + k - target) < abs(got - target):
                chosen.add(e)
                got += k
        if not chosen:
            e = min(evs, key=lambda e: (len(per_event[e]), e))
            chosen.add(e)
        for e in evs:
            side = "calibration" if e in chosen else "test"
            for m in per_event[e]:
                out[m["moment_key"]] = side
    return out


con = scoring.connect()
MOMENTS = scoring.load_moments(con)
CUTS = scoring.load_or_compute_cuts(MOMENTS)
con.close()

rows_all, _ = load("normal")           # primary set, every scored model, mode normal
NEW = event_grouped_split(MOMENTS, CUTS)

print("rows: primary set, forecasts view, mode normal; one row = one model on one market-moment. "
      "Events are source_event_id (a moment with no event id is its own event). The stored split is "
      "analysis/make_split.py's per-market draw (20%% per cell of window x depth bucket x sports, "
      "seed %d); the event-grouped split redraws the same cells with whole events, seed %d. "
      "Recovery = share of the model's Brier gap to the market that the fitted Platt map removes."
      % (SEED, SEED))

print("")
print("=== (a) exposure: how much of the test split shares an event with the calibration split")
for w in "ABC":
    for scope, keep in (("primary set", lambda m: m["primary"]),
                        ("every moment the split covers", lambda m: not m["news_dateline_after_capture"])):
        ms = [m for m in MOMENTS.values() if m["window"] == w and keep(m) and m["split"] in ("calibration", "test")]
        sides = defaultdict(set)
        for m in ms:
            sides[m["event_id"]].add(m["split"])
        straddle = {e for e, s in sides.items() if len(s) > 1}
        test = [m for m in ms if m["split"] == "test"]
        test_events = {m["event_id"] for m in test}
        t_in = [m for m in test if m["event_id"] in straddle]
        keys_in = {m["moment_key"] for m in t_in}
        rows_test = [r for r in rows_all if r["window"] == w and r["split"] == "test"]
        rows_in = [r for r in rows_test if r["moment_key"] in keys_in]
        print("  window %s, %-30s %d moments in %d events; %d test moments in %d events, of which "
              "%d events straddle the split (%.1f%% of test events)"
              % (w, scope + ":", len(ms), len(sides), len(test), len(test_events), len(straddle),
                 100.0 * len(straddle) / max(len(test_events), 1)))
        print("      test moments inside a straddling event: %d of %d (%.1f%%)"
              % (len(t_in), len(test), 100.0 * len(t_in) / max(len(test), 1)))
        if scope == "primary set":
            print("      test ROWS (model x moment) inside a straddling event: %d of %d (%.1f%%)"
                  % (len(rows_in), len(rows_test), 100.0 * len(rows_in) / max(len(rows_test), 1)))

print("")
print("=== (b) the event-grouped split: sizes, and that it breaks no event")
for w in "ABC":
    ms = [m for m in MOMENTS.values() if m["window"] == w and not m["news_dateline_after_capture"]]
    old_cal = sum(1 for m in ms if m["split"] == "calibration")
    new_cal = sum(1 for m in ms if NEW.get(m["moment_key"]) == "calibration")
    sides = defaultdict(set)
    for m in ms:
        sides[m["event_id"]].add(NEW.get(m["moment_key"]))
    print("  window %s: %d moments; stored split calibration %d (%.1f%%), event-grouped calibration "
          "%d (%.1f%%); events broken by the event-grouped split: %d"
          % (w, len(ms), old_cal, 100.0 * old_cal / len(ms), new_cal, 100.0 * new_cal / len(ms),
             sum(1 for s in sides.values() if len(s) > 1)))

print("")
print("=== (b) single-split Platt recovery, stored split vs event-grouped split")
print("    'clean rows only' = the SAME map fitted on the stored calibration split, scored only on")
print("    test rows whose event has no calibration market, so the fit is unchanged and only the")
print("    leaky rows are dropped.")
for w in "ABC":
    print("  === window %s" % w)
    for mdl in MODELS:
        s = [r for r in rows_all if r["window"] == w and r["forecaster"] == mdl]
        if len(s) < 200:
            continue
        sides = defaultdict(set)
        for r in s:
            sides[r["event_id"]].add(r["split"])
        straddle = {e for e, v in sides.items() if len(v) > 1}
        cal = [r for r in s if r["split"] != "test"]
        test = [r for r in s if r["split"] == "test"]
        clean = [r for r in test if r["event_id"] not in straddle]
        ncal = [r for r in s if NEW.get(r["moment_key"]) == "calibration"]
        ntest = [r for r in s if NEW.get(r["moment_key"]) == "test"]
        print("    %-16s n=%-5d stored split cal %d / test %d; event-grouped cal %d / test %d; "
              "test rows in a straddling event %d"
              % (SHORT[mdl], len(s), len(cal), len(test), len(ncal), len(ntest), len(test) - len(clean)))
        for label, fit_on, score_on in (("stored split", cal, test),
                                        ("stored split, clean rows only", cal, clean),
                                        ("event-grouped split", ncal, ntest)):
            if len(fit_on) < 40 or len(score_on) < 50:
                print("      %-32s too few rows" % label)
                continue
            wts = platt_fit([r["p"] for r in fit_on], [r["outcome_yes"] for r in fit_on])
            got = recovery(score_on, platt_apply(wts, [r["p"] for r in score_on]))
            if got is None:
                print("      %-32s too few rows" % label)
                continue
            bm, bn, bq, rec = got
            print("      %-32s n=%-5d Brier %.4f -> %.4f (market %.4f)  recovery %6.1f%%  "
                  "slope %.3f intercept %+.3f" % (label, len(score_on), bm, bn, bq, 100 * rec, wts[1], wts[0]))
