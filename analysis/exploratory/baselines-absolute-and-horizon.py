import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""baselines-absolute-and-horizon: what the scores look like in absolute terms, against naive
forecasters, and by how far ahead the forecast was made.

The gap to the market shows only that the models are worse than the market, not whether they are any
good at all. A forecaster can be far behind a market and still be well ahead of anything naive, so
this script reports the raw scores as well.

Three parts:

  1. ABSOLUTE. Per model and window: Brier, the market's Brier, the outcome variance (the Brier of a
     forecaster who always says the base rate, and the natural zero point for "no skill"), and the
     skill score against that base-rate forecaster, which is the share of the naive error removed.
     A positive skill score means the model beats always-guessing-the-base-rate.

  2. BASELINES. Four naive forecasters scored on the same rows:
       base rate      always the window's share of Yes outcomes, which is the best constant forecast
       always 0.5     the uninformed coin flip
       market         the price itself, for reference
       shrunk market  the price pulled toward the base rate by the average factor the models use
                      (see correlated-errors-shrinkage-null). This is the null the shrinkage
                      reading predicts a model should look like: if the models were simply
                      an honestly-shrunk market price, this row would match them.
     The base rate is computed on the same rows being scored, which flatters it slightly; it is a
     lower bound on the model's advantage, not an upper one.

  3. HORIZON. The same numbers split by how long before resolution the forecast was made, using the
     benchmark's own horizon bands. Time to resolution is the most likely moderator of both market
     accuracy and the model's disadvantage.

Intervals are 1,000 event resamples, seed 20260902. Test split and both splits are both reported.
"""
from collections import defaultdict

from calibration_shape_common import MODELS, SHORT, boot_stat, load, murphy

WINDOWS = ("A", "B", "C")


def brier(rows, f):
    return sum((f(r) - r["outcome_yes"]) ** 2 for r in rows) / len(rows)


def skill(rows, f):
    """Share of the base-rate forecaster's error removed. Positive means better than the base rate."""
    base = sum(r["outcome_yes"] for r in rows) / len(rows)
    ref = brier(rows, lambda r: base)
    return (ref - brier(rows, f)) / ref if ref > 0 else float("nan")


def split_sets(rows):
    return (("test split", [r for r in rows if r["split"] == "test"]), ("both splits", rows))


rows_all, cuts = load("normal")
print("rows: primary set, forecasts view, mode normal; UNC = outcome variance = the Brier of a "
      "forecaster who always says the base rate; skill = share of that error removed, positive is "
      "better than the base rate; intervals = 1,000 event resamples, seed 20260902")

print()
print("### 1. ABSOLUTE SCORES")
for w in WINDOWS:
    for which, rs in split_sets([r for r in rows_all if r["window"] == w]):
        base = sum(r["outcome_yes"] for r in rs) / len(rs) if rs else float("nan")
        print("== window %s, %s: Yes base rate %.4f" % (w, which, base))
        for mdl in MODELS:
            s = [r for r in rs if r["forecaster"] == mdl]
            if len(s) < 50:
                continue
            bm = brier(s, lambda r: r["p"])
            bq = brier(s, lambda r: r["q"])
            _, _, unc = murphy([r["p"] for r in s], [r["outcome_yes"] for r in s])
            sk = boot_stat(s, lambda rs_: skill(rs_, lambda r: r["p"]))
            skq = skill(s, lambda r: r["q"])
            print("  %-16s n=%-5d Brier %.4f  market %.4f  UNC %.4f | skill vs base rate "
                  "%+.4f [%+.4f, %+.4f]  market skill %+.4f"
                  % (SHORT[mdl], len(s), bm, bq, unc, *sk, skq))

print()
print("### 2. NAIVE BASELINES on the same rows (one line per window, all models pooled where relevant)")
for w in WINDOWS:
    for which, rs in split_sets([r for r in rows_all if r["window"] == w]):
        if not rs:
            continue
        base = sum(r["outcome_yes"] for r in rs) / len(rs)
        # The average shrinkage the roster applies, refitted here on these rows.
        qs = [r["q"] for r in rs]
        ps = [r["p"] for r in rs]
        qbar = sum(qs) / len(qs)
        var = sum((q - qbar) ** 2 for q in qs)
        b = sum((q - qbar) * (p - sum(ps) / len(ps)) for q, p in zip(qs, ps)) / var if var else 1.0
        a = sum(ps) / len(ps) - b * qbar
        print("== window %s, %s: %d scored rows, base rate %.4f, roster mean shrinkage b %.4f a %+.4f"
              % (w, which, len(rs), base, b, a))
        for name, f in (("base rate", lambda r: base),
                        ("always 0.5", lambda r: 0.5),
                        ("market", lambda r: r["q"]),
                        ("shrunk market", lambda r: min(max(a + b * r["q"], 0.01), 0.99)),
                        ("roster average model", lambda r: r["p"])):
            print("  %-22s Brier %.4f  skill vs base rate %+.4f" % (name, brier(rs, f), skill(rs, f)))

print()
print("### 3. BY FORECAST HORIZON (benchmark horizon bands)")
for w in WINDOWS:
    for which, rs in split_sets([r for r in rows_all if r["window"] == w]):
        bands = sorted({r["horizon_band"] for r in rs if r["horizon_band"]})
        if not bands:
            continue
        print("== window %s, %s" % (w, which))
        for band in bands:
            sel = [r for r in rs if r["horizon_band"] == band]
            if len(sel) < 50:
                continue
            days = sorted(r["horizon_days"] for r in sel if r["horizon_days"] is not None)
            med = days[len(days) // 2] if days else float("nan")
            bm = brier(sel, lambda r: r["p"])
            bq = brier(sel, lambda r: r["q"])
            gap = boot_stat(sel, lambda rs_: brier(rs_, lambda r: r["q"]) - brier(rs_, lambda r: r["p"]))
            res_p = murphy([r["p"] for r in sel], [r["outcome_yes"] for r in sel])[1]
            res_q = murphy([r["q"] for r in sel], [r["outcome_yes"] for r in sel])[1]
            print("  %-14s n=%-6d median %4.1f days | model Brier %.4f RES %.4f | market %.4f RES %.4f "
                  "| edge %+.4f [%+.4f, %+.4f]"
                  % (band, len(sel), med, bm, res_p, bq, res_q, *gap))

# ---------------------------------------------------------------------------
# The full horizon table behind the statement that the deficit "does not depend cleanly on horizon".
# Section 3 above pools the models; this block adds the same breakdown per model, plus a one-line
# ordering per window.
# ---------------------------------------------------------------------------
BAND_ORDER = ["<1d", "1-3d", "3-7d", ">7d"]
print()
print("### 4. FULL HORIZON BREAKDOWN, PER MODEL")
print("Brier edge = market Brier minus model Brier on the same rows; negative means the market is "
      "ahead. Bands are the benchmark's own: under a day, 1 to 3 days, 3 to 7 days, over 7 days. "
      "Intervals = 1,000 event resamples, seed 20260902.")
for w in WINDOWS:
    for which, rs in split_sets([r for r in rows_all if r["window"] == w]):
        bands = [b for b in BAND_ORDER if any(r["horizon_band"] == b for r in rs)]
        if not bands:
            continue
        print("== window %s, %s" % (w, which))
        pooled = {}
        for band in bands:
            sel = [r for r in rs if r["horizon_band"] == band]
            if len(sel) < 50:
                continue
            pooled[band] = boot_stat(sel, lambda rs_: brier(rs_, lambda r: r["q"]) - brier(rs_, lambda r: r["p"]))
        print("   all models pooled: %s"
              % "  ".join("%s %+.4f [%+.4f, %+.4f]" % (b, *pooled[b]) for b in bands if b in pooled))
        order = sorted(pooled, key=lambda b: pooled[b][0])
        if order:
            print("   worst band %s (%+.4f), best band %s (%+.4f); the bands in order from worst to "
                  "best: %s" % (order[0], pooled[order[0]][0], order[-1], pooled[order[-1]][0],
                                " < ".join(order)))
        for mdl in MODELS:
            cells = []
            for band in bands:
                sel = [r for r in rs if r["horizon_band"] == band and r["forecaster"] == mdl]
                if len(sel) < 50:
                    cells.append("%-6s n=%-5s %-26s" % (band, len(sel), "too few rows"))
                    continue
                g = boot_stat(sel, lambda rs_: brier(rs_, lambda r: r["q"]) - brier(rs_, lambda r: r["p"]))
                cells.append("%-6s n=%-5d %+.4f [%+.4f, %+.4f]" % (band, len(sel), *g))
            if cells:
                print("   %-16s %s" % (SHORT[mdl], " | ".join(cells)))
