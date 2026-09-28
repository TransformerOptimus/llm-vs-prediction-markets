import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""direction-same-side-share: how much of "leaning against the crowd" is really disagreement?

A row counts as leaning against the crowd when the forecast sits on the far side of the price from
the side the market favours. That label covers two quite different situations:

  same side, less confident   market 0.70 (favouring Yes), model 0.60. The model still thinks Yes is
                              the more likely outcome. It differs from the crowd in confidence, not
                              in which way the question will go.
  opposite side               market 0.70, model 0.40. The model favours the other outcome.

Only the second is disagreement in the ordinary sense. The distinction matters here because every
model shrinks its forecasts toward the base rate (see correlated-errors-shrinkage-null), and a
forecaster that shrinks lands between the base rate and the price on any market priced above it --
same side, less confident -- by construction rather than by judgement. So an unknown part of the
"against the crowd" group may be a restatement of the shrinkage.

This script measures that part. Per window and model, over the primary set: of the rows that lean
against the crowd, the share where the model and the market still favour the same outcome (both
above 0.5 or both below), and the share where they favour opposite outcomes. It also reports the
Brier edge within each of those two groups, since if the losses sit mainly in the opposite-side rows
then "against the crowd" is doing real work; if they sit in the same-side rows it is largely
shrinkage.

Test split and both splits are both reported. Intervals are 1,000 event resamples, seed 20260902.
"""
from calibration_shape_common import MODELS, SHORT, boot_stat, load

WINDOWS = ("A", "B", "C")


def same_side(r):
    """True when forecast and price favour the same outcome, both being off even money."""
    return (r["p"] >= 0.5) == (r["q"] >= 0.5)


def edge(rows):
    return sum(r["brier_edge"] for r in rows) / len(rows) if rows else float("nan")


rows_all, cuts = load("normal")
print("rows: primary set, forecasts view, mode normal, rows whose direction is 'against' the crowd; "
      "same side = forecast and price favour the same outcome (both at or above 0.5, or both below); "
      "intervals = 1,000 event resamples, seed 20260902")

for w in WINDOWS:
    for which, keep in (("test split", lambda r: r["split"] == "test"), ("both splits", lambda r: True)):
        base = [r for r in rows_all if r["window"] == w and keep(r) and r["direction"] == "against"]
        if len(base) < 100:
            continue
        pooled_same = [r for r in base if same_side(r)]
        print("== window %s, %s: %d rows leaning against the crowd" % (w, which, len(base)))
        share = boot_stat(base, lambda rs: sum(1 for r in rs if same_side(r)) / len(rs))
        print("   same side, less confident: %d of %d = %.1f%% [%.1f%%, %.1f%%]  Brier edge %+.4f"
              % (len(pooled_same), len(base), 100 * share[0], 100 * share[1], 100 * share[2],
                 edge(pooled_same)))
        opp = [r for r in base if not same_side(r)]
        print("   opposite side:             %d of %d = %.1f%%                  Brier edge %+.4f"
              % (len(opp), len(base), 100 * len(opp) / len(base), edge(opp)))
        for mdl in MODELS:
            s = [r for r in base if r["forecaster"] == mdl]
            if len(s) < 30:
                continue
            ss = [r for r in s if same_side(r)]
            print("     %-16s against n=%-5d same side %5.1f%%  edge same %+.4f  edge opposite %+.4f"
                  % (SHORT[mdl], len(s), 100 * len(ss) / len(s), edge(ss),
                     edge([r for r in s if not same_side(r)])))
