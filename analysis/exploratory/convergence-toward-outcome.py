import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""convergence-toward-outcome: price drift toward the model is the market drifting toward the outcome.

Rows: disagreements (|p - q| >= 0.10) with a rated last in-window price, primary set, mode normal,
models pooled within a window (row = model x market, bootstrap clusters = events); per-model rows follow.
Quantities: share of disagreements whose last price moved toward the outcome by the 0.02 rule; the
toward-model rate when the model sat on the outcome's side of the market versus the other side, and the
difference (scoring.boot_diff, the two sides resampled independently); the share of disagreements on the
outcome's side. 1,000 event resamples, seed 20260902.
"""
from collections import defaultdict
import convergence_common as cc

moments, last, fc = cc.load_all("normal")
print("rows: forecasts view, mode normal, status in %s, primary set, |p-q| >= %.2f, rated last price in the %d-hour window"
      % (cc.STATUS_OK, cc.DISAGREE_MIN, int(cc.HOURS)))

pooled_by_window = {}
for w in cc.WINDOWS:
    print("=== window %s" % w)
    pooled = []
    for model in cc.ROSTER[w]:
        rows = cc.disagreement_rows(moments, last, fc, model, w)
        pooled += rows
        if not rows:
            print("   %-15s no rows" % cc.SHORT[model])
            continue
        on = [r for r in rows if r["on_outcome_side"]]
        off = [r for r in rows if not r["on_outcome_side"]]
        print("   %-15s n=%4d toward_outcome=%s on_outcome_side=%s | toward_model: right-side n=%d %s  wrong-side n=%d %s"
              % (cc.SHORT[model], len(rows), cc.ci_u(cc.boot_mean(rows, "toward_outcome")),
                 cc.ci_u(cc.boot_mean(rows, "on_outcome_side")),
                 len(on), cc.ci_u(cc.boot_mean(on, "toward_model")), len(off), cc.ci_u(cc.boot_mean(off, "toward_model"))))
    pooled_by_window[w] = pooled
    on = [r for r in pooled if r["on_outcome_side"]]
    off = [r for r in pooled if not r["on_outcome_side"]]
    events = len({r["event_id"] for r in pooled})
    print("%s POOLED n=%d events=%d toward_outcome=%s | right-side n=%d %s wrong-side n=%d %s diff=%s | share on outcome side=%.3f"
          % (w, len(pooled), events, cc.ci_u(cc.boot_mean(pooled, "toward_outcome")),
             len(on), cc.ci_u(cc.boot_mean(on, "toward_model")), len(off), cc.ci_u(cc.boot_mean(off, "toward_model")),
             cc.ci(cc.boot_diff(on, off, "toward_model")), cc.rate(pooled, "on_outcome_side")))
    for s in (1, 0):
        g = [r for r in pooled if r["settled"] == s]
        gon = [r for r in g if r["on_outcome_side"]]
        goff = [r for r in g if not r["on_outcome_side"]]
        print("   closed_in_window=%d n=%d toward_outcome=%.3f toward_model=%.3f | right-side n=%d %.3f  wrong-side n=%d %.3f"
              % (s, len(g), cc.rate(g, "toward_outcome"), cc.rate(g, "toward_model"),
                 len(gon), cc.rate(gon, "toward_model"), len(goff), cc.rate(goff, "toward_model")))

print("total disagreement rows: %d" % sum(len(v) for v in pooled_by_window.values()))

if _paths.want_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = range(len(cc.WINDOWS))
    on_r = [cc.rate([r for r in pooled_by_window[w] if r["on_outcome_side"]], "toward_model") for w in cc.WINDOWS]
    off_r = [cc.rate([r for r in pooled_by_window[w] if not r["on_outcome_side"]], "toward_model") for w in cc.WINDOWS]
    ax.bar([x - 0.2 for x in xs], on_r, width=0.4, label="model on outcome's side")
    ax.bar([x + 0.2 for x in xs], off_r, width=0.4, label="model on the other side")
    ax.set_xticks(list(xs)); ax.set_xticklabels(["Window %s" % w for w in cc.WINDOWS])
    ax.set_ylabel("toward-model rate"); ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(_paths.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_paths.FIG_DIR, "convergence-toward-outcome.png"), dpi=150)
