import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""convergence-open-market-move: on disagreements whose market was still open at the end of the 72-hour
window, the price moved a few cents in the model's direction, and a "shrink toward 0.5" direction earns
less.

Rows: disagreements (|p - q| >= 0.10) with a rated last price, primary set, mode normal, models pooled within a window,
split into markets still open at the window end and markets that closed inside it. Signed moves per row:
  toward model    (last - q) x sign(p - q)
  toward 0.5      (last - q) x sign(0.5 - q)        (zero when q is exactly 0.5)
  toward outcome  (last - q) x sign(y - q)
Means with 95% intervals from 1,000 event resamples (scoring.boot_mean, seed 20260902); the open rows also
split by whether the model was more extreme than the market (its disagreement points away from 0.5:
(p - q)(0.5 - q) <= 0) or nearer 0.5, and per model.
"""
import convergence_common as cc

moments, last, fc = cc.load_all("normal")
print("rows: forecasts view, mode normal, status in %s, primary set, |p-q| >= %.2f, rated last price; open = close after forecast + %dh; "
      "intervals = %d event resamples, seed %d" % (cc.STATUS_OK, cc.DISAGREE_MIN, int(cc.HOURS), cc.N_BOOT, cc.SEED))

summary = {}
for w in cc.WINDOWS:
    pooled = cc.pooled_disagreements(moments, last, fc, w)
    op = [r for r in pooled if not r["settled"]]
    se = [r for r in pooled if r["settled"]]
    mm = cc.boot_mean(op, "move_to_model")
    summary[w] = mm
    print("%s open-at-window-end rows n=%d events=%d of %d disagreements (%.0f%%); settled rows n=%d, their move to model=%s"
          % (w, len(op), len({r["event_id"] for r in op}), len(pooled), 100.0 * len(op) / len(pooled), len(se),
             cc.ci(cc.boot_mean(se, "move_to_model"), 4)))
    print("   mean move toward model   %s" % cc.ci(mm, 4))
    print("   mean move toward 0.5     %s" % cc.ci(cc.boot_mean(op, "move_to_half"), 4))
    print("   mean move toward outcome %s" % cc.ci(cc.boot_mean(op, "move_to_outcome"), 4))
    ext = [r for r in op if r["model_more_extreme"]]
    near = [r for r in op if not r["model_more_extreme"]]
    print("   model more extreme than market n=%d move to model %s; model nearer 0.5 n=%d move to model %s"
          % (len(ext), cc.ci(cc.boot_mean(ext, "move_to_model"), 4), len(near), cc.ci(cc.boot_mean(near, "move_to_model"), 4)))
    clear = 0
    for model in cc.ROSTER[w]:
        g = [r for r in op if r["model"] == model]
        if not g:
            continue
        t = cc.boot_mean(g, "move_to_model")
        clear += int(t[1] > 0)
        print("   %-15s n=%d move to model %s" % (cc.SHORT[model], len(g), cc.ci(t, 4)))
    print("   models whose interval clears zero: %d of %d" % (clear, len(cc.ROSTER[w])))

if _paths.want_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    v = [summary[w][0] for w in cc.WINDOWS]
    lo = [summary[w][0] - summary[w][1] for w in cc.WINDOWS]
    hi = [summary[w][2] - summary[w][0] for w in cc.WINDOWS]
    ax.errorbar(range(len(cc.WINDOWS)), v, yerr=[lo, hi], fmt="o", capsize=3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(cc.WINDOWS))); ax.set_xticklabels(["Window %s" % w for w in cc.WINDOWS])
    ax.set_ylabel("mean price move toward the model (open markets)")
    fig.tight_layout()
    os.makedirs(_paths.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_paths.FIG_DIR, "convergence-open-market-move.png"), dpi=150)
