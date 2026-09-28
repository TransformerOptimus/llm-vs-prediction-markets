import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""convergence-settled-in-window: on disagreements whose market closed inside the 72-hour window the
toward-model test reduces to "the model was on the outcome's side and at least halfway from the market to the outcome",
and the excess over the shuffled baseline has opposite signs on settled and still-open rows.

Rows: disagreements (|p - q| >= 0.10) with a rated last price, primary set, mode normal, models pooled within a window,
split by whether the close fell inside the window (settled) or not (open). Per window and split:
  the share of rows with the last price at 0 or 1 (within 0.01) and, on those rows, how often the
  toward-model verdict equals the halfway identity; the toward-outcome and toward-model shares; the excess of the
  toward-model rate over the disagreement-only shuffled baseline (p permuted across the split's
  disagreement rows, 1,000 draws, random.Random(20260902), the test re-applied after each draw), with a
  1,000-resample event bootstrap that recomputes both rates; the share of rows with the model on the
  outcome's side, and on the outcome's side and at least halfway. Per-model rows follow each window.
"""
import convergence_common as cc

moments, last, fc = cc.load_all("normal")
print("rows: forecasts view, mode normal, status in %s, primary set, |p-q| >= %.2f, rated last price; settled = close <= forecast + %dh; "
      "baseline = %d permutations of p across the split's disagreement rows; intervals = %d event resamples, seed %d"
      % (cc.STATUS_OK, cc.DISAGREE_MIN, int(cc.HOURS), cc.N_PERM, cc.N_BOOT, cc.SEED))

summary = {}
for w in cc.WINDOWS:
    pooled = cc.pooled_disagreements(moments, last, fc, w)
    n_settled = sum(r["settled"] for r in pooled)
    print("=== window %s: %d disagreements, %d settled inside the window (%.1f%%), %d open"
          % (w, len(pooled), n_settled, 100.0 * n_settled / len(pooled), len(pooled) - n_settled))
    for s in (1, 0):
        g = [r for r in pooled if r["settled"] == s]
        ext = [r for r in g if r["last_at_extreme"]]
        ident = cc.rate([{"v": int(r["toward_model"] == r["right_and_halfway"])} for r in ext], "v")
        exc = cc.boot_excess(g)
        summary[(w, s)] = exc
        print("%s settled_in_window=%d n=%d events=%d last_price_at_0_or_1=%.3f (n=%d) identity_holds_on_those=%.3f | "
              "toward_outcome=%s toward_model=%s shuffled=%.3f excess_over_shuffle=%s | model right side=%.3f, right side and >= halfway=%.3f"
              % (w, s, len(g), len({r["event_id"] for r in g}), cc.rate(g, "last_at_extreme"), len(ext), ident,
                 cc.ci_u(cc.boot_mean(g, "toward_outcome")), cc.ci_u(cc.boot_mean(g, "toward_model")), exc[4],
                 cc.ci((exc[0], exc[1], exc[2])), cc.rate(g, "on_outcome_side"), cc.rate(g, "right_and_halfway")))
    for model in cc.ROSTER[w]:
        for s in (1, 0):
            g = [r for r in pooled if r["model"] == model and r["settled"] == s]
            if not g:
                continue
            print("   %s %-15s settled=%d n=%d toward_model=%.3f toward_outcome=%.3f right_side=%.3f halfway=%.3f"
                  % (w, cc.SHORT[model], s, len(g), cc.rate(g, "toward_model"), cc.rate(g, "toward_outcome"),
                     cc.rate(g, "on_outcome_side"), cc.rate(g, "right_and_halfway")))

print("excess over the disagreement-only baseline, settled minus open:")
for w in cc.WINDOWS:
    print("   %s settled %s  open %s  gap %+.3f" % (w, cc.ci(summary[(w, 1)][:3]), cc.ci(summary[(w, 0)][:3]),
                                                  summary[(w, 1)][0] - summary[(w, 0)][0]))

if _paths.want_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k, (s, lab) in enumerate(((1, "settled inside the window"), (0, "still open"))):
        xs = [i + (k - 0.5) * 0.3 for i in range(len(cc.WINDOWS))]
        v = [summary[(w, s)][0] for w in cc.WINDOWS]
        lo = [summary[(w, s)][0] - summary[(w, s)][1] for w in cc.WINDOWS]
        hi = [summary[(w, s)][2] - summary[(w, s)][0] for w in cc.WINDOWS]
        ax.errorbar(xs, v, yerr=[lo, hi], fmt="o", capsize=3, label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(cc.WINDOWS))); ax.set_xticklabels(["Window %s" % w for w in cc.WINDOWS])
    ax.set_ylabel("toward-model rate minus shuffled baseline"); ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(_paths.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_paths.FIG_DIR, "convergence-settled-in-window.png"), dpi=150)
