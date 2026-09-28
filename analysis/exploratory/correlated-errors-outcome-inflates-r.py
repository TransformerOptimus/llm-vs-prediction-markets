import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-outcome-inflates-r: about a sixth of the headline pairwise error correlation is the
shared outcome term. Per window, on the common rows (every roster model forecast the row, primary set,
mode normal): the mean over model pairs of the Pearson correlation of the error (p - y), of raw p and of
the departure from the market (p - q); the gap r(error) - r(p - q); the same two measures inside each
price band and on sports and non-sports rows; and sports minus non-sports for each measure. Every number
carries a 1,000-resample event bootstrap (seed 20260902) that recomputes the statistic on the resample.
"""
import numpy as np
import correlated_errors_common as hc

BANDS = ["0.05-0.25", "0.25-0.50", "0.50-0.75", "0.75-0.95"]

data, cuts = hc.load("normal")
print("rows: common rows per window, forecasts view, mode normal, status in %s, primary set; mean over pairs of the Pearson "
      "correlation; intervals = %d event resamples, seed %d" % (hc.STATUS_OK, hc.N_BOOT, hc.SEED))
summary = {}
for w in hc.WINDOWS:
    t = hc.Table(data[w], hc.ROSTER[w])
    print("=== window %s: %d common rows, %d events, %d models, %d pairs" % (w, t.n, len(set(t.event)), t.k, len(t.pairs)))

    def core(idx, t=t):
        e, p, d = t.mean_pair_r("error", idx), t.mean_pair_r("p", idx), t.mean_pair_r("dev", idx)
        return np.array([e, p, d, e - d])

    point, lo, hi = hc.boot(t, core)
    summary[w] = (point, lo, hi)
    for j, lab in enumerate(("mean pair r(error)", "mean pair r(p)", "mean pair r(dev)", "r(error) - r(p-q)")):
        print("  %-18s %s" % (lab + ":", hc.fmt((point[j], lo[j], hi[j]))))
    for band in BANDS:
        sel = t.band == band
        if sel.sum() < 2:
            continue

        def in_band(idx, t=t, sel=sel):
            sub = idx[sel[idx]]
            return np.array([t.mean_pair_r("dev", sub), t.mean_pair_r("error", sub)])

        p_, l_, h_ = hc.boot(t, in_band)
        print("   band=%s n=%d r(p-q) %s  r(error) %s" % (band, sel.sum(), hc.fmt((p_[0], l_[0], h_[0])), hc.fmt((p_[1], l_[1], h_[1]))))
    for s in (0, 1):
        sel = t.sports == bool(s)

        def in_sports(idx, t=t, sel=sel):
            sub = idx[sel[idx]]
            return np.array([t.mean_pair_r("dev", sub), t.mean_pair_r("error", sub)])

        p_, l_, h_ = hc.boot(t, in_sports)
        print("   sports=%d n=%d r(p-q) %s  r(error) %s" % (s, sel.sum(), hc.fmt((p_[0], l_[0], h_[0])), hc.fmt((p_[1], l_[1], h_[1]))))

    def sports_gap(idx, t=t):
        sp, ns = idx[t.sports[idx]], idx[~t.sports[idx]]
        return np.array([t.mean_pair_r("dev", sp) - t.mean_pair_r("dev", ns), t.mean_pair_r("error", sp) - t.mean_pair_r("error", ns)])

    p_, l_, h_ = hc.boot(t, sports_gap)
    print("  sports minus non-sports, r(p-q): %s" % hc.fmt((p_[0], l_[0], h_[0])))
    print("  sports minus non-sports, r(error): %s" % hc.fmt((p_[1], l_[1], h_[1])))

if _paths.want_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for j, (col, lab) in enumerate(((0, "r(error) = corr(p - y)"), (1, "r(p)"), (2, "r(p - q)"))):
        xs = [i + (j - 1) * 0.25 for i in range(len(hc.WINDOWS))]
        v = [summary[w][0][col] for w in hc.WINDOWS]
        lo = [summary[w][0][col] - summary[w][1][col] for w in hc.WINDOWS]
        hi = [summary[w][2][col] - summary[w][0][col] for w in hc.WINDOWS]
        ax.errorbar(xs, v, yerr=[lo, hi], fmt="o", capsize=3, label=lab)
    ax.set_xticks(range(len(hc.WINDOWS))); ax.set_xticklabels(["Window %s" % w for w in hc.WINDOWS])
    ax.set_ylabel("mean pair correlation"); ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(_paths.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_paths.FIG_DIR, "correlated-errors-outcome-inflates-r.png"), dpi=150)
