import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-both-wrong-beyond-market: most shared mistakes happen when the crowd is also wrong,
but on rows where the market is right, model pairs still both miss several times as often as chance.

Per window, on the common rows (every roster model forecast the row, primary set, mode normal), with
wrong = on the wrong side of 0.5 for models and for the market:
  both-wrong rate      mean over model pairs of the share of rows where both models are wrong
  expected             mean over pairs of the product of the two models' wrong rates inside the same
                       stratum (independence)
  on all rows, on market-wrong rows and on market-right rows; the excess and ratio over independence on
  market-right rows; the share of both-wrong pair-events that fall on market-wrong rows; the share of
  market-right rows where every roster model is wrong. Intervals: 1,000 event resamples of the common
  rows (seed 20260902), every statistic recomputed on the resample.
"""
import numpy as np
import correlated_errors_common as hc

data, cuts = hc.load("normal")
print("rows: common rows per window, forecasts view, mode normal, status in %s, primary set; wrong = wrong side of 0.5; "
      "intervals = %d event resamples, seed %d" % (hc.STATUS_OK, hc.N_BOOT, hc.SEED))
summary = {}
for w in hc.WINDOWS:
    t = hc.Table(data[w], hc.ROSTER[w])
    W = t.wrong.astype(float)
    iu = t.iu

    def pair_rates(rows):
        """(mean over pairs of the both-wrong rate, mean over pairs of the independence product) on `rows`."""
        n = len(rows)
        if n == 0:
            return float("nan"), float("nan")
        Wb = W[rows]
        both = (Wb.T @ Wb) / n
        m = Wb.mean(axis=0)
        return float(both[iu].mean()), float(np.outer(m, m)[iu].mean())

    def stats(idx):
        mw = t.mkt_wrong[idx]
        right, wrong = idx[~mw], idx[mw]
        bw_all, ex_all = pair_rates(idx)
        bw_mw, ex_mw = pair_rates(wrong)
        bw_mr, ex_mr = pair_rates(right)
        cnt_all = (W[idx].T @ W[idx])[iu].sum()
        cnt_mw = (W[wrong].T @ W[wrong])[iu].sum() if len(wrong) else 0.0
        share_mw = cnt_mw / cnt_all if cnt_all else float("nan")
        every = float(t.wrong[right].all(axis=1).mean()) if len(right) else float("nan")
        return np.array([bw_all, ex_all, bw_mw, ex_mw, bw_mr, ex_mr, bw_mr - ex_mr,
                         bw_mr / ex_mr if ex_mr else float("nan"), share_mw, every, mw.mean()])

    point, lo, hi = hc.boot(t, stats)
    summary[w] = (point, lo, hi)
    n_mw = int(t.mkt_wrong.sum())
    print("=== window %s: %d common rows, %d events, %d models, %d pairs; market wrong on %d rows (%.1f%%)"
          % (w, t.n, len(set(t.event)), t.k, len(t.pairs), n_mw, 100.0 * point[10]))
    print("  all rows:            both_wrong %.3f  expected %.3f" % (point[0], point[1]))
    print("  market-wrong rows:   both_wrong %.3f  expected %.3f  (n=%d)" % (point[2], point[3], n_mw))
    print("  market-right rows:   both_wrong %s  expected %s  (n=%d)"
          % (hc.fmt((point[4], lo[4], hi[4])), hc.fmt((point[5], lo[5], hi[5])), t.n - n_mw))
    print("  excess over independence on market-right rows: %s ; ratio %s"
          % (hc.fmt((point[6], lo[6], hi[6])), hc.fmt((point[7], lo[7], hi[7]))))
    print("  share of both-wrong pair-events on market-wrong rows: %s" % hc.fmt((point[8], lo[8], hi[8])))
    print("  every model wrong while market right: %s (n market-right = %d)" % (hc.fmt((point[9], lo[9], hi[9])), t.n - n_mw))

if _paths.want_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    xs = range(len(hc.WINDOWS))
    ax.bar([x - 0.2 for x in xs], [summary[w][0][4] for w in hc.WINDOWS], width=0.4, label="both wrong, market right")
    ax.bar([x + 0.2 for x in xs], [summary[w][0][5] for w in hc.WINDOWS], width=0.4, label="expected under independence")
    ax.set_xticks(list(xs)); ax.set_xticklabels(["Window %s" % w for w in hc.WINDOWS])
    ax.set_ylabel("rate on market-right rows"); ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(_paths.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_paths.FIG_DIR, "correlated-errors-both-wrong-beyond-market.png"), dpi=150)
