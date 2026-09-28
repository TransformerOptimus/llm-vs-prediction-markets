import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-pair-ranking: which model pairs agree most closely. Per window, on the common rows (every
roster model forecast the row, primary set, mode normal), the Pearson correlation of (p - q), of raw p and
of (p - y) for every pair, ranked by r(p - q). Contrasts, each with a 1,000-resample event bootstrap:
  the Kimi pair minus the mean of the reference pairs (pairs involving neither gpt-oss-120b nor two models
  of one family: 13 pairs in B and C), the DeepSeek pair minus the same reference, Kimi pair minus DeepSeek
  pair (B and C only, where both pairs exist), and the mean of the pairs containing gpt-oss-120b minus the
  reference pairs (A: 3 vs 3 pairs; B and C: 6 vs 13).
"""
import numpy as np
import correlated_errors_common as hc

GPT_OSS = "openrouter/openai/gpt-oss-120b"
KIMI = ("openrouter/moonshotai/kimi-k2.5", "openrouter/moonshotai/kimi-k2.6")
DEEPSEEK = ("openrouter/deepseek/deepseek-v3.2", "openrouter/deepseek/deepseek-v4-pro")


def family(m):
    return "kimi" if "kimi" in m else ("deepseek" if "deepseek" in m else m)


data, cuts = hc.load("normal")
print("rows: common rows per window (every roster model forecast the row), forecasts view, mode normal, status in %s, primary set; "
      "intervals = %d event resamples, seed %d" % (hc.STATUS_OK, hc.N_BOOT, hc.SEED))
tables = {}
for w in hc.WINDOWS:
    roster = hc.ROSTER[w]
    t = hc.Table(data[w], roster)
    tables[w] = t
    r = {k: t.pair_vals(k, np.arange(t.n)) for k in hc.KINDS}
    print("=== window %s: %d common rows, %d events, %d pairs" % (w, t.n, len(set(t.event)), len(t.pairs)))
    order = np.argsort(-r["dev"])
    for j in order:
        a, b = t.pairs[j]
        print("  %-16s %-16s r(p-q)=%.3f  r(p)=%.3f  r(error)=%.3f" % (t.short[a], t.short[b], r["dev"][j], r["p"][j], r["error"][j]))
    a, b = t.pairs[order[-1]]
    c, d = t.pairs[order[0]]
    print("  loosest pair: %s / %s r(p-q)=%.3f; tightest: %s / %s r(p-q)=%.3f"
          % (t.short[a], t.short[b], r["dev"][order[-1]], t.short[c], t.short[d], r["dev"][order[0]]))

    is_oss = np.array([GPT_OSS in (roster[a], roster[b]) for a, b in t.pairs])
    same_fam = np.array([family(roster[a]) == family(roster[b]) for a, b in t.pairs])
    ref = ~is_oss & ~same_fam
    kimi = np.array([{roster[a], roster[b]} == set(KIMI) for a, b in t.pairs])
    deep = np.array([{roster[a], roster[b]} == set(DEEPSEEK) for a, b in t.pairs])

    def contrasts(idx, t=t, ref=ref, is_oss=is_oss, kimi=kimi, deep=deep):
        v = t.pair_vals("dev", idx)
        out = [np.nanmean(v[is_oss]) - np.nanmean(v[ref])]
        if kimi.any() and deep.any():
            out += [v[kimi][0] - np.nanmean(v[ref]), v[deep][0] - np.nanmean(v[ref]), v[kimi][0] - v[deep][0]]
        return np.array(out)

    point, lo, hi = (np.atleast_1d(x) for x in hc.boot(t, contrasts))
    print("  gpt-oss pairs minus reference pairs, r(p-q): %s  (n pairs %d vs %d)" % (hc.fmt((point[0], lo[0], hi[0])), is_oss.sum(), ref.sum()))
    if point.size > 1:
        print("  kimi-k2.5/kimi-k2.6 minus reference pairs, r(p-q): %s" % hc.fmt((point[1], lo[1], hi[1])))
        print("  deepseek-v3.2/v4-pro minus reference pairs, r(p-q): %s" % hc.fmt((point[2], lo[2], hi[2])))
        print("  kimi pair minus deepseek pair: %s" % hc.fmt((point[3], lo[3], hi[3])))
        for kind in ("p", "error"):
            v = r[kind]
            print("  same contrasts on r(%s): kimi pair %.3f, deepseek pair %.3f, reference mean %.3f, gpt-oss pairs mean %.3f"
                  % ("p" if kind == "p" else "error", v[kimi][0], v[deep][0], np.nanmean(v[ref]), np.nanmean(v[is_oss])))

if _paths.want_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, w in zip(axes, hc.WINDOWS):
        t = tables[w]
        c = t.corr("dev", np.arange(t.n))
        im = ax.imshow(c, vmin=0.6, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(t.k)); ax.set_xticklabels(t.short, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(t.k)); ax.set_yticklabels(t.short, fontsize=8)
        ax.set_title("Window %s: r(p - q)" % w)
    fig.colorbar(im, ax=axes, shrink=0.7)
    os.makedirs(_paths.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_paths.FIG_DIR, "correlated-errors-pair-ranking.png"), dpi=150, bbox_inches="tight")
