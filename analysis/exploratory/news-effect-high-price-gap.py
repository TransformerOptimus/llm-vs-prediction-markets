"""On markets priced 0.75-0.95 for Yes, the no-news forecast sits near 0.5-0.57 (the No lean);
news pushes it up, and the push grows with the market price, but even with news the model closes
only a small part of its accuracy gap to the market."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, statistics as st
from news_effect_common import *
rows = load_pairs()
bars = {}
for w in "ABC":
    R = [r for r in rows if r["window"] == w]
    print("window", w)
    xs, mv, lo, hi = [], [], [], []
    for b in scoring.BAND_NAMES:
        rs = [r for r in R if r["band"] == b]
        m = boot_mean(rs, "move"); d = boot_mean(rs, "d_brier"); tm = boot_mean(rs, "move_toward_market")
        bp, bn, bm = st.fmean(r["brier_probe"] for r in rs), st.fmean(r["brier_normal"] for r in rs), st.fmean(r["brier_market"] for r in rs)
        closed = (bp - bn) / (bp - bm) if bp != bm else float("nan")
        print("  band %s n=%d markets=%d | mean p no-news %.3f with-news %.3f q %.3f outcome %.3f | signed move %s | toward-market %s | d_brier %s | Brier no-news %.4f with-news %.4f market %.4f | share of gap to market closed by news %.0f%%" % (
            b, len(rs), len({r["row_id"] for r in rs}), st.fmean(r["p_probe"] for r in rs), st.fmean(r["p_normal"] for r in rs), st.fmean(r["q"] for r in rs), st.fmean(r["y"] for r in rs), fmt(m), fmt(tm), fmt(d), bp, bn, bm, 100 * closed))
        xs.append(b); mv.append(m[0]); lo.append(m[0] - m[1]); hi.append(m[2] - m[0])
    bars[w] = (xs, mv, lo, hi)
if _paths.want_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, w in zip(axes, "ABC"):
        xs, mv, lo, hi = bars[w]
        ax.bar(xs, mv, yerr=[lo, hi], capsize=3, color="#47a")
        ax.axhline(0, color="k", lw=0.8); ax.set_title("Window %s" % w, fontsize=9); ax.set_xlabel("market price band (Yes)", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("with-news minus no-news probability\n(positive = news pushed toward Yes)", fontsize=8)
    fig.suptitle("News pushes the model toward Yes, and the push grows with the market price", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(_paths.FIG_DIR, "news-effect-high-price-gap.png"), dpi=150)
