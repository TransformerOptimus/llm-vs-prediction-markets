"""Per row, news is a coin flip: it hurts on about as many model-rows as it helps, the median
effect is zero, and the small positive mean comes from a few rows with very large moves."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, statistics as st
from news_effect_common import *
rows = load_pairs()
shares = {}
for w in "ABC":
    R = [r for r in rows if r["window"] == w]
    helped = [r for r in R if r["d_brier"] > 0]; hurt = [r for r in R if r["d_brier"] < 0]
    tot = sum(r["d_brier"] for r in R)
    big = [r for r in R if r["abs_move"] > 0.25]
    print("window %s n=%d markets=%d | helped %.1f%% hurt %.1f%% unchanged %.1f%% | median d_brier %.4f mean %s | mean gain when helped %.4f, mean loss when hurt %.4f | rows moved >0.25: %d (%.1f%%) carry %.0f%% of the total gain, their d_brier %s | rows moved <0.10: n=%d d_brier %s" % (
        w, len(R), len({r["row_id"] for r in R}), 100 * len(helped) / len(R), 100 * len(hurt) / len(R), 100 * (len(R) - len(helped) - len(hurt)) / len(R), st.median(r["d_brier"] for r in R), fmt(boot_mean(R, "d_brier")),
        st.fmean(r["d_brier"] for r in helped), st.fmean(r["d_brier"] for r in hurt), len(big), 100 * len(big) / len(R), 100 * sum(r["d_brier"] for r in big) / tot, fmt(boot_mean(big, "d_brier")),
        sum(1 for r in R if r["abs_move"] < 0.10), fmt(boot_mean([r for r in R if r["abs_move"] < 0.10], "d_brier"))))
    shares[w] = (100 * len(helped) / len(R), 100 * len(hurt) / len(R))
if _paths.want_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, w in zip(axes, "ABC"):
        R = [r for r in rows if r["window"] == w]
        ax.hist([r["d_brier"] for r in R], bins=60, range=(-0.6, 0.6), color="#47a"); ax.axvline(0, color="k", lw=0.8)
        ax.set_title("Window %s: helped %.0f%%, hurt %.0f%%" % (w, *shares[w]), fontsize=9); ax.set_xlabel("Brier(no news) - Brier(with news)", fontsize=8); ax.set_yscale("log")
    axes[0].set_ylabel("model-rows (log scale)", fontsize=8)
    fig.suptitle("Per row, news is a coin flip; the mean gain comes from a thin tail of large moves", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(_paths.FIG_DIR, "news-effect-coin-flip.png"), dpi=150)
