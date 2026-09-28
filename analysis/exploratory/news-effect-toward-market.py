"""News helps only when it pulls the model toward the market price; when news pushes the model
away from the market, accuracy falls (B, C) or does not improve (A).
Pairs: same model, same market, with news (normal) vs no news (memory probe). Primary set, news rows.
d_brier = Brier(no news) - Brier(with news), positive = news helped.
'toward' = |p_normal - q| < |p_probe - q|; 'away' = the reverse; rows moved under 0.02 are 'no move'."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, statistics as st
from news_effect_common import *
rows = load_pairs()
res = {}
for w in "ABC":
    R = [r for r in rows if r["window"] == w]
    tw = [r for r in R if r["abs_move"] >= 0.02 and r["move_toward_market"] > 0]
    aw = [r for r in R if r["abs_move"] >= 0.02 and r["move_toward_market"] <= 0]
    nm = [r for r in R if r["abs_move"] < 0.02]
    t, a = boot_mean(tw, "d_brier"), boot_mean(aw, "d_brier")
    d = boot_diff(tw, aw, "d_brier")
    res[w] = (t, a, d, len(tw), len(aw))
    print("window %s: n=%d (markets %d) | toward market n=%d d_brier %s share_helped %.2f | away n=%d d_brier %s share_helped %.2f | no move n=%d %s | toward minus away %s" % (
        w, len(R), len({r["row_id"] for r in R}), len(tw), fmt(t), st.fmean(r["d_brier"] > 0 for r in tw), len(aw), fmt(a), st.fmean(r["d_brier"] > 0 for r in aw), len(nm), fmt(boot_mean(nm, "d_brier")), fmt(d)))
    for mdl in sorted({r["model"] for r in R}):
        mt = [r for r in tw if r["model"] == mdl]; ma = [r for r in aw if r["model"] == mdl]
        print("   %-16s toward n=%4d %s | away n=%4d %s | diff %s" % (mdl, len(mt), fmt(boot_mean(mt, "d_brier")), len(ma), fmt(boot_mean(ma, "d_brier")), fmt(boot_diff(mt, ma, "d_brier"))))
    T = [r for r in R if r["split"] == "test"]
    tt = [r for r in T if r["abs_move"] >= 0.02 and r["move_toward_market"] > 0]; ta = [r for r in T if r["abs_move"] >= 0.02 and r["move_toward_market"] <= 0]
    print("   test split only: toward n=%d %s | away n=%d %s" % (len(tt), fmt(boot_mean(tt, "d_brier")), len(ta), fmt(boot_mean(ta, "d_brier"))))
if _paths.want_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(3)
    for i, (lab, col, idx) in enumerate([("news pulled model toward market", "#2a7", 0), ("news pushed model away from market", "#c44", 1)]):
        ys = [res[w][idx][0] for w in "ABC"]; lo = [res[w][idx][0] - res[w][idx][1] for w in "ABC"]; hi = [res[w][idx][2] - res[w][idx][0] for w in "ABC"]
        ax.bar([xx + (i - 0.5) * 0.36 for xx in x], ys, 0.34, yerr=[lo, hi], color=col, capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(list(x)); ax.set_xticklabels(["Window A (Polymarket, PolyBench)", "Window B (Polymarket, 2026)", "Window C (Kalshi, 2026)"], fontsize=8)
    ax.set_ylabel("Brier(no news) - Brier(with news)\n(positive = news helped)"); ax.legend(fontsize=8); ax.set_title("Same market, same model: news helps only when it moves the model toward the price", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(_paths.FIG_DIR, "news-effect-toward-market.png"), dpi=150)
