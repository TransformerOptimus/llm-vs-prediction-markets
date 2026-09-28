"""Hypothesis: the extra accuracy gap on Kalshi (Window C) versus Polymarket (Window B) is the
question mix, not the venue. Split both windows into 'rows with a news snapshot and a horizon of
at least one day' and 'the rest' (no news, or same-day). The same models on both windows.
Prints every number behind that statement; saves the figure. Read-only on both databases."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vw_base

M, cuts, rows = vw_base.load()
likeB = lambda r: r["news_available"] == 1 and r["horizon_band"] != "<1d"
groups = [("news and horizon >= 1 day", likeB), ("no news or same-day", lambda r: not likeB(r))]
res = {}
print("Brier edge = market Brier - model Brier (negative = model worse). C - B difference, event bootstrap %d draws, seed %d" % (vw_base.N_BOOT, vw_base.SEED))
for lab, f in groups:
    print("== rows: %s" % lab)
    for sp_lab, spf in [("all", lambda r: True), ("sports only", lambda r: r["sports"] == 1), ("non-sports only", lambda r: r["sports"] == 0)]:
        for mdl in vw_base.MODELS:
            c = [r for r in rows if r["window"] == "C" and r["model"] == mdl and f(r) and spf(r)]
            b = [r for r in rows if r["window"] == "B" and r["model"] == mdl and f(r) and spf(r)]
            d = vw_base.bd(c, b, "brier_edge")
            ec, eb = vw_base.bm(c, "brier_edge"), vw_base.bm(b, "brier_edge")
            res[(lab, sp_lab, mdl)] = (d, len(c), len(b))
            print("  %-12s %-16s C-B %+.4f [%+.4f,%+.4f]  edge C %+.4f [%+.4f,%+.4f] (n=%d)  edge B %+.4f [%+.4f,%+.4f] (n=%d)"
                  % (sp_lab, vw_base.short(mdl), *d, *ec, len(c), *eb, len(b)))
        c = [r for r in rows if r["window"] == "C" and f(r) and spf(r)]; b = [r for r in rows if r["window"] == "B" and f(r) and spf(r)]
        print("  %-12s %-16s C-B %+.4f [%+.4f,%+.4f] (n C=%d, B=%d) [pooled over models; one obs per model-row]" % (sp_lab, "POOLED", *vw_base.bd(c, b, "brier_edge"), len(c), len(b)))
# composition
for w in "BC":
    for lab, f in groups:
        s = [r for r in rows if r["window"] == w and f(r) and r["model"] == "openai/gpt-5.2"]
        print("composition %s / %s: markets=%d sports share=%.2f median horizon days=%.2f" % (w, lab, len(s), st.fmean(r["sports"] for r in s), st.median(r["horizon_days"] for r in s)))

if _paths.want_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # figure: dot + interval per model, two panels (sports only, since sports mix is the cleaner comparison; and all rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    cols = {"news and horizon >= 1 day": "#2a78d6", "no news or same-day": "#eb6834"}
    for ax, sp_lab in zip(axes, ["sports only", "all"]):
        for gi, (lab, _) in enumerate(groups):
            ys, xs, lo, hi = [], [], [], []
            for i, mdl in enumerate(vw_base.MODELS):
                (d, nc, nb) = res[(lab, sp_lab, mdl)]
                ys.append(i + (gi - 0.5) * 0.3); xs.append(d[0]); lo.append(d[0] - d[1]); hi.append(d[2] - d[0])
            ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", color=cols[lab], ecolor=cols[lab], elinewidth=2, capsize=0, markersize=7, label=lab)
        ax.axvline(0, color="#999", lw=1); ax.set_yticks(range(len(vw_base.MODELS))); ax.set_yticklabels([vw_base.short(m) for m in vw_base.MODELS])
        ax.set_title("%s rows" % sp_lab, fontsize=11, color="#0b0b0b"); ax.set_xlabel("Kalshi minus Polymarket B, Brier edge (95% interval)", fontsize=9, color="#52514e")
        ax.grid(axis="x", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis(); axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("The Kalshi gap lives in no-news, same-day rows; it is gone on rows with news and a day or more to run", fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(_paths.FIG_DIR, "venue-and-window-kalshi-gap-is-question-mix.png")
    fig.savefig(out, dpi=150); print("figure", out)
