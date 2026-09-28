"""Hypothesis: the shortfall toward the market's favored side is far larger when that side is Yes.

Per window (models pooled, then per model), primary set: band 0.75-0.95 (Yes favored) versus band
0.05-0.25 (No favored). Shortfall = distance from p to q toward the favored side (q - p in the Yes
band, p - q in the No band). Cost = brier_model - brier_market. Wrong side = p on the other side
of 0.5 from the market. Bootstrap: 1,000 resamples of events, seed 20260902."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import statistics as st
from calibration_shape_common import *

rows, cuts = load()
for r in rows:
    r["cost"] = r["brier_model"] - r["brier_market"]
    r["shortfall"] = (r["q"] - r["p"]) if r["q"] > 0.5 else (r["p"] - r["q"])
    r["wrong_side"] = float((r["p"] < 0.5) if r["q"] > 0.5 else (r["p"] > 0.5))
HI, LO = "0.75-0.95", "0.05-0.25"


def report(label, s):
    hi = [r for r in s if r["band"] == HI]; lo = [r for r in s if r["band"] == LO]
    sh_hi, sh_lo = boot_mean(hi, "shortfall"), boot_mean(lo, "shortfall")
    c_hi, c_lo = boot_mean(hi, "cost"), boot_mean(lo, "cost")
    d_sh = boot_diff(hi, lo, "shortfall"); d_c = boot_diff(hi, lo, "cost")
    print("%-20s Yes band n=%d (mkts %d): market says %.3f, model says %.3f, shortfall %s wrong-side %.2f y-rate %.2f cost %s" % (
        label, len(hi), len({r["row_id"] for r in hi}), st.fmean(r["q"] for r in hi), st.fmean(r["p"] for r in hi),
        fmt(sh_hi), st.fmean(r["wrong_side"] for r in hi), st.fmean(r["outcome_yes"] for r in hi), fmt(c_hi)))
    print("%-20s No  band n=%d (mkts %d): market says %.3f, model says %.3f, shortfall %s wrong-side %.2f y-rate %.2f cost %s" % (
        "", len(lo), len({r["row_id"] for r in lo}), st.fmean(r["q"] for r in lo), st.fmean(r["p"] for r in lo),
        fmt(sh_lo), st.fmean(r["wrong_side"] for r in lo), st.fmean(r["outcome_yes"] for r in lo), fmt(c_lo)))
    print("%-20s Yes minus No: shortfall %s  cost %s" % ("", fmt(d_sh), fmt(d_c)))
    return sh_hi, sh_lo, c_hi, c_lo


summary = {}
for w in "ABC":
    print("=== Window", w, "(all models pooled; bootstrap over events)")
    summary[w] = report("all models", [r for r in rows if r["window"] == w])
    for mdl in MODELS:
        s = [r for r in rows if r["window"] == w and r["forecaster"] == mdl]
        if s: report(SHORT[mdl], s)
    print()

if _paths.want_figure():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    for ax, (k, title) in zip(axes, [((0, 1), "shortfall toward the market's side"), ((2, 3), "Brier cost versus the market")]):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
        xs = range(3)
        for off, idx, col, lab in [(-0.18, k[0], "#eb6834", "market favors Yes (q 0.75-0.95)"), (0.18, k[1], "#2a78d6", "market favors No (q 0.05-0.25)")]:
            vals = [summary[w][idx] for w in "ABC"]
            ax.bar([x + off for x in xs], [v[0] for v in vals], width=0.34, color=col, label=lab)
            ax.errorbar([x + off for x in xs], [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", ecolor=INK, lw=1, capsize=3)
        ax.set_xticks(list(xs)); ax.set_xticklabels(["Window A", "Window B", "Window C"]); ax.set_title(title, fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("All models pooled; bars are means with 95% event-bootstrap intervals", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "calibration-shape-yes-no-asymmetry.png"), dpi=150, facecolor=SURFACE)
    print("figure written")
