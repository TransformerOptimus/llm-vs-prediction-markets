"""Hypothesis: the Yes-side plateau is a sorting (resolution) failure, not a scale that recalibration can fix.

Per model and window on the primary set: Murphy decomposition of the Brier score for the model and
the market (10 equal-width bins), and a Platt recalibration (two-parameter logistic map on logit p)
fitted on the calibration split and scored on the test split. Bootstrap: 1,000 resamples of events,
seed 20260902."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import math, statistics as st
import numpy as np
from collections import defaultdict
from calibration_shape_common import *

rows, cuts = load()


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4); return math.log(p / (1 - p))


def platt_fit(ps, ys):
    X = np.array([[1.0, logit(p)] for p in ps]); y = np.array(ys, float)
    w = np.zeros(2)
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (pr - y); H = (X * (pr * (1 - pr))[:, None]).T @ X + 1e-6 * np.eye(2)
        w -= np.linalg.solve(H, g)
    return w


def sig(z): return 1 / (1 + math.exp(-z))


print("=== Murphy decomposition (10 bins). RES = resolution (higher is better), REL = reliability (lower is better)")
print("window model n | model REL RES | market REL RES | market-minus-model RES [95% CI] | model-minus-market REL [95% CI] | brier gap")
tot = 0
for w in "ABC":
    for mdl in MODELS:
        s = [r for r in rows if r["window"] == w and r["forecaster"] == mdl]
        if not s: continue
        tot += len(s)
        rm = murphy([r["p"] for r in s], [r["outcome_yes"] for r in s])
        rq = murphy([r["q"] for r in s], [r["outcome_yes"] for r in s])
        dres = boot_stat(s, lambda rs: murphy([r["q"] for r in rs], [r["outcome_yes"] for r in rs])[1] - murphy([r["p"] for r in rs], [r["outcome_yes"] for r in rs])[1])
        drel = boot_stat(s, lambda rs: murphy([r["p"] for r in rs], [r["outcome_yes"] for r in rs])[0] - murphy([r["q"] for r in rs], [r["outcome_yes"] for r in rs])[0])
        gap = st.fmean(r["brier_model"] - r["brier_market"] for r in s)
        print("%s %-16s %5d | %.4f %.4f | %.4f %.4f | %s | %s | %.4f  (RES share of gap %.0f%%)" % (
            w, SHORT[mdl], len(s), rm[0], rm[1], rq[0], rq[1], fmt(dres), fmt(drel), gap, 100 * dres[0] / gap))
print("total forecast rows", tot)

print("\n=== Platt recalibration fitted on the calibration split, scored on the test split")
print("window model | a b | test Brier raw recal market | raw-minus-recal [95% CI] | share of gap to market closed")
for w in "ABC":
    for mdl in MODELS:
        cal = [r for r in rows if r["window"] == w and r["forecaster"] == mdl and r["split"] == "calibration"]
        tst = [r for r in rows if r["window"] == w and r["forecaster"] == mdl and r["split"] == "test"]
        if not cal: continue
        a, b = platt_fit([r["p"] for r in cal], [r["outcome_yes"] for r in cal])
        for r in tst:
            r["p_recal"] = sig(a + b * logit(r["p"]))
            r["brier_recal"] = (r["p_recal"] - r["outcome_yes"]) ** 2
            r["gain"] = r["brier_model"] - r["brier_recal"]
        bm = st.fmean(r["brier_model"] for r in tst); br = st.fmean(r["brier_recal"] for r in tst); bq = st.fmean(r["brier_market"] for r in tst)
        g = boot_mean(tst, "gain")
        print("%s %-16s | a=%+.2f b=%.2f | %.4f %.4f %.4f | %s | %.0f%%  (test n %d, cal n %d)" % (
            w, SHORT[mdl], a, b, bm, br, bq, fmt(g), 100 * (bm - br) / (bm - bq), len(tst), len(cal)))

if _paths.want_figure():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Figure: top row mean p by market-probability bin (the plateau), bottom row outcome rate by p bin (reliability).
    edges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    pb = [i / 10 for i in range(11)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), facecolor=SURFACE)
    for j, w in enumerate("ABC"):
        top, bot = axes[0, j], axes[1, j]
        for ax in (top, bot):
            ax.set_facecolor(SURFACE); ax.plot([0, 1], [0, 1], color=INK, lw=1, ls=":", zorder=1)
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(color="#e6e5e1", lw=0.6)
        for mdl in MODELS:
            s = [r for r in rows if r["window"] == w and r["forecaster"] == mdl]
            if not s: continue
            col = COLORS.get(mdl, GREY); lw = 2 if mdl in COLORS else 1.2; z = 3 if mdl in COLORS else 2
            xs, ys = [], []
            for lo, hi in zip(edges, edges[1:]):
                b = [r["p"] for r in s if lo <= r["q"] < hi]
                if b: xs.append((lo + hi) / 2); ys.append(st.fmean(b))
            top.plot(xs, ys, color=col, lw=lw, zorder=z, marker="o", ms=4)
            xs, ys = [], []
            for lo, hi in zip(pb, pb[1:]):
                b = [r["outcome_yes"] for r in s if lo <= r["p"] < hi]
                if len(b) >= 20: xs.append((lo + hi) / 2); ys.append(st.fmean(b))
            bot.plot(xs, ys, color=col, lw=lw, zorder=z, marker="o", ms=4)
        top.set_title("Window %s" % w, color=INK, fontsize=11)
        top.set_xlabel("market probability of Yes (q)"); top.set_ylabel("mean model probability (p)" if j == 0 else "")
        bot.set_xlabel("model probability (p)"); bot.set_ylabel("share that resolved Yes" if j == 0 else "")
    for m, c in COLORS.items(): axes[0, 0].plot([], [], color=c, lw=2, marker="o", ms=4, label=SHORT[m])
    axes[0, 0].plot([], [], color=GREY, lw=1.2, label="other models"); axes[0, 0].plot([], [], color=INK, ls=":", label="p = q")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Top: models flatten against the market above q = 0.5. Bottom: on their own scale they are close to calibrated.", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration-shape-resolution-gap.png"), dpi=150, facecolor=SURFACE)
    print("figure written")
