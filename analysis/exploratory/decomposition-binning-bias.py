import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""decomposition-binning-bias: does the split of the Brier score into calibration error and sorting
survive a better method than ten equal-width bins?

A Murphy decomposition with ten equal-width bins attributes most of each model's shortfall to
resolution (sorting) rather than reliability (calibration error). Binning is a convenience, not the
definition. Inside a bin, forecasts that differ are treated as equal, which throws
away part of the resolution and mislabels part of it as reliability. The size of that error depends on
how the forecasts are spread inside the bins -- and our two sides differ exactly there. Model answers
pile up on round numbers (0.2, 0.25, 0.3) while market prices are continuous, so the binning error is
not the same for the model and for the market, and the attribution could be an artefact.

This script recomputes the decomposition without bins, by the standard method: fit the outcome against
the forecast with pool adjacent violators, which gives the best monotone estimate of "what actually
happens at this forecast level" with no bin width to choose. Reliability is then the mean squared
distance between the forecast and that fitted value, resolution the spread of the fitted values around
the base rate. The two methods are reported side by side, per model and window, along with the share of
the Brier gap that each attributes to resolution.

Also reported: how concentrated each forecaster's answers are (the share of rows landing on the ten
commonest values), which is what drives the difference, and a residual term for the binned method --
Brier minus (reliability - resolution + uncertainty) -- which is exactly zero for the unbinned method
by construction and is the part the ten-bin numbers leave unexplained.

Intervals are 1,000 event resamples, seed 20260902. Test split and both splits are both reported.
"""
from collections import Counter

from calibration_shape_common import MODELS, SHORT, boot_stat, load, murphy


def pav(ps, ys):
    """Pool adjacent violators: the best monotone fit of y against p. Returns per-row fitted values."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    v = [float(ys[i]) for i in order]
    w = [1.0] * len(v)
    idx = [[i] for i in order]
    k = 0
    while k < len(v) - 1:
        if v[k] <= v[k + 1] + 1e-12:
            k += 1
            continue
        tot = w[k] + w[k + 1]
        v[k] = (v[k] * w[k] + v[k + 1] * w[k + 1]) / tot
        w[k] = tot
        idx[k] = idx[k] + idx[k + 1]
        del v[k + 1], w[k + 1], idx[k + 1]
        if k > 0:
            k -= 1
    out = [0.0] * len(ps)
    for block, val in zip(idx, v):
        for i in block:
            out[i] = val
    return out


def corp(ps, ys):
    """Unbinned decomposition: (reliability, resolution, uncertainty). Brier = REL - RES + UNC exactly."""
    fitted = pav(ps, ys)
    n = len(ps)
    ybar = sum(ys) / n
    rel = sum((p - f) ** 2 for p, f in zip(ps, fitted)) / n
    res = sum((f - ybar) ** 2 for f in fitted) / n
    return rel, res, ybar * (1 - ybar)


def brier(ps, ys):
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def concentration(ps, k=10):
    c = Counter(round(p, 4) for p in ps)
    return sum(n for _, n in c.most_common(k)) / len(ps)


def share_res(rel_m, res_m, rel_q, res_q):
    """Share of the model's Brier shortfall that the decomposition attributes to resolution."""
    gap = (rel_m - rel_q) + (res_q - res_m)
    return (res_q - res_m) / gap if gap > 0 else float("nan")


rows_all, cuts = load("normal")
print("rows: primary set, forecasts view, mode normal. BINNED = Murphy with 10 equal-width bins (what "
      "the paper uses); UNBINNED = same decomposition with pool adjacent violators instead of bins. "
      "RES higher is better, REL lower is better. 'res share' = the share of the model's Brier "
      "shortfall attributed to resolution. 'top10' = share of the forecaster's answers falling on its "
      "ten commonest values. intervals = 1,000 event resamples, seed 20260902")

for w in "ABC":
    for which, sel in (("test split", lambda r: r["split"] == "test"), ("both splits", lambda r: True)):
        rs = [r for r in rows_all if r["window"] == w and sel(r)]
        if not rs:
            continue
        qs = [r["q"] for r in rs if r["forecaster"] == MODELS[0]]
        ys_q = [r["outcome_yes"] for r in rs if r["forecaster"] == MODELS[0]]
        print("== window %s, %s | market answers: top10 %.3f" % (w, which, concentration(qs)))
        for mdl in MODELS:
            s = [r for r in rs if r["forecaster"] == mdl]
            if len(s) < 100:
                continue
            ps = [r["p"] for r in s]
            ys = [r["outcome_yes"] for r in s]
            mq = [r["q"] for r in s]
            b_rel_m, b_res_m, unc = murphy(ps, ys)
            b_rel_q, b_res_q, _ = murphy(mq, ys)
            c_rel_m, c_res_m, _ = corp(ps, ys)
            c_rel_q, c_res_q, _ = corp(mq, ys)
            resid_m = brier(ps, ys) - (b_rel_m - b_res_m + unc)
            resid_q = brier(mq, ys) - (b_rel_q - b_res_q + unc)
            sb = share_res(b_rel_m, b_res_m, b_rel_q, b_res_q)
            sc = share_res(c_rel_m, c_res_m, c_rel_q, c_res_q)
            ci = boot_stat(s, lambda t: share_res(*corp([r["p"] for r in t], [r["outcome_yes"] for r in t])[:2],
                                                  *corp([r["q"] for r in t], [r["outcome_yes"] for r in t])[:2]))
            print("  %-16s n=%-5d top10 %.3f | BINNED   model REL %.4f RES %.4f  market REL %.4f RES %.4f"
                  "  res share %5.1f%%  unexplained model %+.5f market %+.5f"
                  % (SHORT[mdl], len(s), concentration(ps), b_rel_m, b_res_m, b_rel_q, b_res_q,
                     100 * sb, resid_m, resid_q))
            print("  %-16s       %s | UNBINNED model REL %.4f RES %.4f  market REL %.4f RES %.4f"
                  "  res share %5.1f%% [%.1f%%, %.1f%%]"
                  % ("", " " * 11, c_rel_m, c_res_m, c_rel_q, c_res_q,
                     100 * sc, 100 * ci[1], 100 * ci[2]))

# ---------------------------------------------------------------------------
# The bin-free figure PER WINDOW (the paper pools it as "79 to 99 per cent" across every model-window
# cell) and the name of the model that reaches 99. This block only re-reads the same numbers and prints
# the per-window range and the model at each end.
# ---------------------------------------------------------------------------
print()
print("### PER-WINDOW SUMMARY of the bin-free (pool-adjacent-violators) resolution share")
print("Same numbers as the UNBINNED lines above, gathered per window: the lowest and highest share of "
      "the Brier shortfall that the bin-free decomposition attributes to resolution (sorting), and "
      "which model sits at each end. Over 100 per cent means the model's calibration error is no "
      "worse than the market's, so resolution accounts for more than the whole shortfall.")
for w in "ABC":
    for which, sel in (("test split", lambda r: r["split"] == "test"), ("both splits", lambda r: True)):
        rs = [r for r in rows_all if r["window"] == w and sel(r)]
        if not rs:
            continue
        got = []
        for mdl in MODELS:
            s = [r for r in rs if r["forecaster"] == mdl]
            if len(s) < 100:
                continue
            ps = [r["p"] for r in s]
            ys = [r["outcome_yes"] for r in s]
            mq = [r["q"] for r in s]
            c_rel_m, c_res_m, _ = corp(ps, ys)
            c_rel_q, c_res_q, _ = corp(mq, ys)
            b_rel_m, b_res_m, _ = murphy(ps, ys)
            b_rel_q, b_res_q, _ = murphy(mq, ys)
            got.append((share_res(c_rel_m, c_res_m, c_rel_q, c_res_q),
                        share_res(b_rel_m, b_res_m, b_rel_q, b_res_q), SHORT[mdl]))
        if not got:
            continue
        got.sort()
        print("  window %s, %-11s bin-free %5.1f%% (%s) to %5.1f%% (%s) over %d models"
              % (w, which, 100 * got[0][0], got[0][2], 100 * got[-1][0], got[-1][2], len(got)))
        byb = sorted(got, key=lambda g: g[1])
        print("  %s  ten-bin  %5.1f%% (%s) to %5.1f%% (%s)"
              % (" " * 21, 100 * byb[0][1], byb[0][2], 100 * byb[-1][1], byb[-1][2]))
        print("  %s  per model, bin-free: %s"
              % (" " * 21, ", ".join("%s %.1f%%" % (g[2], 100 * g[0]) for g in got)))
