import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""recalibration-crossfit-and-isotonic: how much of the gap can be recovered by rescaling the model's
probabilities, without giving it any new information?

One Platt fit on one held-out slice is a weak test: on Window A it gives a negative recovery and
slopes above one, a sign the slice is small rather than that recalibration cannot help. This script
compares three rescalings:

  platt (single split)  the existing method, repeated here for comparison: a two-parameter logistic
                        map on log-odds p, fitted on the calibration split, scored on the test split
  platt (cross-fit)     the same map, but fitted k times on k-1 folds and applied to the held-out fold,
                        so every row is scored by a map that never saw it and all the data is used
  isotonic (cross-fit)  the same cross-fitting with a monotone step function instead, fitted by pool
                        adjacent violators. This assumes only that a higher forecast should mean a
                        higher chance, not any particular shape, so it is the strongest rescaling a
                        forecaster could hope for

The ceiling matters: rescaling can only fix reliability. It cannot create resolution, because a
monotone map does not change the ORDER of the forecasts, and resolution depends only on the order and
the outcomes. So the Murphy resolution is reported before and after; if it barely moves while the
Brier improves, the recovered part was calibration error and the rest of the gap is sorting.

Recovery is the share of the model's Brier gap to the market that the rescaling removes, on the same
rows. Folds are split by EVENT, never by row, so two markets about the same thing cannot land on
opposite sides of a fold. k = 5, seed 20260902. Intervals are 1,000 event resamples of the same seed.
"""
import math
import random
from collections import defaultdict

import numpy as np
from calibration_shape_common import MODELS, SHORT, boot_stat, load, murphy

K = 5
SEED = 20260902
EPS = 1e-4


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def platt_fit(ps, ys):
    X = np.array([[1.0, logit(p)] for p in ps])
    y = np.array(ys, dtype=float)
    w = np.zeros(2)
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (pr - y)
        H = (X * (pr * (1 - pr))[:, None]).T @ X + 1e-6 * np.eye(2)
        w -= np.linalg.solve(H, g)
    return w


def platt_apply(w, ps):
    return [1 / (1 + math.exp(-(w[0] + w[1] * logit(p)))) for p in ps]


def isotonic_fit(ps, ys):
    """Pool adjacent violators. Returns (sorted knots, fitted values) for step lookup."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    x = [ps[i] for i in order]
    v = [float(ys[i]) for i in order]
    wt = [1.0] * len(v)
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-12:
            i += 1
            continue
        tot = wt[i] + wt[i + 1]
        v[i] = (v[i] * wt[i] + v[i + 1] * wt[i + 1]) / tot
        wt[i] = tot
        del v[i + 1], wt[i + 1], x[i + 1]
        if i > 0:
            i -= 1
    return x, v


def isotonic_apply(knots, vals, ps):
    x = np.asarray(knots)
    out = []
    for p in ps:
        j = int(np.searchsorted(x, p, side="right")) - 1
        out.append(vals[min(max(j, 0), len(vals) - 1)])
    return out


def folds(rows, k=K, seed=SEED):
    """k folds of row indices, split by event so a whole event stays together."""
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[r["event_id"]].append(i)
    ids = sorted(g)
    random.Random(seed).shuffle(ids)
    out = [[] for _ in range(k)]
    for n, e in enumerate(ids):
        out[n % k].extend(g[e])
    return out


def crossfit(rows, method):
    """Every row's rescaled probability, from a map fitted on the other folds only."""
    out = [None] * len(rows)
    fs = folds(rows)
    for j, held in enumerate(fs):
        train = [i for m, f in enumerate(fs) if m != j for i in f]
        if len(train) < 40 or not held:
            continue
        ps = [rows[i]["p"] for i in train]
        ys = [rows[i]["outcome_yes"] for i in train]
        hp = [rows[i]["p"] for i in held]
        new = platt_apply(platt_fit(ps, ys), hp) if method == "platt" else \
            isotonic_apply(*isotonic_fit(ps, ys), hp)
        for i, v in zip(held, new):
            out[i] = v
    return out


def recovery(rows, newp):
    """Share of the model's Brier gap to the market removed by the rescaling."""
    keep = [i for i, v in enumerate(newp) if v is not None]
    if len(keep) < 50:
        return None
    bm = sum(rows[i]["brier_model"] for i in keep) / len(keep)
    bq = sum(rows[i]["brier_market"] for i in keep) / len(keep)
    bn = sum((newp[i] - rows[i]["outcome_yes"]) ** 2 for i in keep) / len(keep)
    gap = bm - bq
    return bm, bn, bq, ((bm - bn) / gap if gap > 0 else float("nan"))


rows_all, cuts = load("normal")
print("rows: primary set, forecasts view, mode normal; recovery = share of the model's Brier gap to "
      "the market that the rescaling removes; folds split by event, k=%d, seed %d; RES = Murphy "
      "resolution (10 bins), which a monotone rescaling cannot raise" % (K, SEED))

for w in "ABC":
    print("=== window %s" % w)
    for mdl in MODELS:
        s = [r for r in rows_all if r["window"] == w and r["forecaster"] == mdl]
        if len(s) < 200:
            continue
        cal = [r for r in s if r["split"] != "test"]
        test = [r for r in s if r["split"] == "test"]
        parts = []
        if len(cal) >= 40 and len(test) >= 50:
            w_ = platt_fit([r["p"] for r in cal], [r["outcome_yes"] for r in cal])
            got = recovery(test, platt_apply(w_, [r["p"] for r in test]))
            if got:
                parts.append(("platt (single split)", test, got, w_))
        for method in ("platt", "isotonic"):
            got = recovery(s, crossfit(s, method))
            if got:
                parts.append(("%s (cross-fit)" % method, s, got, None))
        _, res_before, _ = murphy([r["p"] for r in s], [r["outcome_yes"] for r in s])
        _, res_mkt, _ = murphy([r["q"] for r in s], [r["outcome_yes"] for r in s])
        print("  %-16s n=%-5d RES %.4f  market RES %.4f" % (SHORT[mdl], len(s), res_before, res_mkt))
        for name, used, (bm, bn, bq, rec), wts in parts:
            newp = crossfit(used, name.split()[0]) if wts is None else platt_apply(wts, [r["p"] for r in used])
            keep = [i for i, v in enumerate(newp) if v is not None]
            _, res_after, _ = murphy([newp[i] for i in keep], [used[i]["outcome_yes"] for i in keep])
            extra = "" if wts is None else "  slope %.3f intercept %+.3f" % (wts[1], wts[0])
            print("    %-22s n=%-5d Brier %.4f -> %.4f (market %.4f)  recovery %5.1f%%  RES %.4f -> %.4f%s"
                  % (name, len(keep), bm, bn, bq, 100 * rec, res_before, res_after, extra))
