import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""calibration-shape-sports-split: is the gap sorting rather than calibration on non-sports rows too?

The paper attributes 79 to 99 per cent of each model's Brier shortfall to resolution (sorting) and
shows that no monotone rescaling recovers more than about a fifth. Both are computed on all rows, and
sports make up 73 to 82 per cent of them. The non-sports gap is several times wider per row, and the
calibration gate finds non-sports rows badly calibrated, so a pooled result could hide a non-sports
shortfall that is mostly calibration. This script reruns both analyses on each half separately.

Per window, per model, for sports rows and non-sports rows separately:
  - the bin-free decomposition (pool adjacent violators, as decomposition-binning-bias.py) of the
    model and of the market on the same rows, and the share of the model's Brier shortfall that it
    attributes to resolution, with a 1,000-resample event interval;
  - the ten-bin share, for comparison with the paper's ten-bin figures;
  - the share of the shortfall that a cross-fitted monotone rescaling (isotonic, pool adjacent
    violators; and Platt) removes, folds split by event, k = 5, as recalibration-crossfit-and-
    isotonic.py.

Rows: primary set, test split, forecasts view, mode normal. Sports is the benchmark's sports flag.
Seed 20260902 throughout. Read-only on both databases. The decomposition and rescaling functions are
copied from the two scripts named above so this file runs on its own.
"""
import math
import random
from collections import defaultdict

import numpy as np

from calibration_shape_common import MODELS, SHORT, boot_stat, load, murphy

K, SEED, EPS = 5, 20260902, 1e-4


def pav(ps, ys):
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
    fitted = pav(ps, ys)
    n = len(ps)
    ybar = sum(ys) / n
    return (sum((p - f) ** 2 for p, f in zip(ps, fitted)) / n,
            sum((f - ybar) ** 2 for f in fitted) / n)


def share(rel_m, res_m, rel_q, res_q):
    gap = (rel_m - rel_q) + (res_q - res_m)
    return (res_q - res_m) / gap if gap > 0 else float("nan")


def share_pav(rs):
    ys = [r["outcome_yes"] for r in rs]
    return share(*corp([r["p"] for r in rs], ys), *corp([r["q"] for r in rs], ys))


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def platt(train, held):
    X = np.array([[1.0, logit(r["p"])] for r in train])
    y = np.array([r["outcome_yes"] for r in train], dtype=float)
    w = np.zeros(2)
    for _ in range(50):
        pr = 1 / (1 + np.exp(-(X @ w)))
        H = (X * (pr * (1 - pr))[:, None]).T @ X + 1e-6 * np.eye(2)
        w -= np.linalg.solve(H, X.T @ (pr - y))
    return [1 / (1 + math.exp(-(w[0] + w[1] * logit(r["p"])))) for r in held]


def isotonic(train, held):
    order = sorted(range(len(train)), key=lambda i: train[i]["p"])
    x = [train[i]["p"] for i in order]
    v = [float(train[i]["outcome_yes"]) for i in order]
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
    xs = np.asarray(x)
    return [v[min(max(int(np.searchsorted(xs, r["p"], side="right")) - 1, 0), len(v) - 1)] for r in held]


def recovery(rs, fit):
    g = defaultdict(list)
    for i, r in enumerate(rs):
        g[r["event_id"]].append(i)
    ids = sorted(g)
    random.Random(SEED).shuffle(ids)
    fs = [[] for _ in range(K)]
    for n, e in enumerate(ids):
        fs[n % K].extend(g[e])
    newp = [None] * len(rs)
    for j, held in enumerate(fs):
        train = [rs[i] for m, f in enumerate(fs) if m != j for i in f]
        if len(train) < 40 or not held:
            continue
        for i, v in zip(held, fit(train, [rs[i] for i in held])):
            newp[i] = v
    keep = [i for i, v in enumerate(newp) if v is not None]
    bm = sum(rs[i]["brier_model"] for i in keep) / len(keep)
    bq = sum(rs[i]["brier_market"] for i in keep) / len(keep)
    bn = sum((newp[i] - rs[i]["outcome_yes"]) ** 2 for i in keep) / len(keep)
    return (bm - bn) / (bm - bq) if bm > bq else float("nan")


rows_all, _ = load("normal")
rows_all = [r for r in rows_all if r["split"] == "test"]
print("rows: primary set, test split, forecasts view, mode normal; sports = benchmark sports flag. "
      "'res share' = share of the model's Brier shortfall attributed to resolution (bin-free, with a "
      "1,000-resample event interval; ten-bin in brackets after it). 'rescale' = share of the "
      "shortfall removed by a cross-fitted isotonic / Platt map, folds by event, k=5, seed 20260902.")

for w in "ABC":
    print("=== window %s" % w)
    summary = defaultdict(list)
    for mdl in MODELS:
        s = [r for r in rows_all if r["window"] == w and r["forecaster"] == mdl]
        if not s:
            continue
        for label, sub in (("sports", [r for r in s if r["sports"]]),
                           ("non-sports", [r for r in s if not r["sports"]])):
            if len(sub) < 100:
                print("  %-16s %-10s n=%-5d (fewer than 100 rows; not reported)" % (SHORT[mdl], label, len(sub)))
                continue
            ys = [r["outcome_yes"] for r in sub]
            rel_m, res_m = corp([r["p"] for r in sub], ys)
            rel_q, res_q = corp([r["q"] for r in sub], ys)
            bm = sum(r["brier_model"] for r in sub) / len(sub)
            bq = sum(r["brier_market"] for r in sub) / len(sub)
            pt, lo, hi = boot_stat(sub, share_pav)
            b_rel_m, b_res_m, _ = murphy([r["p"] for r in sub], ys)
            b_rel_q, b_res_q, _ = murphy([r["q"] for r in sub], ys)
            ten = share(b_rel_m, b_res_m, b_rel_q, b_res_q)
            iso, pl = recovery(sub, isotonic), recovery(sub, platt)
            summary[label].append((pt, ten, iso, pl, bm - bq))
            print("  %-16s %-10s n=%-5d Brier gap %.4f | model REL %.4f RES %.4f | market REL %.4f RES %.4f "
                  "| res share %5.1f%% [%5.1f%%, %5.1f%%] (ten-bin %5.1f%%) | rescale isotonic %5.1f%% "
                  "Platt %5.1f%%" % (SHORT[mdl], label, len(sub), bm - bq, rel_m, res_m, rel_q, res_q,
                                     100 * pt, 100 * lo, 100 * hi, 100 * ten, 100 * iso, 100 * pl))
    for label in ("sports", "non-sports"):
        v = summary[label]
        if v:
            rng = lambda j: (100 * min(x[j] for x in v), 100 * max(x[j] for x in v))
            print("  RANGE %-10s over %d models: Brier gap %.4f to %.4f | res share %.1f to %.1f%% "
                  "(ten-bin %.1f to %.1f%%) | isotonic %.1f to %.1f%% | Platt %.1f to %.1f%%"
                  % (label, len(v), min(x[4] for x in v), max(x[4] for x in v), *rng(0), *rng(1), *rng(2), *rng(3)))
