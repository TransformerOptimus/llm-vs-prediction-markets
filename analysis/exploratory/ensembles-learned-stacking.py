import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""ensembles-learned-stacking: equal weights are not the only way to combine models. Do LEARNED
weights close the gap to the market?

The paper tests the plain average, the median, the log-odds average and an extremized pool
(ensembles-and-extremizing), finds they tie or trail the best single model, and concludes that
combining the models cannot close the gap. But equal weighting is not the same thing
as ensembling: the usual method is to LEARN how much weight each model deserves.

This script learns the weights. Everything learned is cross-fitted by event, which means: the rows are
split into five groups ("folds") by event, so every market belonging to one event lands in the same
fold; the weights used to score a row are fitted on the other four folds only. No row is ever scored by
weights that saw its own event. Folds are drawn with seed 20260902.

Combinations tested, all on the rows every roster model forecast:

  mean, median, log-odds mean   the equal-weight pools the paper already reports, repeated here on the
                               same rows so the learned versions have something to beat.
  NNLS stack                   non-negative least squares: choose weights w >= 0 minimising the squared
                               error of sum(w_j p_j) against the outcome. Non-negative weights are the
                               standard choice for forecast combination because a negative weight means
                               betting against a model, which rarely generalises.
  NNLS stack + intercept       the same with a free constant added, which lets the stack shift the whole
                               pool toward or away from the base rate.
  logistic stack               a logistic regression of the outcome on the models' forecasts in log-odds,
                               with a small ridge penalty so it cannot blow up when two models are nearly
                               identical. This is the form that can also undo shrinkage, because it is
                               free to scale the log-odds up.
  diversity subset (mean)      a subset of models chosen for how well they work TOGETHER: greedy forward
                               selection on the training folds, adding whichever model most improves the
                               equal-weight mean of the subset, stopping when nothing improves it. This
                               tests the idea that the weakest or most redundant models are dragging the
                               pool down.
  best single, picked on cal   the single model with the lowest Brier on the CALIBRATION split, then
                               scored on the test rows. Picking the best single model on the test rows
                               themselves flatters it; this is the honest version.
  best single, picked on test  the pick made on the test rows themselves, kept for comparison.

Reported per window: Brier for each combination, the Brier edge to the market on the same rows (market
Brier minus the combination's Brier, so negative means the market is still ahead) with a 95 per cent
interval, and a plain verdict on whether the combination reaches the market. Intervals are 1,000 event
resamples, seed 20260902. The test split is the headline; both splits follow for reference.
Read-only on both databases.
"""
import numpy as np
from scipy.optimize import nnls

import correlated_errors_common as hc

K_FOLDS = 5
EPS = 0.01
RIDGE = 1e-3


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def folds_by_event(events, k=K_FOLDS, seed=hc.SEED):
    """Assign every row a fold 0..k-1, all rows of one event sharing a fold."""
    uniq = sorted({str(e) for e in events})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    of = {uniq[i]: int(order[i] % k) for i in range(len(uniq))}
    return np.array([of[str(e)] for e in events])


def fit_nnls(P, y, intercept=False):
    """Weights w >= 0 minimising the squared error of (c + P w) against y; c is free only when asked
    for, and is handled by centring both sides, which is the exact solution for a free constant."""
    if intercept:
        pbar, ybar = P.mean(axis=0), float(y.mean())
        w, _ = nnls(P - pbar, y - ybar)
        return (float(ybar - pbar @ w), w)
    w, _ = nnls(P, y)
    return (0.0, w)


def apply_nnls(fit, P):
    c, w = fit
    return np.clip(c + P @ w, EPS, 1 - EPS)


def fit_logistic(P, y, ridge=RIDGE, iters=50):
    """Ridge-penalised logistic regression of y on [1, logit(P)], by iteratively reweighted least squares."""
    X = np.column_stack([np.ones(len(P)), logit(P)])
    b = np.zeros(X.shape[1])
    pen = ridge * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    for _ in range(iters):
        mu = sigmoid(X @ b)
        W = np.maximum(mu * (1 - mu), 1e-6)
        H = X.T @ (X * W[:, None]) + pen
        g = X.T @ (y - mu) - pen @ b
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def apply_logistic(b, P):
    return np.clip(sigmoid(np.column_stack([np.ones(len(P)), logit(P)]) @ b), EPS, 1 - EPS)


def fit_subset(P, y):
    """Greedy forward selection of models whose equal-weight mean scores best on these rows."""
    chosen, best = [], float("inf")
    remaining = list(range(P.shape[1]))
    while remaining:
        cand = min(remaining, key=lambda j: brier(P[:, chosen + [j]].mean(axis=1), y))
        s = brier(P[:, chosen + [cand]].mean(axis=1), y)
        if s >= best - 1e-9:
            break
        chosen, best = chosen + [cand], s
        remaining.remove(cand)
    return chosen or [int(np.argmin([brier(P[:, j], y) for j in range(P.shape[1])]))]


def crossfit(P, y, folds, fit, apply_):
    """Out-of-fold predictions: each fold scored by a fit made on the other folds."""
    out = np.full(len(y), np.nan)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        if tr.sum() < 50 or te.sum() == 0:
            continue
        m = fit(P[tr], y[tr])
        out[te] = apply_(m, P[te])
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = P[bad].mean(axis=1)
    return out


data, cuts = hc.load("normal")
print("rows: common rows per window (every roster model forecast the row), forecasts view, mode normal, "
      "status in %s, primary set. Learned weights are cross-fitted by event: %d folds split on "
      "source_event_id, seed %d, so no row is scored by weights fitted on its own event. Edge = market "
      "Brier minus the combination's Brier on the same rows; negative means the market is still ahead. "
      "Intervals = %d event resamples, seed %d."
      % (hc.STATUS_OK, K_FOLDS, hc.SEED, hc.N_BOOT, hc.SEED))

for w in hc.WINDOWS:
    rows_w = data[w]
    t_all = hc.Table(rows_w, hc.ROSTER[w])
    split_all = np.array([r["split"] for r in rows_w])
    cal = np.flatnonzero(split_all != "test")
    if len(cal) < 50:
        print("=== window %s: calibration split too small (%d rows); not reported" % (w, len(cal)))
        continue
    # The honest single-model pick: lowest Brier on the calibration split, made once.
    Pc, yc = t_all.P[cal], t_all.y[cal]
    cal_best = int(np.argmin([brier(Pc[:, j], yc) for j in range(t_all.k)]))

    for which, keep in (("test split", split_all == "test"), ("both splits", np.ones(len(rows_w), bool))):
        sel = np.flatnonzero(keep)
        rows = [rows_w[i] for i in sel]
        t = hc.Table(rows, hc.ROSTER[w])
        P, q, y = t.P, t.q, t.y
        folds = folds_by_event(t.event)
        nf = [int((folds == f).sum()) for f in range(K_FOLDS)]
        print("=== window %s, %s: %d rows, %d events, %d models | fold sizes %s | best single on the "
              "calibration split = %s" % (w, which, t.n, len(set(t.event)), t.k, nf, t.short[cal_best]))

        subsets = []
        for f in range(K_FOLDS):
            tr = folds != f
            if tr.sum() >= 50:
                subsets.append(tuple(sorted(fit_subset(P[tr], y[tr]))))
        common = max(set(subsets), key=subsets.count) if subsets else tuple(range(t.k))
        print("  diversity subset chosen on the training folds (commonest of the %d folds): %s"
              % (len(subsets), ", ".join(t.short[j] for j in common)))
        nn = fit_nnls(P, y)
        print("  NNLS weights refitted on all these rows, for description only (not scored): %s"
              % ", ".join("%s %.2f" % (t.short[j], nn[1][j]) for j in range(t.k)))

        combos = {
            "mean (equal weight)": P.mean(axis=1),
            "median": np.median(P, axis=1),
            "log-odds mean": sigmoid(logit(P).mean(axis=1)),
            "NNLS stack": crossfit(P, y, folds, lambda A, b: fit_nnls(A, b), apply_nnls),
            "NNLS stack + intercept": crossfit(P, y, folds, lambda A, b: fit_nnls(A, b, True), apply_nnls),
            "logistic stack": crossfit(P, y, folds, fit_logistic, apply_logistic),
            "diversity subset (mean)": crossfit(
                P, y, folds, lambda A, b: fit_subset(A, b), lambda s, A: A[:, s].mean(axis=1)),
            "best single, picked on cal (%s)" % t.short[cal_best]: P[:, cal_best],
        }
        test_best = int(np.argmin([brier(P[:, j], y) for j in range(t.k)]))
        combos["best single, picked on test (%s)" % t.short[test_best]] = P[:, test_best]

        mkt = brier(q, y)
        print("  market on the same rows: Brier %.4f" % mkt)
        for name, p in combos.items():
            p = np.asarray(p, dtype=float)

            def stat(idx, p=p):
                return np.array([np.mean((q[idx] - y[idx]) ** 2) - np.mean((p[idx] - y[idx]) ** 2)])

            pt_, lo_, hi_ = hc.boot(t, stat)
            reach = "REACHES the market" if hi_ >= 0 else "does not reach the market"
            print("  %-34s Brier %.4f  edge %s  %s" % (name, brier(p, y), hc.fmt((pt_, lo_, hi_), 4), reach))
