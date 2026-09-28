import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""ensembles-and-extremizing: does combining the models close the gap to the market?

Does averaging models help? It is worth testing for two reasons. Averaging correlated forecasts still
cancels independent noise, so an ensemble can beat every one of its members even when they agree closely. And because every model shrinks toward the base
rate (see correlated-errors-shrinkage-null), an average of them shrinks just as hard, so an ensemble
pushed back outward could recover much of what shrinkage costs.

Per window, on the common rows (every roster model forecast the row, primary set, mode normal), six
combinations, each scored against the market on the same rows:

  best single    the roster member with the lowest Brier on these rows. This is chosen on the test
                 rows themselves, so it flatters the single model and is not a fair contest against
                 the combinations; it is here as an upper bound on what picking one model could do.
                 The question being tested is whether a combination closes the gap to the MARKET, and
                 that comparison is unaffected.
  mean           the plain average of the forecasts
  median         the middle forecast, which ignores an outlier model
  log-odds mean  the average taken in log-odds, which is the usual way to pool probabilities
  extremized     the log-odds mean multiplied by a factor fitted on the calibration split. Above 1
                 the factor pushes answers away from the middle, which is the standard correction
                 for the shrinkage that pooling causes; below 1 it pulls them in. The grid runs
                 from 0.50 to 3.00 so that both directions are available and "no change" is not
                 the boundary of the search.
  unshrunk       the mean with each model's fitted shrinkage undone: (p - a) / b, clipped to [0.01,0.99]

Every parameter that has to be learned -- the extremizing factor a, and the a and b of the shrinkage
fits -- is fitted on the CALIBRATION split and applied unchanged to the TEST split, which is what that
split was reserved for. Nothing is tuned on the numbers reported. The extremizing factor is chosen over
a grid by Brier on the calibration split alone.

Reported per window: Brier, the market's Brier on the same rows, the edge (market minus combination:
negative means the market is still ahead), and the Murphy resolution, so it is visible whether a
combination closes the gap by sorting better or only by being better calibrated. Intervals are 1,000
event resamples, seed 20260902.
"""
import numpy as np
import correlated_errors_common as hc
from calibration_shape_common import murphy

# Two-sided: a factor below 1 pulls the pooled forecast IN toward even money, above 1
# pushes it OUT. Starting the grid at 1.0 would have made "no extremizing" the floor and
# so unfalsifiable; forecasters who already agree closely can need pulling in, not out.
GRID = np.round(np.arange(0.50, 3.01, 0.05), 2)
EPS = 0.01


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_ab(P, q):
    """Least squares p = a + b q per model column."""
    qc = q - q.mean()
    var = float(qc @ qc)
    b = (qc @ (P - P.mean(axis=0))) / var if var > 0 else np.ones(P.shape[1])
    a = P.mean(axis=0) - b * q.mean()
    return a, b


def brier(p, y):
    return float(np.mean((p - y) ** 2))


data, cuts = hc.load("normal")
print("rows: common rows per window, forecasts view, mode normal, status in %s, primary set; "
      "combinations fitted on the calibration split and scored on the test split; "
      "intervals = %d event resamples, seed %d" % (hc.STATUS_OK, hc.N_BOOT, hc.SEED))

for w in hc.WINDOWS:
    rows = data[w]
    t = hc.Table(rows, hc.ROSTER[w])
    split = np.array([r["split"] for r in rows])
    cal, test = np.flatnonzero(split != "test"), np.flatnonzero(split == "test")
    if len(cal) < 50 or len(test) < 50:
        print("=== window %s: split too small (cal %d, test %d); not reported" % (w, len(cal), len(test)))
        continue
    Pc, qc_, yc = t.P[cal], t.q[cal], t.y[cal]
    Pt, qt, yt = t.P[test], t.q[test], t.y[test]
    print("=== window %s: %d common rows, %d models | calibration %d rows, test %d rows"
          % (w, t.n, t.k, len(cal), len(test)))

    # Parameters learned on the calibration split only.
    a_hat, b_hat = fit_ab(Pc, qc_)
    lo_cal = logit(Pc).mean(axis=1)
    scores = [(brier(sigmoid(g * lo_cal), yc), g) for g in GRID]
    g_hat = min(scores)[1]
    print("  fitted on the calibration split: extremizing factor a = %.2f; "
          "mean shrinkage slope b = %.4f, intercept a = %+.4f" % (g_hat, b_hat.mean(), a_hat.mean()))

    b_safe = np.where(np.abs(b_hat) < 0.05, np.nan, b_hat)
    combos = {
        "mean": Pt.mean(axis=1),
        "median": np.median(Pt, axis=1),
        "log-odds mean": sigmoid(logit(Pt).mean(axis=1)),
        "extremized (a=%.2f)" % g_hat: sigmoid(g_hat * logit(Pt).mean(axis=1)),
        "unshrunk": np.clip(np.nanmean((Pt - a_hat) / b_safe, axis=1), EPS, 1 - EPS),
    }
    best_j = int(np.argmin([brier(Pt[:, j], yt) for j in range(t.k)]))
    combos = {"best single (%s)" % t.short[best_j]: Pt[:, best_j], **combos}

    sub = hc.Table([rows[i] for i in test], hc.ROSTER[w])
    mkt = brier(qt, yt)
    _, res_q, unc = murphy(list(qt), list(yt))
    print("  market on the same rows: Brier %.4f  RES %.4f  UNC %.4f" % (mkt, res_q, unc))
    for name, p in combos.items():
        p = np.asarray(p, dtype=float)

        def stat(idx, p=p):
            return np.array([np.mean((qt[idx] - yt[idx]) ** 2) - np.mean((p[idx] - yt[idx]) ** 2)])

        pt_, lo_, hi_ = hc.boot(sub, stat)
        _, res_p, _ = murphy(list(p), list(yt))
        print("  %-22s Brier %.4f  edge %s  RES %.4f  (market RES minus this %+.4f)"
              % (name, brier(p, yt), hc.fmt((pt_, lo_, hi_), 4), res_p, res_q - res_p))
