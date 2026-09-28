import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-shrinkage-null: how much of the pairwise agreement between models is just the
fact that they all shrink the same way toward the same base rate?

Every model forecasts without seeing the price, and every model pulls its answers toward the middle.
If two models both do that, their departures from the price, p - q, are both dominated by the same
function of q, so they correlate strongly even when the two models share no judgement at all. The
extreme case: two forecasters that always answer 0.30 have a departure correlation of exactly 1.

This script measures that. Per window, on the common rows (every roster model forecast the row,
primary set, mode normal):

  1. SHRINKAGE. Least squares fit of each model's forecast on the market price, p = a + b q, on the
     probability scale and on the log-odds scale. b below 1 is shrinkage. The fixed point a / (1 - b)
     is the probability the model is shrinking toward; compare it with the window's Yes base rate.

  2. THE REAL NUMBERS. Mean over model pairs of the correlation of departures p - q (the measure the
     paper reports), and of the residuals from each model's own fit, which is the same correlation
     after the shared shrinkage has been taken out. Also the both-wrong rate against independence.

  3. THE NULL. Simulated forecasters that shrink exactly as much as the real ones but share nothing
     else: p_sim = a + b q + noise, with noise drawn independently per model. Two noise models, because
     neither is obviously right:
       gauss   normal noise with the model's residual standard deviation, per price band
       permute the model's own residuals, shuffled across rows within a price band, which keeps the
               real residual shape and its dependence on the price without any cross-model link
     Both are clipped to [0.01, 0.99], as a probability must be. Every statistic in step 2 is
     recomputed on each simulated draw, giving the range a shared-shrinkage-only world would produce.

  VERDICT per window: whether the real value falls inside the simulated 95 per cent range. Inside
  means the agreement is shrinkage. Above means the excess is agreement that shrinkage cannot explain,
  and the excess is the number worth reporting.

Reported on the test split (what the result tables use) and on both splits together (what the earlier
exploratory scripts used), so the two are comparable. Intervals on real quantities are 1,000 event
resamples, seed 20260902, refitting inside each resample. The null range is the 2.5th to 97.5th
percentile over 400 simulated draws, same seed.
"""
import numpy as np
import correlated_errors_common as hc

N_SIM = 400
BANDS = ["0.05-0.25", "0.25-0.50", "0.50-0.75", "0.75-0.95"]


def fit(P, q):
    """Least squares p = a + b q per column. Returns a (k,), b (k,), residuals (n, k)."""
    qc = q - q.mean()
    var = float(qc @ qc)
    b = (qc @ (P - P.mean(axis=0))) / var if var > 0 else np.zeros(P.shape[1])
    a = P.mean(axis=0) - b * q.mean()
    return a, b, P - (a + np.outer(q, b))


def logit(x, eps=0.01):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def mean_pair_corr(V, iu):
    if len(V) < 3:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(V, rowvar=False)
    return float(np.nanmean(c[iu])) if np.isfinite(c[iu]).any() else float("nan")


def both_wrong(P, y, iu):
    """(mean pair both-wrong rate, mean pair independence product, ratio)."""
    W = ((P >= 0.5) != (y[:, None] >= 0.5)).astype(float)
    n = len(P)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    both = float(((W.T @ W) / n)[iu].mean())
    m = W.mean(axis=0)
    exp = float(np.outer(m, m)[iu].mean())
    return both, exp, (both / exp if exp > 0 else float("nan"))


def both_wrong_market_right(P, q, y, iu):
    """The same ratio restricted to rows the market itself got right.

    The paper reports this to show a shared mistake cannot be blamed on a surprising
    outcome. It needs the same null as the unrestricted ratio, and arguably more: a
    forecaster that shrinks toward the base rate is wrong precisely where the price is
    moderately high, which is where the market is most often right.
    """
    right = (q >= 0.5) == (y >= 0.5)
    if right.sum() < 30:
        return float("nan")
    return both_wrong(P[right], y[right], iu)[2]


def simulate(P, q, band, iu, y, kind, rng):
    """One draw of shrinkage-only forecasters. Same fit, independent noise per model."""
    a, b, R = fit(P, q)
    base = a + np.outer(q, b)
    noise = np.empty_like(P)
    for bd in set(band):
        sel = band == bd
        if kind == "gauss":
            sd = R[sel].std(axis=0, ddof=1) if sel.sum() > 1 else np.zeros(P.shape[1])
            noise[sel] = rng.normal(0.0, np.maximum(sd, 1e-9), size=(int(sel.sum()), P.shape[1]))
        else:
            block = R[sel]
            noise[sel] = np.column_stack([rng.permutation(block[:, j]) for j in range(P.shape[1])])
    sim = np.clip(base + noise, 0.01, 0.99)
    _, _, Rs = fit(sim, q)
    return (mean_pair_corr(sim - q[:, None], iu), mean_pair_corr(Rs, iu), both_wrong(sim, y, iu)[2],
            both_wrong_market_right(sim, q, y, iu))


def pctile(v):
    v = np.sort(np.asarray([x for x in v if np.isfinite(x)]))
    return (float(v.mean()), float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v))])) if len(v) else (float("nan"),) * 3


def verdict(real, lo, hi):
    if not np.isfinite(real) or not np.isfinite(lo):
        return "n/a"
    if real < lo:
        return "BELOW the shrinkage-only range"
    if real <= hi:
        return "INSIDE the shrinkage-only range: shrinkage explains it"
    return "ABOVE the shrinkage-only range by %+.3f" % (real - hi)


data, cuts = hc.load("normal")
print("rows: common rows per window, forecasts view, mode normal, status in %s, primary set; "
      "intervals on real values = %d event resamples, seed %d, refit inside each resample; "
      "null range = 2.5-97.5 percentile over %d simulated draws"
      % (hc.STATUS_OK, hc.N_BOOT, hc.SEED, N_SIM))

for w in hc.WINDOWS:
    rows = data[w]
    t = hc.Table(rows, hc.ROSTER[w])
    split = np.array([r["split"] for r in rows])
    for which, sel in (("test split", split == "test"), ("both splits", np.ones(t.n, dtype=bool))):
        idx = np.flatnonzero(sel)
        P, q, y, band = t.P[idx], t.q[idx], t.y[idx], t.band[idx]
        events = [t.event[i] for i in idx]
        iu = t.iu
        base_rate = float(y.mean())
        print("=== window %s, %s: %d common rows, %d events, %d models, %d pairs, Yes base rate %.4f"
              % (w, which, len(idx), len(set(events)), t.k, len(t.pairs), base_rate))

        a, b, R = fit(P, q)
        al, bl, _ = fit(logit(P), logit(q))
        print("  shrinkage, p = a + b q on the probability scale (b < 1 is shrinkage toward a/(1-b)):")
        for j, name in enumerate(t.short):
            fp = a[j] / (1 - b[j]) if abs(1 - b[j]) > 1e-9 else float("nan")
            print("    %-16s a %+.4f  b %.4f  fixed point %.4f  resid sd %.4f | log-odds b %.4f"
                  % (name, a[j], b[j], fp, R[:, j].std(ddof=1), bl[j]))
        print("    roster mean b %.4f (probability scale), %.4f (log-odds)" % (b.mean(), bl.mean()))

        # Real values, with an event bootstrap that refits inside every resample.
        def real_stats(ii, P=P, q=q, y=y, iu=iu):
            Pi, qi, yi = P[ii], q[ii], y[ii]
            _, _, Ri = fit(Pi, qi)
            return np.array([mean_pair_corr(Pi - qi[:, None], iu), mean_pair_corr(Ri, iu),
                             both_wrong(Pi, yi, iu)[2], both_wrong_market_right(Pi, qi, yi, iu)])

        sub = hc.Table([rows[i] for i in idx], hc.ROSTER[w])
        pt, lo, hi = hc.boot(sub, real_stats)
        labels = ("mean pair r(p-q)", "mean pair r(residual)", "both-wrong ratio",
                  "both-wrong ratio, market-right rows")
        for j, lab in enumerate(labels):
            print("  real %-22s %s" % (lab + ":", hc.fmt((pt[j], lo[j], hi[j]))))

        for kind in ("gauss", "permute"):
            rng = np.random.default_rng(hc.SEED)
            draws = np.array([simulate(P, q, band, iu, y, kind, rng) for _ in range(N_SIM)])
            print("  null (%s):" % kind)
            for j, lab in enumerate(labels):
                m, l, h = pctile(draws[:, j])
                print("    %-22s mean %+.3f range [%+.3f, %+.3f]   real %+.3f -> %s"
                      % (lab, m, l, h, pt[j], verdict(pt[j], l, h)))
