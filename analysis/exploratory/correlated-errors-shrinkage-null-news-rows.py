import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-shrinkage-null-news-rows: the same shrinkage null as
correlated-errors-shrinkage-null.py, but on the news-bearing rows only.

The paper reports that a pair of models lands on the same wrong side of even money 2.33 times as
often as independent forecasters would on Window B's news-bearing rows, and 2.56 times on Window C's.
"News-bearing" means the row was served a news digest (news_available = 1 in the benchmark). Every
other wrong-together ratio in the paper is printed beside a shrinkage null -- simulated forecasters
that pull toward the base rate exactly as hard as the real models do but share nothing else, which
already miss together more often than independence because they all lean the same way. These two
ratios have no such null, so a reader cannot tell how much of the 2.33 and the 2.56 is shared
judgement and how much is shared shrinkage.

This script supplies it. Rows: the common rows of each window (every roster model forecast the row),
primary set, mode normal, restricted to news_available = 1. For each window, and for the test split
(the headline) and both splits together:

  1. the shrinkage fit p = a + b q per model on those rows, so the null carries the right lean;
  2. the real wrong-together ratio and the real mean pair correlations, with 1,000 event resamples;
  3. the null, both noise flavours -- "gauss" (normal noise with each model's own residual spread
     inside its price band) and "permute" (each model's own residuals reshuffled across rows inside
     a price band) -- over 400 simulated draws, seed 20260902.

The real ratios printed here are computed from the forecasts directly, so they also act as a check on
the 2.33 and 2.56 quoted from analysis/out/results/expC_overall.csv. Read-only on both databases.
"""
import numpy as np

import correlated_errors_common as hc
import scoring

N_SIM = 400


def fit(P, q):
    """Least squares p = a + b q per column. Returns a (k,), b (k,), residuals (n, k)."""
    qc = q - q.mean()
    var = float(qc @ qc)
    b = (qc @ (P - P.mean(axis=0))) / var if var > 0 else np.zeros(P.shape[1])
    a = P.mean(axis=0) - b * q.mean()
    return a, b, P - (a + np.outer(q, b))


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
    right = (q >= 0.5) == (y >= 0.5)
    if right.sum() < 30:
        return float("nan")
    return both_wrong(P[right], y[right], iu)[2]


def simulate(P, q, band, iu, y, kind, rng):
    """One draw of shrinkage-only forecasters: same fit, independent noise per model."""
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
    return (float(v.mean()), float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v))])) \
        if len(v) else (float("nan"),) * 3


def verdict(real, lo, hi):
    if not np.isfinite(real) or not np.isfinite(lo):
        return "n/a"
    if real < lo:
        return "BELOW the shrinkage-only range"
    if real <= hi:
        return "INSIDE the shrinkage-only range: shrinkage explains it"
    return "ABOVE the shrinkage-only range by %+.3f" % (real - hi)


con = scoring.connect()
NEWS = {rid: int(m["news_available"] or 0) for rid, m in scoring.load_moments(con).items()}
con.close()

data, cuts = hc.load("normal")
print("rows: common rows per window, forecasts view, mode normal, status in %s, primary set, "
      "restricted to news-bearing rows (news_available = 1 in benchmark.db); intervals on real "
      "values = %d event resamples, seed %d, refit inside each resample; null range = 2.5-97.5 "
      "percentile over %d simulated draws, same seed"
      % (hc.STATUS_OK, hc.N_BOOT, hc.SEED, N_SIM))

for w in hc.WINDOWS:
    rows_all = [r for r in data[w] if NEWS.get(r["row_id"], 0) == 1]
    if len(rows_all) < 60:
        print("=== window %s: only %d news-bearing common rows, skipped" % (w, len(rows_all)))
        continue
    t_all = hc.Table(rows_all, hc.ROSTER[w])
    split = np.array([r["split"] for r in rows_all])
    n_all = len(data[w])
    for which, sel in (("test split", split == "test"), ("both splits", np.ones(t_all.n, dtype=bool))):
        idx = np.flatnonzero(sel)
        rows = [rows_all[i] for i in idx]
        t = hc.Table(rows, hc.ROSTER[w])
        P, q, y, band, iu = t.P, t.q, t.y, t.band, t.iu
        print("=== window %s, %s, NEWS-BEARING ROWS: %d rows (the window has %d common rows in all, "
              "both splits together), "
              "%d events, %d models, %d pairs, Yes base rate %.4f"
              % (w, which, t.n, n_all, len(set(t.event)), t.k, len(t.pairs), float(y.mean())))
        a, b, R = fit(P, q)
        print("  shrinkage on these rows, p = a + b q (b < 1 is shrinkage toward a/(1-b)):")
        for j, name in enumerate(t.short):
            fp = a[j] / (1 - b[j]) if abs(1 - b[j]) > 1e-9 else float("nan")
            print("    %-16s a %+.4f  b %.4f  fixed point %.4f  resid sd %.4f"
                  % (name, a[j], b[j], fp, R[:, j].std(ddof=1)))
        bw, exp, ratio = both_wrong(P, y, iu)
        print("  wrong together %.4f of pairs-rows against %.4f expected under independence: ratio %.4f"
              % (bw, exp, ratio))

        def real_stats(ii, P=P, q=q, y=y, iu=iu):
            Pi, qi, yi = P[ii], q[ii], y[ii]
            _, _, Ri = fit(Pi, qi)
            return np.array([mean_pair_corr(Pi - qi[:, None], iu), mean_pair_corr(Ri, iu),
                             both_wrong(Pi, yi, iu)[2], both_wrong_market_right(Pi, qi, yi, iu)])

        pt, lo, hi = hc.boot(t, real_stats)
        labels = ("mean pair r(p-q)", "mean pair r(residual)", "both-wrong ratio",
                  "both-wrong ratio, market-right rows")
        for j, lab in enumerate(labels):
            print("  real %-38s %s" % (lab + ":", hc.fmt((pt[j], lo[j], hi[j]))))
        for kind in ("gauss", "permute"):
            rng = np.random.default_rng(hc.SEED)
            draws = np.array([simulate(P, q, band, iu, y, kind, rng) for _ in range(N_SIM)])
            print("  null (%s):" % kind)
            for j, lab in enumerate(labels):
                m_, l_, h_ = pctile(draws[:, j])
                print("    %-38s mean %+.3f range [%+.3f, %+.3f]   real %+.3f -> %s"
                      % (lab, m_, l_, h_, pt[j], verdict(pt[j], l_, h_)))
