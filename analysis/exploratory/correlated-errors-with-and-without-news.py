import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-with-and-without-news: does the models' agreement fall when they are given news?

The paper reads the models' agreement (residual correlation after the shrinkage fit p = a + b q, and
the wrong-together ratio on rows the market got right) as a sign that their signals are common to them.
A simpler explanation competes: the price carries information none of the models was given, so every
model's departure from the price carries the same missing term, and they agree because they are all
missing the same facts. Call it shared missing information.

Every news-bearing row was asked twice: once with the news digest (mode normal) and once with it
removed (mode memory_probe). Same model, question, moment and outcome; only the news differs. If
shared missing information drives the agreement, adding news should lower it. This script measures the
agreement in each pass on exactly the same rows and reports the change.

Rows: the common rows of each window (every roster model has an ok forecast in BOTH passes), primary
set, news_available = 1. Statistics per pass: mean pair r(p - q), mean pair r(residual) after the
per-model fit p = a + b q refitted within the pass, the wrong-together ratio on all rows, and on rows
the market got right. Change = with news minus without. Intervals: 1,000 event resamples, seed
20260902, both passes resampled together so the change is paired. Test split (the headline) and both
splits together. Read-only on both databases.

What this can and cannot show: a fall means part of the agreement was missing information that our
news supplied. No fall means the news we supplied did not reduce it; it cannot rule out shared
ignorance of facts the news did not carry (line-ups, injuries, late line movement).
"""
import numpy as np

import correlated_errors_common as hc
import scoring


def fit_resid(P, q):
    qc = q - q.mean()
    var = float(qc @ qc)
    b = (qc @ (P - P.mean(axis=0))) / var if var > 0 else np.zeros(P.shape[1])
    a = P.mean(axis=0) - b * q.mean()
    return P - (a + np.outer(q, b))


def mean_pair_corr(V, iu):
    if len(V) < 3:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(V, rowvar=False)
    return float(np.nanmean(c[iu])) if np.isfinite(c[iu]).any() else float("nan")


def ratio(P, y, iu):
    W = ((P >= 0.5) != (y[:, None] >= 0.5)).astype(float)
    if len(W) == 0:
        return float("nan")
    both = float(((W.T @ W) / len(W))[iu].mean())
    m = W.mean(axis=0)
    exp = float(np.outer(m, m)[iu].mean())
    return both / exp if exp > 0 else float("nan")


def stats(P, q, y, iu):
    right = (q >= 0.5) == (y >= 0.5)
    return np.array([mean_pair_corr(P - q[:, None], iu), mean_pair_corr(fit_resid(P, q), iu),
                     ratio(P, y, iu), ratio(P[right], y[right], iu) if right.sum() >= 30 else float("nan")])


con = scoring.connect()
NEWS = {rid: int(m["news_available"] or 0) for rid, m in scoring.load_moments(con).items()}
con.close()

with_news, _ = hc.load("normal")
without, _ = hc.load("memory_probe")
LABELS = ("mean pair r(p-q)", "mean pair r(residual)", "wrong-together ratio, all rows",
          "wrong-together ratio, market-right rows")

print("rows: common rows per window with an ok forecast from every roster model in BOTH passes (normal = "
      "with news, memory_probe = news removed), primary set, news_available = 1; intervals = %d event "
      "resamples, seed %d, passes resampled together" % (hc.N_BOOT, hc.SEED))

for w in hc.WINDOWS:
    wo = {r["row_id"]: r for r in without.get(w, [])}
    rows = [r for r in with_news.get(w, []) if NEWS.get(r["row_id"], 0) == 1 and r["row_id"] in wo]
    if len(rows) < 60:
        print("=== window %s: only %d paired news rows, skipped" % (w, len(rows)))
        continue
    split = np.array([r["split"] for r in rows])
    for which, sel in (("test split", split == "test"), ("both splits", np.ones(len(rows), dtype=bool))):
        rs = [rows[i] for i in np.flatnonzero(sel)]
        t = hc.Table(rs, hc.ROSTER[w])
        P1 = t.P
        P0 = np.array([[wo[r["row_id"]]["p"][m] for m in hc.ROSTER[w]] for r in rs], dtype=float)
        q, y, iu = t.q, t.y, t.iu
        print("=== window %s, %s: %d paired news rows, %d events, %d models"
              % (w, which, t.n, len(set(t.event)), t.k))

        def stat(ii):
            s1, s0 = stats(P1[ii], q[ii], y[ii], iu), stats(P0[ii], q[ii], y[ii], iu)
            return np.concatenate([s1, s0, s1 - s0])

        pt, lo, hi = hc.boot(t, stat)
        k = len(LABELS)
        for j, lab in enumerate(LABELS):
            print("  %-40s with news %s | without %s | change %s"
                  % (lab, hc.fmt((pt[j], lo[j], hi[j])), hc.fmt((pt[k + j], lo[k + j], hi[k + j])),
                     hc.fmt((pt[2 * k + j], lo[2 * k + j], hi[2 * k + j]))))
