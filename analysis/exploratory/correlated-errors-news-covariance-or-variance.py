import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-news-covariance-or-variance: when news is added and the models' residual
correlation falls, is that because they share less, or because each became noisier?

correlated-errors-with-and-without-news.py finds that adding news lowers the mean pairwise residual
correlation (after each model's fit p = a + b q) by 0.036 to 0.055. The paper reads that as some of
the agreement being information none of the models had. A correlation can fall for two different
reasons, though:

  shared part falls   the pairwise residual COVARIANCE drops: the models' departures from the price
                      move together less once they have the news (the paper's reading);
  own part rises      each model's residual VARIANCE rises: the news makes each model react in its
                      own way, adding idiosyncratic noise, while what they share is unchanged.

This script splits the change. Per pass it reports the mean pairwise residual covariance, the mean
residual standard deviation and the mean pairwise correlation. It then recomputes the with-news
correlation twice, swapping in one ingredient from the without-news pass at a time:

  covariance only   with-news covariances over without-news standard deviations: the change that
                    comes from the shared part alone;
  variance only     without-news covariances over with-news standard deviations: the change that
                    comes from the models' own spreads alone.

Rows: as correlated-errors-with-and-without-news.py -- common rows per window with an ok forecast
from every roster model in BOTH passes, primary set, news_available = 1. Each pass is refitted on its
own. Intervals: 1,000 event resamples, seed 20260902, both passes resampled together. Test split (the
headline) and both splits. Read-only on both databases.
"""
import numpy as np

import correlated_errors_common as hc
import scoring


def resid(P, q):
    qc = q - q.mean()
    var = float(qc @ qc)
    b = (qc @ (P - P.mean(axis=0))) / var if var > 0 else np.zeros(P.shape[1])
    a = P.mean(axis=0) - b * q.mean()
    return P - (a + np.outer(q, b))


def parts(R, iu):
    C = np.cov(R, rowvar=False)
    sd = np.sqrt(np.diag(C))
    return C, sd


def mean_r(C, sd, iu):
    return float(np.mean(C[iu] / np.outer(sd, sd)[iu]))


con = scoring.connect()
NEWS = {rid: int(m["news_available"] or 0) for rid, m in scoring.load_moments(con).items()}
con.close()

with_news, _ = hc.load("normal")
without, _ = hc.load("memory_probe")
LABELS = ("mean pair residual covariance (x1000)", "mean residual sd", "mean pair r(residual)",
          "r, covariance only swapped in", "r, variance only swapped in")

print("rows: common rows per window with an ok forecast from every roster model in BOTH passes, "
      "primary set, news_available = 1; each pass refitted p = a + b q on its own; intervals = %d "
      "event resamples, seed %d, passes resampled together" % (hc.N_BOOT, hc.SEED))
print("reading: 'r, covariance only' is the with-news correlation if only the shared part had "
      "changed; 'r, variance only' is the with-news correlation if only the models' own spreads had "
      "changed. Compare each with the without-news r.")

for w in hc.WINDOWS:
    wo = {r["row_id"]: r for r in without.get(w, [])}
    rows = [r for r in with_news.get(w, []) if NEWS.get(r["row_id"], 0) == 1 and r["row_id"] in wo]
    if len(rows) < 60:
        continue
    split = np.array([r["split"] for r in rows])
    for which, sel in (("test split", split == "test"), ("both splits", np.ones(len(rows), dtype=bool))):
        rs = [rows[i] for i in np.flatnonzero(sel)]
        t = hc.Table(rs, hc.ROSTER[w])
        P1 = t.P
        P0 = np.array([[wo[r["row_id"]]["p"][m] for m in hc.ROSTER[w]] for r in rs], dtype=float)
        q, iu = t.q, t.iu
        print("=== window %s, %s: %d paired news rows, %d events, %d models"
              % (w, which, t.n, len(set(t.event)), t.k))

        def stat(ii):
            C1, s1 = parts(resid(P1[ii], q[ii]), iu)
            C0, s0 = parts(resid(P0[ii], q[ii]), iu)
            one = [1000 * float(np.mean(C1[iu])), float(np.mean(s1)), mean_r(C1, s1, iu),
                   mean_r(C1, s0, iu), mean_r(C0, s1, iu)]
            zero = [1000 * float(np.mean(C0[iu])), float(np.mean(s0)), mean_r(C0, s0, iu),
                    mean_r(C0, s0, iu), mean_r(C0, s0, iu)]
            return np.array(one + zero + [a - b for a, b in zip(one, zero)])

        pt, lo, hi = hc.boot(t, stat)
        k = len(LABELS)
        for j, lab in enumerate(LABELS):
            if j < 3:
                print("  %-38s with news %s | without %s | change %s"
                      % (lab, hc.fmt((pt[j], lo[j], hi[j]), 4), hc.fmt((pt[k + j], lo[k + j], hi[k + j]), 4),
                         hc.fmt((pt[2 * k + j], lo[2 * k + j], hi[2 * k + j]), 4)))
            else:
                print("  %-38s %s | change from without-news r %s"
                      % (lab, hc.fmt((pt[j], lo[j], hi[j]), 4), hc.fmt((pt[2 * k + j], lo[2 * k + j], hi[2 * k + j]), 4)))
