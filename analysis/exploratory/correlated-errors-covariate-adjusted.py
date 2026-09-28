import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""correlated-errors-covariate-adjusted: once the models' shared INPUTS are controlled for, how much
of their agreement is left?

The paper's existing test (correlated-errors-shrinkage-null) fits each model's forecast on the price,
p = a + b q, correlates what is left over across model pairs, and compares that against simulated
forecasters carrying the same a, b and residual spread but independent noise. The obvious objection:
the simulated residuals are uncorrelated by construction, so ANY shared influence at
all beats that null. The alternative reading is that the models share inputs -- the same question,
the same news item, the same kind of row -- and that shared inputs alone would make their errors move
together without any shared judgement.

This script tests that reading directly. Instead of fitting each model on the price alone, it fits
each model on the price AND on the row's observable features, then correlates what is left. If the
agreement is shared inputs, the correlation should collapse once those inputs are held fixed. If it
survives, the agreement is something the observable features do not contain.

Three fits, per window, each run for every model separately (the same rows, the same design matrix,
one fitted column per model):

  (a) price only          p = a + b q. This reproduces the existing residual correlation.
  (b) price + covariates  the same, plus: a sports flag; indicators for the window's ten commonest
                          topic tags; the forecast horizon (days to settlement, entered as its
                          logarithm and as the benchmark's four horizon bands); a has-news flag; and
                          indicators for the four price bands. "Indicator" means a column that is 1
                          when the row has that feature and 0 otherwise. One level of each group of
                          indicators is dropped so the design is not redundant.
  (c) (b) + event grouping
                          plus a separate intercept for every event (an "event fixed effect": the
                          event's own mean is subtracted from every column before fitting, so only
                          differences WITHIN an event are used). This is the strongest control
                          available: it absorbs everything common to all the markets of one event,
                          including anything unobserved. It is also the most aggressive. Events with
                          a single row have no within-event variation left at all -- their residuals
                          are exactly zero for every model, which would manufacture perfect agreement
                          -- so (c) is reported on rows from multi-row events only, and the row count
                          is printed beside it. It is reported separately from the headline for that
                          reason.

Reported per window, for each fit:

  real mean pair r(residual)   the mean, over model pairs, of the correlation between the two models'
                               leftovers. Falling from (a) to (b) to (c) is the size of the shared-inputs
                               effect.
  null mean pair r(residual)   simulated forecasters carrying the SAME fitted surface (so they share
                               every covariate in the fit) and independent noise, refitted the same
                               way. Near zero by construction; printed so the fits can be checked.
  both-wrong ratio             how much more often a pair lands on the same wrong side of even money
                               than two independent forecasters with the same individual error rates
                               would. This is a property of the forecasts, not of a fit, so the real
                               value is the same under (a), (b) and (c); what changes is the NULL,
                               which under (b) and (c) is a simulated pair that shares every covariate.
                               Reported unrestricted and restricted to rows the market itself got
                               right, which is the version the paper leans on.

Noise for the simulated forecasters is drawn two ways, as in the existing script: normal noise with the
model's own residual spread within its price band ("gauss"), and the model's own residuals shuffled
across rows within a price band ("permute").

Test split is the headline; both splits are printed afterwards for reference. Intervals on real values
are 1,000 event resamples, seed 20260902, with every fit redone inside each resample. Null ranges are
the 2.5th to 97.5th percentile over 400 simulated draws, same seed. Read-only on both databases.
"""
from collections import defaultdict, Counter

import numpy as np
import correlated_errors_common as hc
import scoring

N_SIM = 400
N_TAGS = 10
EPS = 0.01
DESIGNS = ("price only", "price + covariates", "price + covariates + event grouping")


_MOMENTS = {}


def all_moments():
    """The benchmark's market-moments, loaded once (read-only)."""
    if not _MOMENTS:
        con = scoring.connect()
        _MOMENTS.update(scoring.load_moments(con))
        con.close()
    return _MOMENTS


def covariates(rows, window):
    """The covariate block for one window's rows: (matrix without intercept or q, column names)."""
    moments = all_moments()
    tags = Counter()
    for r in rows:
        for t in (moments[r["row_id"]]["topic_tags"] or []):
            tags[str(t).strip().lower()] += 1
    top = [t for t, _ in tags.most_common(N_TAGS)]
    bands = sorted({r["band"] for r in rows if r["band"]})[1:]          # one band dropped
    hbands = sorted({moments[r["row_id"]]["horizon_band"] for r in rows
                     if moments[r["row_id"]]["horizon_band"]})[1:]      # one horizon band dropped
    cols, names = [], []

    def add(name, f):
        v = np.array([float(f(moments[r["row_id"]])) for r in rows])
        if v.std() > 1e-12:
            cols.append(v)
            names.append(name)

    add("sports", lambda m: m["sports"])
    add("has news", lambda m: m["news_available"] or 0)
    add("log horizon days", lambda m: np.log10(max(m["horizon_days"] or 0.0, 0.01) + 0.5))
    for hb in hbands:
        add("horizon %s" % hb, lambda m, hb=hb: m["horizon_band"] == hb)
    for bd in bands:
        add("band %s" % bd, lambda m, bd=bd: scoring.band_of(m["q"]) == bd)
    for tg in top:
        add("tag %s" % tg, lambda m, tg=tg: tg in [str(t).strip().lower() for t in (m["topic_tags"] or [])])
    X = np.column_stack(cols) if cols else np.zeros((len(rows), 0))
    return X, names


def demean(A, groups):
    """Subtract each group's mean from every column: what including one intercept per group does."""
    out = np.array(A, dtype=float)
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    for ii in idx.values():
        out[ii] -= out[ii].mean(axis=0)
    return out


def residuals(X, P):
    """Least squares of every column of P on X; returns the leftovers, one column per model."""
    beta, *_ = np.linalg.lstsq(X, P, rcond=None)
    return P - X @ beta, beta


def mean_pair_corr(V, iu):
    if len(V) < 3:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(V, rowvar=False)
    return float(np.nanmean(c[iu])) if np.isfinite(c[iu]).any() else float("nan")


def both_wrong(P, y, iu):
    W = ((P >= 0.5) != (y[:, None] >= 0.5)).astype(float)
    n = len(P)
    if n == 0:
        return float("nan")
    both = float(((W.T @ W) / n)[iu].mean())
    m = W.mean(axis=0)
    exp = float(np.outer(m, m)[iu].mean())
    return both / exp if exp > 0 else float("nan")


def both_wrong_market_right(P, q, y, iu):
    right = (q >= 0.5) == (y >= 0.5)
    if right.sum() < 30:
        return float("nan")
    return both_wrong(P[right], y[right], iu)


def pctile(v):
    v = np.sort(np.asarray([x for x in v if np.isfinite(x)]))
    if not len(v):
        return (float("nan"),) * 3
    return float(v.mean()), float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v))])


def simulate(X, P, band, q, y, iu, kind, rng, groups=None):
    """One draw of forecasters that share the whole fitted surface and nothing else."""
    Xu = demean(X, groups) if groups is not None else X
    Pu = demean(P, groups) if groups is not None else P
    R, beta = residuals(Xu, Pu)
    base = (P - R)                       # fitted values on the probability scale
    noise = np.empty_like(P)
    for bd in set(band):
        sel = band == bd
        if kind == "gauss":
            sd = R[sel].std(axis=0, ddof=1) if sel.sum() > 1 else np.zeros(P.shape[1])
            noise[sel] = rng.normal(0.0, np.maximum(sd, 1e-9), size=(int(sel.sum()), P.shape[1]))
        else:
            block = R[sel]
            noise[sel] = np.column_stack([rng.permutation(block[:, j]) for j in range(P.shape[1])])
    sim = np.clip(base + noise, EPS, 1 - EPS)
    Su = demean(sim, groups) if groups is not None else sim
    Rs, _ = residuals(Xu, Su)
    return (mean_pair_corr(Rs, iu), both_wrong(sim, y, iu), both_wrong_market_right(sim, q, y, iu))


data, cuts = hc.load("normal")
print("rows: common rows per window (every roster model forecast the row), forecasts view, mode normal, "
      "status in %s, primary set. Covariates come from benchmark.db (topic_tags, news_available, "
      "resolved_at minus forecast_ts, price band); event grouping uses source_event_id. Intervals on "
      "real values = %d event resamples, seed %d, every fit redone inside each resample; null range = "
      "2.5-97.5 percentile over %d simulated draws, same seed."
      % (hc.STATUS_OK, hc.N_BOOT, hc.SEED, N_SIM))

for w in hc.WINDOWS:
    rows_w = data[w]
    Xcov_all, cov_names = covariates(rows_w, w)
    t_all = hc.Table(rows_w, hc.ROSTER[w])
    split_all = np.array([r["split"] for r in rows_w])
    for which, keep in (("test split", split_all == "test"), ("both splits", np.ones(len(rows_w), bool))):
        sel = np.flatnonzero(keep)
        rows = [rows_w[i] for i in sel]
        t = hc.Table(rows, hc.ROSTER[w])
        Xcov = Xcov_all[sel]
        P, q, y, band, iu = t.P, t.q, t.y, t.band, t.iu
        ev = np.array(t.event, dtype=object)
        one = np.ones((t.n, 1))
        X_a = np.column_stack([one, q])
        X_b = np.column_stack([one, q, Xcov]) if Xcov.size else X_a
        # Event grouping: rows whose event holds more than one of these rows.
        size = Counter(t.event)
        multi = np.array([size[e] > 1 for e in t.event])
        print("=== window %s, %s: %d common rows, %d events, %d models, %d pairs; %d rows sit in "
              "multi-row events (%d events), which is where the event-grouping fit is run"
              % (w, which, t.n, len(set(t.event)), t.k, len(t.pairs), int(multi.sum()),
                 sum(1 for e, c in size.items() if c > 1)))
        print("  covariates (%d columns besides intercept and price): %s" % (len(cov_names), ", ".join(cov_names)))

        def stat(idx, X_a=X_a, X_b=X_b, P=P, ev=ev, multi=multi, iu=iu, q=q, y=y):
            out = []
            out.append(mean_pair_corr(residuals(X_a[idx], P[idx])[0], iu))
            out.append(mean_pair_corr(residuals(X_b[idx], P[idx])[0], iu))
            m = idx[multi[idx]]
            if len(m) > X_b.shape[1] + 5:
                g = ev[m]
                out.append(mean_pair_corr(residuals(demean(X_b[m], g), demean(P[m], g))[0], iu))
            else:
                out.append(float("nan"))
            out.append(mean_pair_corr(P[idx] - q[idx][:, None], iu))
            out.append(both_wrong(P[idx], y[idx], iu))
            out.append(both_wrong_market_right(P[idx], q[idx], y[idx], iu))
            return np.array(out, dtype=float)

        pt, lo, hi = hc.boot(t, stat)
        print("  real mean pair r(departure p - q), for reference: %s" % hc.fmt((pt[3], lo[3], hi[3])))
        for j, name in enumerate(DESIGNS):
            extra = "" if j < 2 else "   (multi-row-event rows only)"
            print("  real mean pair r(residual), %-36s %s%s" % (name + ":", hc.fmt((pt[j], lo[j], hi[j])), extra))
        if np.isfinite(pt[1]):
            print("  FALL from price-only to price + covariates: %+.3f (%.1f%% of the price-only value)"
                  % (pt[1] - pt[0], 100 * (pt[1] - pt[0]) / pt[0] if pt[0] else float("nan")))
        if np.isfinite(pt[2]):
            print("  FALL from price-only to + event grouping:   %+.3f (%.1f%% of the price-only value)"
                  % (pt[2] - pt[0], 100 * (pt[2] - pt[0]) / pt[0] if pt[0] else float("nan")))
        print("  real both-wrong ratio:                    %s" % hc.fmt((pt[4], lo[4], hi[4])))
        print("  real both-wrong ratio, market-right rows:  %s" % hc.fmt((pt[5], lo[5], hi[5])))

        for j, name in enumerate(DESIGNS):
            if j == 0:
                Xd, Pd, qd, yd, bd_, ivd, grp = X_a, P, q, y, band, iu, None
            elif j == 1:
                Xd, Pd, qd, yd, bd_, ivd, grp = X_b, P, q, y, band, iu, None
            else:
                m = np.flatnonzero(multi)
                if len(m) <= X_b.shape[1] + 5:
                    continue
                Xd, Pd, qd, yd, bd_, ivd, grp = X_b[m], P[m], q[m], y[m], band[m], iu, ev[m]
            real_r = pt[j]
            real_bw = both_wrong(Pd, yd, ivd)
            real_bwr = both_wrong_market_right(Pd, qd, yd, ivd)
            where = " on the multi-row-event rows only, so its 'real' values differ from the headline" if j == 2 else ""
            print("  -- null under %s (simulated forecasters sharing this whole fit, independent noise)%s"
                  % (name, where))
            for kind in ("gauss", "permute"):
                rng = np.random.default_rng(hc.SEED)
                draws = np.array([simulate(Xd, Pd, bd_, qd, yd, ivd, kind, rng, grp) for _ in range(N_SIM)])
                labs = ("mean pair r(residual)", "both-wrong ratio", "both-wrong ratio, market-right")
                reals = (real_r, real_bw, real_bwr)
                for c, lab in enumerate(labs):
                    m_, l_, h_ = pctile(draws[:, c])
                    verdict = ("ABOVE by %+.3f" % (reals[c] - h_)) if np.isfinite(reals[c]) and np.isfinite(h_) \
                        and reals[c] > h_ else ("INSIDE the null range" if np.isfinite(reals[c]) and np.isfinite(l_)
                                                and reals[c] >= l_ else "BELOW / n/a")
                    print("     %-9s %-32s null mean %+.3f range [%+.3f, %+.3f]   real %+.3f -> %s"
                          % (kind, lab, m_, l_, h_, reals[c], verdict))

# ---------------------------------------------------------------------------
# Same-rows comparator.
#
# The event-grouping fit above runs only on the rows that sit in multi-row events, so its residual
# correlation is not comparable with the price-only and price-plus-covariates figures printed above,
# which use every common row. A reader cannot tell whether a fall is the event control or the smaller
# subset. This section prints all three fits on EXACTLY the multi-row-event rows, so the comparison is
# like for like.
# ---------------------------------------------------------------------------
print("")
print("=== SAME-ROWS COMPARATOR: all three fits on the multi-row-event rows only")
print("    (the rows the event-grouping fit uses, so the three numbers differ only by the fit)")

for w in hc.WINDOWS:
    rows_w = data[w]
    Xcov_all, _ = covariates(rows_w, w)
    split_all = np.array([r["split"] for r in rows_w])
    for which, keep in (("test split", split_all == "test"), ("both splits", np.ones(len(rows_w), bool))):
        sel = np.flatnonzero(keep)
        rows = [rows_w[i] for i in sel]
        t = hc.Table(rows, hc.ROSTER[w])
        size = Counter(t.event)
        m = np.flatnonzero(np.array([size[e] > 1 for e in t.event]))
        if len(m) < 50:
            print("  window %s, %s: too few multi-row-event rows" % (w, which))
            continue
        sub_rows = [rows[i] for i in m]
        sub = hc.Table(sub_rows, hc.ROSTER[w])
        Xcov = Xcov_all[sel][m]
        one = np.ones((sub.n, 1))
        X_a = np.column_stack([one, sub.q])
        X_b = np.column_stack([one, sub.q, Xcov]) if Xcov.size else X_a
        ev = np.array(sub.event, dtype=object)
        P = sub.P
        print("  window %s, %s: %d rows in %d multi-row events, %d models, %d pairs"
              % (w, which, sub.n, len(set(sub.event)), sub.k, len(sub.pairs)))

        def stat(idx, X_a=X_a, X_b=X_b, P=P, ev=ev, iu=sub.iu):
            out = [mean_pair_corr(residuals(X_a[idx], P[idx])[0], iu),
                   mean_pair_corr(residuals(X_b[idx], P[idx])[0], iu)]
            g = ev[idx]
            if len(idx) > X_b.shape[1] + 5:
                out.append(mean_pair_corr(residuals(demean(X_b[idx], g), demean(P[idx], g))[0], iu))
            else:
                out.append(float("nan"))
            return np.array(out, dtype=float)

        pt, lo, hi = hc.boot(sub, stat)
        for j, name in enumerate(DESIGNS):
            print("    real mean pair r(residual), %-36s %s" % (name + ":", hc.fmt((pt[j], lo[j], hi[j]))))
        if np.isfinite(pt[2]) and pt[0]:
            print("    FALL on these same rows, price only -> event grouping: %+.3f (%.1f%% of the "
                  "price-only value on these rows)" % (pt[2] - pt[0], 100 * (pt[2] - pt[0]) / pt[0]))
            print("    FALL on these same rows, price + covariates -> event grouping: %+.3f"
                  % (pt[2] - pt[1]))
