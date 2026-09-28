"""Shared loader and statistics for the correlated-errors scripts. Read-only on both databases.

Rows: per window, the primary-set market-moments that EVERY model in that window's roster forecast
(the "common rows"), from the forecasts view in one mode (normal unless a script says otherwise), status
in ('ok', 'ok_after_retry'), parsed_probability not null, the probe-validation model excluded, joined to
the benchmark by moment_key. Each row carries every roster model's probability p, the market probability
q, the outcome y, the event id (bootstrap cluster), price band, depth bucket and sports flag.

Pair statistics are Pearson correlations between two models over the common rows of
  error   p - y        (the error correlation)
  p       p            (raw probability)
  dev     p - q        (departure from the market)
and "wrong" means on the wrong side of 0.5. Intervals: 95% bootstrap
resampling events, 1,000 resamples, seed 20260902, the statistic recomputed on every resample.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import itertools
import random
from collections import defaultdict

import numpy as np
import scoring

STATUS_OK = ("ok", "ok_after_retry")
EXCLUDE = _roster.VALIDATION       # the probe-validation model, never scored
FIG_DIR = _paths.FIG_DIR
SEED = 20260902
N_BOOT = 1000
SHORT = _roster.SHORT              # display names from harness/models.json
ROSTER = _roster.ROSTER            # window -> the scored models the roster runs there
WINDOWS = ("A", "B", "C")
KINDS = ("error", "p", "dev")


def load(mode="normal"):
    """window -> list of common rows {row_id, event_id, q, y, band, bucket, sports, split, p: {model: p}}."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    con.close()
    bykey = {m["moment_key"]: m for m in moments.values()}
    rc = _paths.runs_con()
    per = defaultdict(dict)   # row_id -> model -> p
    sql = ("SELECT model, moment_key, parsed_probability p FROM forecasts WHERE mode = ? AND status IN (%s) "
           "AND parsed_probability IS NOT NULL AND model != ?" % ",".join("?" * len(STATUS_OK)))
    for r in rc.execute(sql, (mode, *STATUS_OK, EXCLUDE)):
        m = bykey.get(r["moment_key"])
        if m is None or not m["primary"]:
            continue
        per[m["row_id"]][r["model"]] = float(r["p"])
    rc.close()
    out = defaultdict(list)
    for rid, ps in per.items():
        m = moments[rid]
        roster = ROSTER.get(m["window"])
        if roster is None or any(k not in ps for k in roster):
            continue
        if _paths.test_split_only() and m["split"] != "test":
            continue
        out[m["window"]].append({"row_id": rid, "event_id": m["event_id"], "q": m["q"], "y": float(m["outcome_yes"]),
                                 "band": scoring.band_of(m["q"]), "bucket": scoring.bucket_of(m, cuts),
                                 "sports": int(m["sports"]), "split": m["split"], "p": {k: ps[k] for k in roster}})
    for w in out:
        out[w].sort(key=lambda r: r["row_id"])
    return dict(out), cuts


def pairs(roster):
    return list(itertools.combinations(range(len(roster)), 2))


class Table:
    """The common rows of one window as arrays: P (rows x models), q, y, and the wrong-side flags."""

    def __init__(self, rows, roster):
        self.roster = roster
        self.short = [SHORT.get(m, m) for m in roster]
        self.n, self.k = len(rows), len(roster)
        self.P = np.array([[r["p"][m] for m in roster] for r in rows], dtype=float)
        self.q = np.array([r["q"] for r in rows], dtype=float)
        self.y = np.array([r["y"] for r in rows], dtype=float)
        self.sports = np.array([r["sports"] for r in rows], dtype=bool)
        self.band = np.array([r["band"] or "" for r in rows])
        self.event = [r["event_id"] for r in rows]
        self.wrong = (self.P >= 0.5) != (self.y[:, None] >= 0.5)          # rows x models
        self.mkt_wrong = (self.q >= 0.5) != (self.y >= 0.5)
        self.pairs = pairs(roster)
        self.iu = np.triu_indices(self.k, 1)

    def values(self, kind, idx):
        if kind == "error":
            return self.P[idx] - self.y[idx][:, None]
        if kind == "p":
            return self.P[idx]
        return self.P[idx] - self.q[idx][:, None]

    def corr(self, kind, idx):
        """Pearson correlation matrix between models over rows idx (nan where a column is constant)."""
        v = self.values(kind, idx)
        if len(idx) < 2:
            return np.full((self.k, self.k), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.corrcoef(v, rowvar=False)

    def pair_vals(self, kind, idx):
        """The k(k-1)/2 pair correlations in itertools.combinations order."""
        return self.corr(kind, idx)[self.iu]

    def mean_pair_r(self, kind, idx):
        v = self.pair_vals(kind, idx)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def boot(table, stat, n=N_BOOT, seed=SEED):
    """Cluster bootstrap over events of stat(idx) -> float or 1-d array. Returns (point, lo, hi), each
    shaped like the statistic; resamples where a component is nan are left out of that component."""
    g = defaultdict(list)
    for i, e in enumerate(table.event):
        g[e].append(i)
    ids = list(g)
    all_idx = np.arange(table.n)
    point = np.atleast_1d(np.asarray(stat(all_idx), dtype=float))
    rng = random.Random(seed)
    draws = np.empty((n, point.size))
    for b in range(n):
        idx = np.array([i for e in rng.choices(ids, k=len(ids)) for i in g[e]], dtype=np.intp)
        draws[b] = np.atleast_1d(np.asarray(stat(idx), dtype=float))
    lo = np.empty(point.size); hi = np.empty(point.size)
    for j in range(point.size):
        v = np.sort(draws[np.isfinite(draws[:, j]), j])
        lo[j] = v[int(0.025 * len(v))] if len(v) else float("nan")
        hi[j] = v[int(0.975 * len(v))] if len(v) else float("nan")
    if point.size == 1:
        return float(point[0]), float(lo[0]), float(hi[0])
    return point, lo, hi


def fmt(t, d=3):
    f = "%%+.%df" % d
    return (f + " [" + f + ", " + f + "]") % tuple(t)
