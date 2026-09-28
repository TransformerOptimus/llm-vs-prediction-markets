"""Shared loader for the horizon angle. Read-only. Reuses analysis/scoring.py."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import os, sys, sqlite3, json, math, random, statistics as st
from collections import defaultdict
from datetime import datetime, timezone
REPO = _paths.REPO
import scoring as S

RUNS_DB = os.path.join(REPO, "harness", "runs", "runs.db")
EXCLUDE = set(_roster.NOT_SCORED)  # the probe-validation model and the fixture
MODELS = _roster.SCORED            # every scored model in the roster
SHORT = _roster.SHORT              # display names from harness/models.json
SEED = 20260902
N_BOOT = 1000


def parse_ts(s):
    s = s.replace("Z", "+00:00").replace(" ", "T")
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def load_all(mode="normal"):
    """Scored rows for every model in `mode`, primary set only, with horizon fields attached."""
    con = S.connect()
    moments = S.load_moments(con)
    cuts = S.load_or_compute_cuts(moments)
    by_key = {m["moment_key"]: m for m in moments.values()}
    r = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    r.row_factory = sqlite3.Row
    rows = []
    for f in r.execute("SELECT model, mode, window, moment_key, parsed_probability, status FROM forecasts WHERE mode=?", (mode,)):
        if f["model"] in EXCLUDE or f["parsed_probability"] is None or f["status"] not in ("ok", "ok_after_retry"):
            continue
        m = by_key.get(f["moment_key"])
        if m is None or not m["primary"]:
            continue
        row = S.score_forecast(m, float(f["parsed_probability"]), SHORT[f["model"]], cuts)
        ft = parse_ts(m["forecast_ts"])
        row["hour_utc"] = ft.hour
        row["dow"] = ft.weekday()
        row["forecast_date"] = ft.date().isoformat()
        row["abs_err_model"] = abs(row["p"] - row["outcome_yes"])
        row["p_minus_q"] = row["p"] - row["q"]
        row["abs_gap"] = abs(row["p"] - row["q"])
        row["log_horizon"] = math.log(max(row["horizon_days"], 1 / 24.0)) if row["horizon_days"] is not None else None
        rows.append(row)
    return rows, moments, cuts


def bm(rows, key, cluster="event"):
    return S.boot_mean(rows, key, n=N_BOOT, seed=SEED, cluster=cluster)


def bd(a, b, key, cluster="event"):
    return S.boot_diff(a, b, key, n=N_BOOT, seed=SEED, cluster=cluster)


# ---- fast cluster bootstrap for OLS coefficients (numpy, per-event sufficient statistics) ----
import numpy as np
BN = S.BAND_NAMES


def design_matrix(rs, ctrl=True, xkey="log_horizon"):
    X = []
    for r in rs:
        x = [1.0, r[xkey]]
        if ctrl:
            x += [1.0 if r["band"] == b else 0.0 for b in BN[1:]]
            x += [r["log_depth"] if r["log_depth"] is not None else 0.0, float(r["sports"])]
        X.append(x)
    return np.array(X)


def boot_coef(rs, y, ctrl=True, xkey="log_horizon", n=N_BOOT, seed=SEED, coef=1):
    """Coefficient `coef` (default: the x term) of OLS y ~ x [+ price band + log depth + sports],
    with a 95% interval from resampling events."""
    rs = [r for r in rs if r.get(y) is not None and r.get(xkey) is not None]
    X = design_matrix(rs, ctrl, xkey)
    Y = np.array([r[y] for r in rs], dtype=float)
    ev = {}
    idx = np.array([ev.setdefault(r["event_id"], len(ev)) for r in rs])
    k = X.shape[1]; E = len(ev)
    XtX = np.zeros((E, k, k)); XtY = np.zeros((E, k))
    for i in range(len(rs)):
        XtX[idx[i]] += np.outer(X[i], X[i]); XtY[idx[i]] += X[i] * Y[i]
    def solve(w):
        A = np.tensordot(w, XtX, 1); b = w @ XtY
        return np.linalg.lstsq(A, b, rcond=None)[0][coef]
    pt = solve(np.ones(E))
    rng = np.random.default_rng(seed)
    bs = np.sort([solve(np.bincount(rng.integers(0, E, E), minlength=E).astype(float)) for _ in range(n)])
    return pt, bs[int(0.025 * n)], bs[int(0.975 * n)], len(rs)


def boot_mean_fast(rs, key, n=N_BOOT, seed=SEED):
    rs = [r for r in rs if r.get(key) is not None]
    if not rs:
        return float("nan"), float("nan"), float("nan"), 0
    ev = {}
    idx = np.array([ev.setdefault(r["event_id"], len(ev)) for r in rs])
    v = np.array([r[key] for r in rs], dtype=float)
    E = len(ev)
    sums = np.bincount(idx, weights=v, minlength=E); cnt = np.bincount(idx, minlength=E).astype(float)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, E, E), minlength=E).astype(float)
        bs.append((w @ sums) / (w @ cnt))
    bs = np.sort(bs)
    return v.mean(), bs[int(0.025 * n)], bs[int(0.975 * n)], len(rs)


def boot_diff_fast(a, b, key, n=N_BOOT, seed=SEED):
    """mean(a) - mean(b); both resampled jointly by event (events shared across a and b move together)."""
    a = [r for r in a if r.get(key) is not None]; b = [r for r in b if r.get(key) is not None]
    ev = {}
    ia = np.array([ev.setdefault(r["event_id"], len(ev)) for r in a]); ib = np.array([ev.setdefault(r["event_id"], len(ev)) for r in b])
    E = len(ev)
    va = np.array([r[key] for r in a], float); vb = np.array([r[key] for r in b], float)
    sa = np.bincount(ia, weights=va, minlength=E); ca = np.bincount(ia, minlength=E).astype(float)
    sb = np.bincount(ib, weights=vb, minlength=E); cb = np.bincount(ib, minlength=E).astype(float)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, E, E), minlength=E).astype(float)
        da, db = w @ ca, w @ cb
        if da == 0 or db == 0: continue
        bs.append((w @ sa) / da - (w @ sb) / db)
    bs = np.sort(bs)
    return va.mean() - vb.mean(), bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))], len(a), len(b)
