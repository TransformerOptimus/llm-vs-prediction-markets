"""Shared loading and test code for the four convergence scripts (read-only on both databases).

Rows (as the harness records them):
  forecasts view, one mode (normal unless a script says otherwise), status in ('ok', 'ok_after_retry'),
  parsed_probability not null, the probe-validation model excluded, joined to the benchmark by moment_key,
  primary analysis set = m["primary"] from analysis/scoring.load_moments (both splits, no gate, no quarantine).

Price-path quantities, exactly as analysis/results.py and analysis/price_paths.py define them:
  q            market probability (scoring.market_prob: mid on Window A when present, else price_yes)
  window end   the earlier of 72 hours after the forecast and the close (price_paths.last_price_in_window)
  last         the last price-path point at or before the window end; rows without one are "unrated"
  disagreement |p - q| >= 0.10
  toward model |last - p| <= |q - p| - 0.02;  toward outcome: |last - y| <= |q - y| - 0.02
  settled      the close falls inside the window (close <= forecast + 72 h)

Shuffled baselines: p permuted across a named pool of rows with random.Random(20260902), the whole test
(disagreement filter, then drift) re-applied to the permuted p, the rate averaged over the draws.
  rule baseline               pool = every admitted (rated) row of one model and window (the rule in results.py)
  disagreement-only baseline  pool = the disagreement rows only
Intervals: 95% bootstrap resampling events (m["event_id"]), 1,000 resamples, seed 20260902, through
scoring.boot_mean / scoring.boot_diff for plain means, and the same loop written here (boot_excess) for
"real rate minus shuffled rate", which is not a plain mean.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import random
import statistics as st
from collections import defaultdict
from datetime import timedelta

import numpy as np
import scoring, price_paths  # noqa: E401

SEED = 20260902
N_BOOT = 1000            # event resamples for every interval
N_PERM = 1000            # permutation draws for a shuffled baseline (a script may state its own count)
DISAGREE_MIN, DRIFT_MIN, HOURS = 0.10, 0.02, 72.0
EXTREME_TOL = 0.01       # "last price at 0 or 1": within 0.01 of the extreme (Kalshi settlement prints 0.995 / 0.005)
STATUS_OK = ("ok", "ok_after_retry")
EXCLUDE = _roster.VALIDATION       # the probe-validation model, never scored
SHORT = _roster.SHORT              # display names from harness/models.json
BRIDGE = _roster.BRIDGE            # the three-window set (roster label "bridge"), from harness/models.json
ROSTER = _roster.ROSTER            # window -> the scored models the roster runs there
WINDOWS = ("A", "B", "C")


def load_all(mode="normal"):
    """moments (row_id -> moment), last (row_id -> last in-window price or None), fc (model -> row_id -> p)."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    raw = defaultdict(list)
    for r in con.execute("SELECT row_id, ts, price_yes FROM price_paths WHERE price_yes IS NOT NULL ORDER BY row_id, ts"):
        raw[r["row_id"]].append({"ts": r["ts"], "price_yes": r["price_yes"]})
    last = {}
    for rid, m in moments.items():
        path = [p for p in raw.get(rid, []) if p["ts"] >= m["forecast_ts"]]     # points at or after the forecast
        last[rid] = price_paths.last_price_in_window(path, m, HOURS)
    con.close()
    bykey = {m["moment_key"]: m for m in moments.values()}
    rcon = _paths.runs_con()
    fc = defaultdict(dict)
    sql = ("SELECT model, moment_key, parsed_probability FROM forecasts WHERE mode = ? AND status IN (%s) "
           "AND parsed_probability IS NOT NULL AND model != ?" % ",".join("?" * len(STATUS_OK)))
    for r in rcon.execute(sql, (mode, *STATUS_OK, EXCLUDE)):
        m = bykey.get(r["moment_key"])
        if m is None:
            continue
        fc[r["model"]][m["row_id"]] = float(r["parsed_probability"])
    rcon.close()
    return moments, last, fc


def settled(m):
    """The market closed inside the 72-hour window (close time at or before forecast + 72 h)."""
    c = price_paths.close_time(m)
    return int(bool(c) and price_paths._iso(c) <= price_paths._iso(m["forecast_ts"]) + timedelta(hours=HOURS))


def _sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def admitted_rows(moments, last, fc, model, window):
    """Every primary row of `window` with a forecast by `model` and a rated last price (unrated rows are
    dropped, as results.py drops them; their count is returned separately)."""
    out, unrated = [], 0
    for rid, p in fc[model].items():
        m = moments[rid]
        if m["window"] != window or not m["primary"]:
            continue
        if last[rid] is None:
            unrated += 1
            continue
        q, y, lp = m["q"], float(m["outcome_yes"]), last[rid]
        out.append({"row_id": rid, "event_id": m["event_id"], "model": model, "short": SHORT.get(model, model),
                    "window": window, "p": p, "q": q, "y": y, "last": lp, "m": m,
                    "settled": settled(m), "disagree": int(abs(p - q) >= DISAGREE_MIN)})
    out.sort(key=lambda r: r["row_id"])
    return out, unrated


def add_fields(r):
    """The derived price-path quantities of one disagreement row."""
    p, q, y, lp = r["p"], r["q"], r["y"], r["last"]
    r["toward_model"] = int(abs(lp - p) <= abs(q - p) - DRIFT_MIN)
    r["toward_outcome"] = int(abs(lp - y) <= abs(q - y) - DRIFT_MIN)
    r["on_outcome_side"] = int((p - q) * (y - q) > 0)
    # "at least halfway from the market to the outcome": with last = y the toward-model rule |y - p| <= |q - p| - 0.02 is exactly
    # (model on the outcome's side) and |p - q| >= (|y - q| + 0.02) / 2, so the 0.02 drift floor stays in the halfway rule
    r["halfway"] = int(abs(p - q) >= (abs(y - q) + DRIFT_MIN) / 2.0)
    r["right_and_halfway"] = int(r["on_outcome_side"] and r["halfway"])
    r["last_at_extreme"] = int(lp <= EXTREME_TOL or lp >= 1.0 - EXTREME_TOL)
    r["move_to_model"] = (lp - q) * _sign(p - q)
    r["move_to_half"] = (lp - q) * _sign(0.5 - q)
    r["move_to_outcome"] = (lp - q) * _sign(y - q)
    r["model_side_of_half"] = int((p - q) * (0.5 - q) > 0)               # the model's disagreement points toward 0.5
    r["model_more_extreme"] = 1 - r["model_side_of_half"]               # ... or away from 0.5 (model more extreme than the market)
    return r


def disagreement_rows(moments, last, fc, model, window):
    rows, _ = admitted_rows(moments, last, fc, model, window)
    return [add_fields(r) for r in rows if r["disagree"]]


def pooled_disagreements(moments, last, fc, window, models=None):
    out = []
    for model in (models or ROSTER[window]):
        out += disagreement_rows(moments, last, fc, model, window)
    return out


# ----------------------------------------------------------------------------
# The toward-model test on arrays, and its shuffled baselines
# ----------------------------------------------------------------------------

def _arrays(rows):
    p = np.array([r["p"] for r in rows], dtype=float)
    q = np.array([r["q"] for r in rows], dtype=float)
    lp = np.array([r["last"] for r in rows], dtype=float)
    return p, q, lp


def real_rate(rows):
    """Toward-model rate over the disagreements among `rows` (rows are all rated)."""
    p, q, lp = _arrays(rows)
    mask = np.abs(p - q) >= DISAGREE_MIN
    if not mask.any():
        return float("nan")
    drift = np.abs(lp - p) <= np.abs(q - p) - DRIFT_MIN
    return float((mask & drift).sum() / mask.sum())


def _rates_for_perms(p, q, lp, perms):
    """Toward-model rate after the whole test is re-applied to each permuted p (rows of `perms`)."""
    P = p[perms]
    mask = np.abs(P - q) >= DISAGREE_MIN
    drift = np.abs(lp - P) <= np.abs(q - P) - DRIFT_MIN
    num = (mask & drift).sum(axis=1)
    den = mask.sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def shuffled_rate(rows, n_perm=N_PERM, seed=SEED):
    """Mean toward-model rate over `n_perm` permutations of p across `rows` (random.Random(seed).shuffle,
    each draw shuffling the previous order on), the disagreement filter re-applied after each shuffle."""
    p, q, lp = _arrays(rows)
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rates = []
    block = []
    for k in range(n_perm):
        rng.shuffle(idx)
        block.append(list(idx))
        if len(block) == 250 or k == n_perm - 1:
            rates.extend(_rates_for_perms(p, q, lp, np.array(block, dtype=np.intp)).tolist())
            block = []
    return float(np.nanmean(rates))


def _fast_shuffled_rate(p, q, lp, n_perm, nprng):
    """Same statistic as shuffled_rate, with numpy permutations (used inside the bootstrap loop only)."""
    n = len(p)
    rates = []
    done = 0
    while done < n_perm:
        k = min(250, n_perm - done)
        perms = nprng.permuted(np.tile(np.arange(n), (k, 1)), axis=1)
        rates.append(_rates_for_perms(p, q, lp, perms))
        done += k
    return float(np.nanmean(np.concatenate(rates)))


def boot_excess(rows, n_perm=N_PERM, n_boot=N_BOOT, seed=SEED):
    """Real toward-model rate minus the shuffled rate over `rows` (the permutation pool), with a 95% interval
    from `n_boot` event resamples; every resample recomputes both rates on the resampled rows.
    Returns (excess, lo, hi, real, shuffled)."""
    real = real_rate(rows)
    shuf = shuffled_rate(rows, n_perm, seed)
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[r["event_id"]].append(i)
    ids = list(g)
    p, q, lp = _arrays(rows)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        take = np.array([i for e in rng.choices(ids, k=len(ids)) for i in g[e]], dtype=np.intp)
        pb, qb, lb = p[take], q[take], lp[take]
        mask = np.abs(pb - qb) >= DISAGREE_MIN
        if not mask.any():
            continue
        rr = float((mask & (np.abs(lb - pb) <= np.abs(qb - pb) - DRIFT_MIN)).sum() / mask.sum())
        nprng = np.random.default_rng(rng.randrange(1 << 30))
        diffs.append(rr - _fast_shuffled_rate(pb, qb, lb, n_perm, nprng))
    diffs.sort()
    n = len(diffs)
    return real - shuf, diffs[int(0.025 * n)], diffs[int(0.975 * n)], real, shuf


def boot_mean(rows, key, n=N_BOOT, seed=SEED):
    """scoring.boot_mean with the 1,000-resample convention of these scripts."""
    return scoring.boot_mean(rows, key, n=n, seed=seed)


def boot_diff(rows_a, rows_b, key, n=N_BOOT, seed=SEED):
    """scoring.boot_diff (mean(a) - mean(b), the two sets resampled independently), 1,000 resamples."""
    return scoring.boot_diff(rows_a, rows_b, key, n=n, seed=seed)


def rate(rows, key):
    return st.fmean(r[key] for r in rows) if rows else float("nan")


def ci(t, d=3):
    v, lo, hi = t
    f = "%%+.%df" % d
    return (f + " [" + f + "," + f + "]") % (v, lo, hi)


def ci_u(t, d=3):
    v, lo, hi = t
    f = "%%.%df" % d
    return (f + " [" + f + "," + f + "]") % (v, lo, hi)
