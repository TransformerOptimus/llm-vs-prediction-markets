"""Shared loader for the calibration-shape scripts. Read-only on both databases."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import os, sys, sqlite3, random, statistics as st
from collections import defaultdict
import scoring

RUNS_DB = _paths.RUNS_DB
EXCLUDE = _roster.VALIDATION       # the probe-validation model, never scored
FIG_DIR = _paths.FIG_DIR
SEED = 20260902
N_BOOT = 1000
SHORT = _roster.SHORT              # display names from harness/models.json
MODELS = _roster.SCORED            # every scored model in the roster


def load(mode="normal", primary_only=True):
    """Scored rows (scoring.score_forecast) for every final forecast with a parsed probability."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    bykey = {m["moment_key"]: m for m in moments.values()}
    rc = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    rc.row_factory = sqlite3.Row
    rows = []
    for r in rc.execute("SELECT model, window, moment_key, parsed_probability p FROM forecasts "
                        "WHERE mode=? AND model!=? AND parsed_probability IS NOT NULL", (mode, EXCLUDE)):
        m = bykey.get(r["moment_key"])
        if m is None or (primary_only and not m["primary"]):
            continue
        if _paths.test_split_only() and m["split"] != "test":
            continue
        row = scoring.score_forecast(m, float(r["p"]), r["model"], cuts)
        row["topic_tags"] = m["topic_tags"]
        rows.append(row)
    return rows, cuts


def boot_mean(rows, key, n=N_BOOT, seed=SEED):
    return scoring.boot_mean(rows, key, n=n, seed=seed)


def boot_diff(a, b, key, n=N_BOOT, seed=SEED):
    return scoring.boot_diff(a, b, key, n=n, seed=seed)


def fmt(t):
    return "%+.3f [%+.3f, %+.3f]" % t


def boot_stat(rows, stat, n=N_BOOT, seed=SEED):
    """Cluster bootstrap (events) of an arbitrary statistic stat(rows) -> float. Returns (point, lo, hi)."""
    g = defaultdict(list)
    for r in rows:
        g[r["event_id"]].append(r)
    ids = list(g)
    point = stat(rows)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        draw = [r for c in rng.choices(ids, k=len(ids)) for r in g[c]]
        vals.append(stat(draw))
    vals.sort()
    return point, vals[int(0.025 * n)], vals[int(0.975 * n)]


def murphy(ps, ys, nb=10):
    """Murphy decomposition with nb equal-width bins: (reliability, resolution, uncertainty).
    Brier = reliability - resolution + uncertainty."""
    N = len(ps); ybar = st.fmean(ys)
    bins = defaultdict(list)
    for p, y in zip(ps, ys):
        bins[min(int(p * nb), nb - 1)].append((p, y))
    rel = sum(len(b) * (st.fmean(p for p, _ in b) - st.fmean(y for _, y in b)) ** 2 for b in bins.values()) / N
    res = sum(len(b) * (st.fmean(y for _, y in b) - ybar) ** 2 for b in bins.values()) / N
    return rel, res, ybar * (1 - ybar)


# Figure style: reference dataviz palette; the whole roster is too many hues, so two models are
# highlighted by name (worst and least-bad plateau) and every other model in the roster is drawn in
# muted grey; the market is near-black.
COLORS = {"openrouter/openai/gpt-oss-120b": "#eb6834", "openrouter/deepseek/deepseek-v4-pro": "#2a78d6"}
GREY = "#b9b8b3"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"
