"""Shared loader for the ladders-and-extremes scripts. Read-only on both databases."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import os, sqlite3, sys, random, statistics as st
from collections import defaultdict
REPO = _paths.REPO
import scoring

RUNS_DB = os.path.join(REPO, "harness", "runs", "runs.db")
EXCLUDE = set(_roster.NOT_SCORED)  # the probe-validation model and the fixture
SEED, N_BOOT = 20260902, 1000
SHORT = _roster.SHORT              # display names from harness/models.json
def price_class(q):
    if q < scoring.PRICE_LO: return "lo"
    if q > scoring.PRICE_HI: return "hi"
    return "mid"


def load(mode="normal"):
    """Scored rows: one per (model, market-moment) with a parsed probability. Also returns moments."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    bykey = {m["moment_key"]: m for m in moments.values()}
    r = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    view = "forecasts" if mode == "normal" else "probe_forecasts"
    sql = "SELECT model, mode, moment_key, parsed_probability, status FROM %s WHERE parsed_probability IS NOT NULL" % view
    rows = []
    for model, md, key, p, status in r.execute(sql):
        if model in EXCLUDE or key not in bykey:
            continue
        if mode == "normal" and md != "normal":
            continue
        m = bykey[key]
        row = scoring.score_forecast(m, float(p), SHORT.get(model, model), cuts)
        row["price_class"] = price_class(row["q"])
        row["mode"] = md
        rows.append(row)
    return rows, moments


def boot(rows, key, n=N_BOOT, seed=SEED):
    return scoring.boot_mean(rows, key, n=n, seed=seed)


def boot_diff(a, b, key, n=N_BOOT, seed=SEED):
    return scoring.boot_diff(a, b, key, n=n, seed=seed)


def fmt(t):
    return "%+.4f [%+.4f, %+.4f]" % t
