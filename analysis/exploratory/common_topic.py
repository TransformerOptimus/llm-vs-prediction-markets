"""Shared loader for the topic-and-sports scripts. Read-only; never writes to the repo."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import sys, sqlite3, statistics as st
import scoring

RUNS = _paths.RUNS_DB
EXCLUDE = set(_roster.NOT_SCORED)  # the probe-validation model and the fixture
SHORT = _roster.SHORT              # display names from harness/models.json
N_BOOT = 1000

def load(mode="normal"):
    con = scoring.connect()
    M = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(M)
    bykey = {m["moment_key"]: m for m in M.values()}
    rc = sqlite3.connect("file:%s?mode=ro" % RUNS, uri=True)
    rows = []
    for model, mk, p in rc.execute(
            "SELECT model, moment_key, parsed_probability FROM forecasts WHERE mode=? AND parsed_probability IS NOT NULL", (mode,)):
        if model in EXCLUDE or mk not in bykey:
            continue
        m = bykey[mk]
        if not m["primary"]:
            continue
        r = scoring.score_forecast(m, float(p), SHORT[model], cuts)
        r["tags"] = [str(t).strip().lower() for t in m["topic_tags"]]
        rows.append(r)
    return rows

def bm(rows, key):
    return scoring.boot_mean(rows, key, n=N_BOOT)

def bd(a, b, key):
    return scoring.boot_diff(a, b, key, n=N_BOOT)

def fmt(t):
    return "%+.4f [%+.4f, %+.4f]" % t
