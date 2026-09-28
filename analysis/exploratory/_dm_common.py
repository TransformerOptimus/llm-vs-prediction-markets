"""Shared loader for the direction-and-money scripts. Read-only on both databases."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import os, sys, sqlite3
import statistics as st
import scoring

RUNS_DB = _paths.RUNS_DB
EXCLUDE = set(_roster.NOT_SCORED)  # the probe-validation model and the fixture
SHORT = _roster.SHORT              # display names from harness/models.json
N_BOOT = 1000
SEED = scoring.SEED


def load_rows(mode="normal", primary_only=True):
    """Scored rows (scoring.score_forecast) for every final forecast of every scored model."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    by_key = {m["moment_key"]: m for m in moments.values()}
    rc = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    rc.row_factory = sqlite3.Row
    rows = []
    for r in rc.execute("SELECT model, window, moment_key, parsed_probability, status FROM forecasts WHERE mode=? "
                        "AND parsed_probability IS NOT NULL", (mode,)):
        if r["model"] in EXCLUDE:
            continue
        m = by_key.get(r["moment_key"])
        if m is None:
            continue
        if primary_only and not m["primary"]:
            continue
        s = scoring.score_forecast(m, float(r["parsed_probability"]), SHORT[r["model"]], cuts)
        s["gap"] = s["p"] - s["q"]
        s["absgap"] = abs(s["gap"])
        rows.append(s)
    return rows, moments, cuts


def boot(rows, key):
    return scoring.boot_mean(rows, key, n=N_BOOT, seed=SEED)


def bootdiff(a, b, key):
    return scoring.boot_diff(a, b, key, n=N_BOOT, seed=SEED)


def fmt(t):
    return "%.4f [%.4f, %.4f]" % t
