import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, os, sqlite3, json
import scoring as sc

REPO = _paths.REPO
RUNS_DB = REPO + "/harness/runs/runs.db"
EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"

def load_benchmark():
    con = sc.connect()
    moments = sc.load_moments(con)
    cuts = sc.load_or_compute_cuts(moments)
    return moments, cuts

def load_forecasts(mode="normal"):
    con = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    con.row_factory = sqlite3.Row
    sql = "SELECT model, mode, window, row_id, moment_key, parsed_probability, status FROM forecasts WHERE mode=? AND model != ?"
    out = []
    for r in con.execute(sql, (mode, EXCLUDE_MODEL)):
        if r["status"] not in ("ok", "ok_after_retry") or r["parsed_probability"] is None:
            continue
        out.append(dict(r))
    return out

def build_rows(moments, cuts, mode="normal"):
    fcs = load_forecasts(mode)
    rows = []
    for f in fcs:
        m = moments.get(f["row_id"])
        if m is None:
            continue
        if _paths.test_split_only() and m["split"] != "test":
            continue
        row = sc.score_forecast(m, f["parsed_probability"], f["model"], cuts)
        row["model"] = f["model"]
        row["topic_tags"] = m["topic_tags"]
        rows.append(row)
    return rows
