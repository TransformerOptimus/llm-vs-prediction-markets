import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, os, sqlite3
import scoring as S

REPO = _paths.REPO
RUNS_DB = os.path.join(REPO, "harness", "runs", "runs.db")
EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"

def load_moments_cuts():
    con = S.connect()
    moments = S.load_moments(con)
    cuts = S.load_or_compute_cuts(moments)
    con.close()
    return moments, cuts

def load_forecasts(mode="normal"):
    con = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    con.row_factory = sqlite3.Row
    sql = """SELECT model, mode, window, row_id, moment_key, parsed_probability, raw_reply,
                     input_tokens, output_tokens, finish_reason, status, parse_warning, timestamp
              FROM forecasts WHERE mode=? AND status IN ('ok','ok_after_retry') AND parsed_probability IS NOT NULL AND model != ?"""
    rows = [dict(r) for r in con.execute(sql, (mode, EXCLUDE_MODEL))]
    con.close()
    return rows

def build_scored(moments, cuts, forecasts, primary_only=True):
    out = []
    for f in forecasts:
        m = moments.get(f["row_id"])
        if m is None:
            continue
        if primary_only and not m["primary"]:
            continue
        p = f["parsed_probability"]
        row = S.score_forecast(m, p, f["model"], cuts)
        row["model"] = f["model"]
        row["raw_reply"] = f["raw_reply"]
        out.append(row)
    return out
