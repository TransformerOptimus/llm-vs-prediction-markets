import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, os, sqlite3
import scoring

RUNS_DB = _paths.RUNS_DB
BENCH_DB = _paths.BENCH_DB

EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"

def runs_con():
    c = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    c.row_factory = sqlite3.Row
    return c

def bench_con():
    return scoring.connect(BENCH_DB)

def load_moments():
    con = bench_con()
    m = scoring.load_moments(con)
    con.close()
    return m

def load_forecasts(mode=None):
    con = runs_con()
    sql = "SELECT * FROM forecasts WHERE model != ? AND status IN ('ok','ok_after_retry') AND parsed_probability IS NOT NULL"
    args = [EXCLUDE_MODEL]
    if mode:
        sql += " AND mode = ?"
        args.append(mode)
    rows = [dict(r) for r in con.execute(sql, args)]
    con.close()
    return rows
