import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, sqlite3, statistics as st, random
from common_scripts import RUNS_DB

con = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
con.row_factory = sqlite3.Row

model = "openrouter/moonshotai/kimi-k2.5"

# first attempt = attempt 1, transport_try presumably 1 too; "no usable number" = parsed_probability is null
def first_attempt_fail(mode, window):
    rows = con.execute("""
        SELECT row_id, parsed_probability, finish_reason, attempt FROM calls
        WHERE model=? AND mode=? AND window=? AND attempt=1
        ORDER BY row_id, transport_try
    """, (model, mode, window)).fetchall()
    # dedupe by row_id keeping first row (transport_try=1 ideally)
    seen = {}
    for r in rows:
        if r["row_id"] not in seen:
            seen[r["row_id"]] = r
    fails = {rid: (r["parsed_probability"] is None) for rid, r in seen.items()}
    return fails

from common_ins import load_moments
moments = load_moments()

for window in ["A","B","C"]:
    fn = first_attempt_fail("normal", window)
    fp = first_attempt_fail("memory_probe", window)
    common_rows = set(fn) & set(fp)
    news_rows = {r for r in common_rows if moments.get(r) and moments[r]["news_available"]}
    primary_news_rows = {r for r in news_rows if moments[r]["primary"]}
    for label, rs in [("all rows", common_rows), ("news_available=1", news_rows), ("primary & news", primary_news_rows)]:
        n = len(rs)
        if n == 0: continue
        nfn = sum(fn[r] for r in rs); nfp = sum(fp[r] for r in rs)
        print(f"window {window} [{label}]: n={n} normal_fail_rate={nfn/n*100:.1f}% probe_fail_rate={nfp/n*100:.1f}%")

    # paired diff bootstrap over rows (clustered by event via moments)
