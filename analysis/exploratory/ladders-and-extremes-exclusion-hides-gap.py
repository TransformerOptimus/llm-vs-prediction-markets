import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, sqlite3, statistics as st
import scoring as sc
from common_scripts import RUNS_DB, EXCLUDE_MODEL

moments = sc.load_moments(sc.connect())

con = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
con.row_factory = sqlite3.Row
calls = con.execute("SELECT model, window, row_id, parsed_probability FROM forecasts WHERE mode='normal' AND model != ?", (EXCLUDE_MODEL,)).fetchall()
con.close()

# build scored rows: model brier and market brier, restricted to dateline flag out (news_dateline_after_capture==0), dateline/end-date flagged rows are out in ALL views (only ladder+price range vary)
def dateline_end_out(mom):
    return not mom["news_dateline_after_capture"] and not (mom.get("end_date_revised") or 0)

rows_by_window = {"A":[], "B":[], "C":[]}
for c in calls:
    if c["parsed_probability"] is None: continue
    mom = moments.get(c["row_id"])
    if mom is None: continue
    if not dateline_end_out(mom): continue
    if _paths.test_split_only() and mom["split"] != "test": continue
    w = mom["window"]
    y = mom["outcome_yes"]
    q = mom["q"]
    brier_model = (c["parsed_probability"]-y)**2
    brier_market = (q-y)**2
    edge = brier_market - brier_model
    in_primary = mom["in_price_range"] and not mom["is_ladder"]
    rows_by_window[w].append({"event_id": mom["event_id"], "edge": edge, "primary": in_primary,
                                "is_ladder": mom["is_ladder"], "q": q})

for w in ["A","B","C"]:
    rs = rows_by_window[w]
    prim = [r for r in rs if r["primary"]]
    allr = rs
    excl = [r for r in rs if not r["primary"]]
    def mean(xs): return st.fmean(x["edge"] for x in xs) if xs else float('nan')
    print(f"window {w}: n_all={len(allr)} n_primary={len(prim)} n_excl={len(excl)}")
    print(f"  primary edge = {mean(prim):.4f}")
    print(f"  all edge = {mean(allr):.4f}")
    print(f"  excluded edge = {mean(excl):.4f}")
    print(f"  excluded-primary = {mean(excl)-mean(prim):.4f}")
    print(f"  excluded rows %% = {len(excl)/len(allr)*100:.1f}")
    sum_gap_all = sum(-x["edge"] for x in allr)  # gap = market Brier minus model brier is edge; "summed gap" likely sum of -edge (model worse) or edge? Let's just report both
    sum_gap_excl = sum(-x["edge"] for x in excl)
    print(f"  excluded share of summed (negative edge magnitude): {sum_gap_excl/sum_gap_all*100:.1f}%" if sum_gap_all else "n/a")
