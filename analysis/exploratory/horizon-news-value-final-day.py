import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sqlite3, random, statistics as st, json
from collections import defaultdict
from datetime import datetime

REPO = _paths.REPO
RUNS = REPO + "/harness/runs/runs.db"
BENCH = REPO + "/benchmark/benchmark.db"
EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"

def connect(path):
    c = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    c.row_factory = sqlite3.Row
    return c

WINDOWS = {("polymarket","polybench_snapshot"):"A", ("polymarket","pmxt_archive"):"B", ("kalshi","pmxt_archive"):"C"}
SPORTS_TAGS = {
    "sports","games","soccer","football","nfl","nba","wnba","mlb","baseball","nhl","hockey",
    "ncaa","ncaab","ncaaf","college football","college basketball","tennis","golf","pga","pga tour",
    "ufc","mma","boxing","esports","chess","cricket","rugby","formula 1","f1","grand prix",
    "motorsport","nascar","cycling","tour de france","professional cycling","olympics","mls",
    "ucl","uel","fifa world cup","2026 fifa world cup","world cup","wc game props","atp","wta",
    "athletics","swimming","volleyball","handball","darts","snooker","table tennis","badminton",
    "horse racing","lacrosse","cfl","xfl","afl","nrl","ipl","kbo","npb","liga mx","serie a",
    "bundesliga","la liga","ligue 1","premier league","epl","champions league","europa league",
}
HBANDS = [("<1d",0,1),("1-3d",1,3),("3-7d",3,7),(">7d",7,float("inf"))]

def hours_between(a,b):
    fa = datetime.fromisoformat(a.replace("Z","+00:00"))
    fb = datetime.fromisoformat(b.replace("Z","+00:00"))
    return (fb-fa).total_seconds()/3600.0

def hband(days):
    if days is None: return None
    for name,lo,hi in HBANDS:
        if lo<=days<hi: return name
    return None

def band_of(q):
    for lo,hi in [(0.05,0.25),(0.25,0.5),(0.5,0.75),(0.75,0.95)]:
        if lo<=q<=hi: return "%.2f-%.2f"%(lo,hi)
    return None

def load_moments():
    con = connect(BENCH)
    sql = """SELECT m.row_id, m.venue, m.book_source, m.price_yes, m.mid_yes,
             m.is_ladder, m.end_date_revised, m.news_dateline_after_capture, m.depth_usd, m.forecast_ts,
             o.outcome_yes, o.topic_tags, o.source_event_id, o.resolved_at
      FROM market_moments m JOIN moment_outcomes o USING (row_id)"""
    out = {}
    for r in con.execute(sql):
        d = dict(r)
        d["window"] = WINDOWS.get((d["venue"], d["book_source"]), "?")
        q = d["mid_yes"] if (d["window"]=="A" and d["mid_yes"] is not None) else d["price_yes"]
        d["q"] = q; d["band"] = band_of(q)
        d["primary"] = int(0.05 <= q <= 0.95 and not d["is_ladder"] and not d["end_date_revised"]
                            and not d["news_dateline_after_capture"])
        tags = json.loads(d["topic_tags"] or "[]")
        d["sports"] = int(any(str(t).strip().lower() in SPORTS_TAGS for t in tags))
        d["event_id"] = d["source_event_id"] if d["source_event_id"] is not None else "row:%d" % d["row_id"]
        try:
            days = hours_between(d["forecast_ts"], d["resolved_at"])/24.0
        except Exception:
            days = None
        d["horizon_days"] = days
        d["horizon_band"] = hband(days)
        out[d["row_id"]] = d
    return out

def load_forecasts():
    con = connect(RUNS)
    sql = """SELECT model, mode, row_id, parsed_probability FROM forecasts
             WHERE model != ? AND status IN ('ok','ok_after_retry') AND parsed_probability IS NOT NULL"""
    return [dict(r) for r in con.execute(sql, (EXCLUDE_MODEL,))]

def boot_mean(vals_by_cluster, n=1000, seed=20260902):
    ids = list(vals_by_cluster)
    if not ids: return (float('nan'),)*3
    allv = [v for c in ids for v in vals_by_cluster[c]]
    mean = st.fmean(allv)
    if len(ids) < 2: return (mean, float('nan'), float('nan'))
    rng = random.Random(seed)
    means=[]
    for _ in range(n):
        tot=cnt=0.0
        for c in rng.choices(ids, k=len(ids)):
            tot += sum(vals_by_cluster[c]); cnt += len(vals_by_cluster[c])
        means.append(tot/cnt)
    means.sort()
    return mean, means[int(0.025*n)], means[int(0.975*n)]

def boot_diff(a_by_cluster, b_by_cluster, n=1000, seed=20260902):
    ia, ib = list(a_by_cluster), list(b_by_cluster)
    if not ia or not ib: return (float('nan'),)*3
    point = st.fmean(v for c in ia for v in a_by_cluster[c]) - st.fmean(v for c in ib for v in b_by_cluster[c])
    rng = random.Random(seed)
    diffs=[]
    for _ in range(n):
        sa=[v for c in rng.choices(ia,k=len(ia)) for v in a_by_cluster[c]]
        sb=[v for c in rng.choices(ib,k=len(ib)) for v in b_by_cluster[c]]
        diffs.append(st.fmean(sa)-st.fmean(sb))
    diffs.sort()
    return point, diffs[int(0.025*n)], diffs[int(0.975*n)]

def group_by_cluster(rows, key):
    g = defaultdict(list)
    for r in rows: g[r["event_id"]].append(r[key])
    return g

moments = load_moments()
fc = load_forecasts()
idx = defaultdict(dict)
for r in fc:
    idx[(r["model"], r["row_id"])][r["mode"]] = r["parsed_probability"]

pairs = []
for (model, row_id), modes in idx.items():
    if "normal" in modes and "memory_probe" in modes:
        m = moments.get(row_id)
        if not m or not m["primary"] or m["window"]!="A":
            continue
        p_n, p_p = modes["normal"], modes["memory_probe"]
        y = m["outcome_yes"]
        pairs.append({
            "event_id": m["event_id"], "model": model, "row_id": row_id,
            "hband": m["horizon_band"], "band": m["band"], "sports": m["sports"], "depth": m["depth_usd"],
            "d_brier": (p_n-y)**2 - (p_p-y)**2,  # with news minus no news (negative = news helps)
        })

print("Window A paired rows total:", len(pairs))
for hb in ["<1d","1-3d","3-7d",">7d"]:
    sub=[p for p in pairs if p["hband"]==hb]
    markets = len({p["row_id"] for p in sub})
    g = group_by_cluster(sub,"d_brier")
    mean,lo,hi = boot_mean(g)
    print(hb, "n=",len(sub),"markets=",markets, "d_brier=%.4f [%.4f,%.4f]"%(mean,lo,hi))

sub1=[p for p in pairs if p["hband"]=="<1d"]
sub2=[p for p in pairs if p["hband"]=="3-7d"]
g1=group_by_cluster(sub1,"d_brier"); g2=group_by_cluster(sub2,"d_brier")
print("<1d minus 3-7d:", boot_diff(g1,g2))
