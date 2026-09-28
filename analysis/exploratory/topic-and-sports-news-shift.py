import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys
from common_ins import *
import random, statistics as st
from collections import defaultdict

# group_by_cluster / boot_mean / boot_diff: event-cluster bootstrap, 1,000 resamples, seed 20260902.
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

# index by (model,row_id,mode)
idx = defaultdict(dict)
for r in fc:
    idx[(r["model"], r["row_id"])][r["mode"]] = r["parsed_probability"]

pairs = []
for (model, row_id), modes in idx.items():
    if "normal" in modes and "memory_probe" in modes:
        m = moments.get(row_id)
        if not m or not m["primary"]:
            continue
        p_n, p_p = modes["normal"], modes["memory_probe"]
        y = m["outcome_yes"]
        pairs.append({
            "event_id": m["event_id"], "window": m["window"], "sports": m["sports"],
            "d_p": abs(p_p - p_n),
            "d_brier": (p_p-y)**2 - (p_n-y)**2,
        })

print("total paired rows:", len(pairs))
for w in ["A","B","C"]:
    for sp in [0,1]:
        sub = [p for p in pairs if p["window"]==w and p["sports"]==sp]
        markets = len({ (p["event_id"]) for p in sub})
        g = group_by_cluster(sub, "d_p")
        mean, lo, hi = boot_mean(g)
        gb = group_by_cluster(sub, "d_brier")
        bmean, blo, bhi = boot_mean(gb)
        print(w, "sports" if sp else "non-sports", "n=",len(sub), "events=",markets,
              "d_p=%.3f [%.3f,%.3f]"%(mean,lo,hi), "d_brier=%.4f [%.4f,%.4f]"%(bmean,blo,bhi))
    sportsrows=[p for p in pairs if p["window"]==w and p["sports"]==1]
    nonsp=[p for p in pairs if p["window"]==w and p["sports"]==0]
    ga = group_by_cluster(sportsrows,"d_p"); gb2=group_by_cluster(nonsp,"d_p")
    diff = boot_diff(gb2, ga)  # non-sports minus sports
    print(w, "d_p non-sports minus sports:", diff)
    ga2 = group_by_cluster(sportsrows,"d_brier"); gb3=group_by_cluster(nonsp,"d_brier")
    diffb = boot_diff(gb3, ga2)
    print(w, "d_brier non-sports minus sports:", diffb)
