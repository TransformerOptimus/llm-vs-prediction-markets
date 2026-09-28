import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, statistics as st
from common_ins import load_moments, load_forecasts
import scoring as sc

moments = load_moments()
fc = load_forecasts("normal")
cuts = sc.load_or_compute_cuts(moments)

rows = {"A": [], "B": []}
for f in fc:
    m = moments.get(f["row_id"])
    if m is None or not m["primary"]:
        continue
    if m["q"] < 0.6:
        continue
    w = m["window"]
    if w not in rows:
        continue
    rows[w].append({"p": f["parsed_probability"], "q": m["q"], "sports": m["sports"],
                     "horizon_days": m["horizon_days"], "depth_usd": m["depth_usd"],
                     "bucket": sc.bucket_of(m, cuts),
                     "outcome": m["outcome_yes"], "event_id": m["event_id"]})

for w, rs in rows.items():
    sp = [r for r in rs if r["sports"]]
    ns = [r for r in rs if not r["sports"]]
    print(f"window {w}: sports n={len(sp)} mean q={st.fmean(r['q'] for r in sp):.3f} "
          f"mean horizon_days={st.fmean(r['horizon_days'] for r in sp if r['horizon_days']):.1f} "
          f"mean depth={st.fmean(r['depth_usd'] for r in sp):.0f}")
    print(f"window {w}: non-sports n={len(ns)} mean q={st.fmean(r['q'] for r in ns):.3f} "
          f"mean horizon_days={st.fmean(r['horizon_days'] for r in ns if r['horizon_days']):.1f} "
          f"mean depth={st.fmean(r['depth_usd'] for r in ns):.0f}")

print()
print("=== Confound test: control for horizon band -- within similar horizon, does sports/non-sports gap persist? ===")
def band(h):
    if h is None: return None
    if h <= 1: return "<=1d"
    if h <= 3: return "1-3d"
    if h <= 7: return "3-7d"
    if h <= 30: return "7-30d"
    return ">30d"

for w in ["A","B"]:
    print(f"-- window {w} --")
    hb = {}
    for r in rows[w]:
        b = band(r["horizon_days"])
        hb.setdefault(b, {"sports": [], "ns": []})
        (hb[b]["sports"] if r["sports"] else hb[b]["ns"]).append(r)
    for b, d in sorted(hb.items(), key=lambda x: str(x[0])):
        if len(d["sports"]) < 15 or len(d["ns"]) < 15:
            print(f"  horizon {b}: sports n={len(d['sports'])} non-sports n={len(d['ns'])} (too small, skip)")
            continue
        sp_shortfall = st.fmean(r["q"] - r["p"] for r in d["sports"])
        ns_shortfall = st.fmean(r["q"] - r["p"] for r in d["ns"])
        print(f"  horizon {b}: sports n={len(d['sports'])} shortfall={sp_shortfall:+.3f}  "
              f"non-sports n={len(d['ns'])} shortfall={ns_shortfall:+.3f}  diff={ns_shortfall-sp_shortfall:+.3f}")

print()
print("=== Confound test: control for depth quintile ===")
for w in ["A","B"]:
    print(f"-- window {w} --")
    db = {}
    for r in rows[w]:
        db.setdefault(r["bucket"], {"sports": [], "ns": []})
        (db[r["bucket"]]["sports"] if r["sports"] else db[r["bucket"]]["ns"]).append(r)
    for b in sorted(db, key=lambda x: (x is None, x)):
        d = db[b]
        if len(d["sports"]) < 15 or len(d["ns"]) < 15:
            print(f"  depth bucket {b}: sports n={len(d['sports'])} non-sports n={len(d['ns'])} (too small, skip)")
            continue
        sp_shortfall = st.fmean(r["q"] - r["p"] for r in d["sports"])
        ns_shortfall = st.fmean(r["q"] - r["p"] for r in d["ns"])
        print(f"  depth bucket {b}: sports n={len(d['sports'])} shortfall={sp_shortfall:+.3f}  "
              f"non-sports n={len(d['ns'])} shortfall={ns_shortfall:+.3f}  diff={ns_shortfall-sp_shortfall:+.3f}")
