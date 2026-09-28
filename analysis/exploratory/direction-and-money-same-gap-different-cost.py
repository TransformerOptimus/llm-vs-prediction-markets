import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys
import common_scripts as C
import scoring as sc
import statistics as st

moments, cuts = C.load_benchmark()
rows = C.build_rows(moments, cuts, mode="normal")

for r in rows:
    r["gap"] = r["p"] - r["q"]
    r["gap2"] = r["gap"] ** 2
    r["cost"] = r["brier_model"] - r["brier_market"]   # positive = model worse than market

primary = [r for r in rows if r["primary"] and r["direction"] in ("with","against")]

def gapbin(g):
    ag = abs(g)
    if ag < 0.05: return None
    for lo,hi,name in [(0.05,0.1,"0.05-0.1"),(0.1,0.2,"0.10-0.20"),(0.2,0.3,"0.20-0.30"),(0.3,0.4,"0.30-0.40"),(0.4,0.5,"0.40-0.50")]:
        if lo <= ag < hi:
            return name
    if ag >= 0.5:
        return "0.5+"
    return None

for r in rows:
    r["gapbin"] = gapbin(r["gap"])

print("=== pooled ratio cost/gap^2 by direction, gap 0.05-0.5, per window ===")
for w in ["A","B","C"]:
    for d in ["against","with"]:
        sub = [r for r in primary if r["window"]==w and r["direction"]==d and 0.05<=abs(r["gap"])<0.5]
        if not sub: continue
        mc = st.fmean(r["cost"] for r in sub)
        mg = st.fmean(r["gap2"] for r in sub)
        print(w, d, "n=",len(sub), "mean_cost=%.4f"%mc, "mean_gap2=%.4f"%mg, "ratio=%.3f"%(mc/mg))

print()
print("=== within gap bin 0.10-0.20, with vs against, per window (Brier cost, bootstrap) ===")
for w in ["A","B","C"]:
    vals = {}
    for d in ["with","against"]:
        sub = [r for r in primary if r["window"]==w and r["direction"]==d and r["gapbin"]=="0.10-0.20"]
        mean, lo, hi = sc.boot_mean(sub, "cost", n=1000, seed=20260902)
        vals[d] = (mean,lo,hi,len(sub))
        print(w, d, "n=",len(sub), "mean_cost=%.4f"%mean, "CI=[%.4f,%.4f]"%(lo,hi), "mean_gap2=%.4f"%(st.fmean(r["gap2"] for r in sub) if sub else float('nan')))
    subw = [r for r in primary if r["window"]==w and r["direction"]=="with" and r["gapbin"]=="0.10-0.20"]
    suba = [r for r in primary if r["window"]==w and r["direction"]=="against" and r["gapbin"]=="0.10-0.20"]
    diff, lo, hi = sc.boot_diff(subw, suba, "cost", n=1000, seed=20260902)
    print(w, "with-minus-against diff=%.4f"%diff, "CI=[%.4f,%.4f]"%(lo,hi))

print()
print("=== gap bin 0.20-0.30, with-minus-against, per window ===")
for w in ["A","B","C"]:
    subw = [r for r in primary if r["window"]==w and r["direction"]=="with" and r["gapbin"]=="0.20-0.30"]
    suba = [r for r in primary if r["window"]==w and r["direction"]=="against" and r["gapbin"]=="0.20-0.30"]
    diff, lo, hi = sc.boot_diff(subw, suba, "cost", n=1000, seed=20260902)
    print(w, "n_with=",len(subw), "n_against=",len(suba), "diff=%.4f"%diff, "CI=[%.4f,%.4f]"%(lo,hi))
