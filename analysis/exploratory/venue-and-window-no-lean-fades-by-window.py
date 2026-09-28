import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys
import common_scripts as C
import scoring as sc

moments, cuts = C.load_benchmark()
rows = C.build_rows(moments, cuts, mode="normal")

for r in rows:
    r["pq"] = r["p"] - r["q"]
    r["qy"] = r["q"] - r["outcome_yes"]

nonsports = [r for r in rows if not r["sports"] and r["primary"]]

print("=== pooled over models, non-sports, mean(p-q) by window ===")
for w in ["A","B","C"]:
    sub = [r for r in nonsports if r["window"]==w]
    models = sorted(set(r["model"] for r in sub))
    mean, lo, hi = sc.boot_mean(sub, "pq", n=1000, seed=20260902)
    print(w, "n=",len(sub), "models=",len(models), "mean=%.3f"%mean, "CI=[%.3f,%.3f]"%(lo,hi))

print()
print("=== per model, non-sports, by window ===")
for w in ["A","B","C"]:
    sub = [r for r in nonsports if r["window"]==w]
    for model in sorted(set(r["model"] for r in sub)):
        msub = [r for r in sub if r["model"]==model]
        mean, lo, hi = sc.boot_mean(msub, "pq", n=1000, seed=20260902)
        excl = "excl0" if (lo>0 or hi<0) else "incl0"
        print(w, model, "n=",len(msub), "mean=%.3f"%mean, "CI=[%.3f,%.3f]"%(lo,hi), excl)

print()
print("=== 'mentions' tag, non-sports, by window ===")
for w in ["A","B","C"]:
    sub = [r for r in nonsports if r["window"]==w and any(str(t).strip().lower()=="mentions" for t in r["topic_tags"])]
    mean, lo, hi = sc.boot_mean(sub, "pq", n=1000, seed=20260902)
    print(w, "n=",len(sub), "mean=%.3f"%mean, "CI=[%.3f,%.3f]"%(lo,hi))

print()
print("=== q in 0.25-0.50, non-sports, by window ===")
for w in ["A","B","C"]:
    sub = [r for r in nonsports if r["window"]==w and 0.25<=r["q"]<=0.50]
    mean, lo, hi = sc.boot_mean(sub, "pq", n=1000, seed=20260902)
    print(w, "n=",len(sub), "mean=%.3f"%mean, "CI=[%.3f,%.3f]"%(lo,hi))

print()
print("=== market's own bias q-outcome, non-sports, by window ===")
for w in ["A","B","C"]:
    sub = [r for r in nonsports if r["window"]==w]
    mean, lo, hi = sc.boot_mean(sub, "qy", n=1000, seed=20260902)
    print(w, "n=",len(sub), "mean=%.3f"%mean, "CI=[%.3f,%.3f]"%(lo,hi))

print()
print("=== sports rows q in 0.40-0.60, by window ===")
sportsrows = [r for r in rows if r["sports"] and r["primary"]]
for w in ["A","B","C"]:
    sub = [r for r in sportsrows if r["window"]==w and 0.40<=r["q"]<=0.60]
    mean, lo, hi = sc.boot_mean(sub, "pq", n=1000, seed=20260902)
    print(w, "n=",len(sub), "mean=%.3f"%mean, "CI=[%.3f,%.3f]"%(lo,hi))

# tag mix by window on non-sports
print()
print("=== tag mix (top tags) non-sports by window ===")
from collections import Counter
for w in ["A","B","C"]:
    sub = [r for r in nonsports if r["window"]==w]
    cnt = Counter()
    for r in sub:
        for t in r["topic_tags"]:
            cnt[str(t).strip().lower()] += 1
    total = len(sub)
    print(w, "n=",total, cnt.most_common(6))
