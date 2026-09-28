import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, re
from _lib import load_moments_cuts, load_forecasts, build_scored
import scoring as S

moments, cuts = load_moments_cuts()
forecasts = load_forecasts("normal")
rows = build_scored(moments, cuts, forecasts, primary_only=True)
print("total primary scored rows:", len(rows))

HEDGE_RE = re.compile(r"uncertain|limited information|hard to say|difficult to (assess|determine|say|predict)|not (enough|much) information|unclear|insufficient (information|data)|no (specific|clear) information", re.I)

def is_hedged(reply):
    if not reply:
        return False
    # strip probability line
    body = re.sub(r"PROBABILITY\s*[:=].*", "", reply, flags=re.I)
    return bool(HEDGE_RE.search(body))

for r in rows:
    r["hedged"] = is_hedged(r.get("raw_reply"))
    r["dist"] = abs(r["p"] - r["q"])
    r["conf"] = abs(r["p"] - 0.5)
    r["crowd_side"] = 1.0 if r["direction"] == "with" else (0.0 if r["direction"] == "against" else None)

models = sorted(set(r["model"] for r in rows))
print("models:", models)

for win in ["A", "B", "C"]:
    wr = [r for r in rows if r["window"] == win]
    hedged = [r for r in wr if r["hedged"]]
    nothedged = [r for r in wr if not r["hedged"]]
    print(f"[{win}] n={len(wr)} hedged={len(hedged)} share={len(hedged)/len(wr):.3f}")
    # within-model diff, equal weight avg
    diffs_dist, diffs_conf, diffs_p, diffs_crowd, diffs_be = [], [], [], [], []
    for mdl in models:
        h = [r for r in wr if r["model"] == mdl and r["hedged"]]
        n_ = [r for r in wr if r["model"] == mdl and not r["hedged"]]
        if len(h) < 30 or len(n_) < 30:
            continue
        diffs_dist.append(sum(r["dist"] for r in h)/len(h) - sum(r["dist"] for r in n_)/len(n_))
        diffs_conf.append(sum(r["conf"] for r in h)/len(h) - sum(r["conf"] for r in n_)/len(n_))
        diffs_p.append(sum(r["p"] for r in h)/len(h) - sum(r["p"] for r in n_)/len(n_))
        hc = [r["crowd_side"] for r in h if r["crowd_side"] is not None]
        nc = [r["crowd_side"] for r in n_ if r["crowd_side"] is not None]
        if hc and nc:
            diffs_crowd.append(sum(hc)/len(hc) - sum(nc)/len(nc))
        diffs_be.append(sum(r["brier_edge"] for r in h)/len(h) - sum(r["brier_edge"] for r in n_)/len(n_))
    def avg(x): return sum(x)/len(x) if x else float("nan")
    print(f"   models_qualifying(30+ both sides)={len(diffs_dist)}  dist_diff={avg(diffs_dist):.4f} conf_diff={avg(diffs_conf):.4f} p_diff={avg(diffs_p):.4f} crowd_share_diff={avg(diffs_crowd):.4f} be_diff={avg(diffs_be):.4f}")

print()
print("=== confound: hedging share per model (is it driven by one/two models?) ===")
for mdl in models:
    mr = [r for r in rows if r["model"] == mdl]
    h = [r for r in mr if r["hedged"]]
    print(f"{mdl:45s} n={len(mr):6d} hedged={len(h):5d} share={len(h)/len(mr) if mr else 0:.3f}")

print()
print("=== confound: does hedging correlate with q band (mechanical: q<0.5 rows more numerous)? ===")
for lo,hi in [(0.05,0.25),(0.25,0.5),(0.5,0.75),(0.75,0.95)]:
    br = [r for r in rows if lo <= r["q"] <= hi]
    h = [r for r in br if r["hedged"]]
    print(f"band {lo}-{hi}: n={len(br)} hedge_share={len(h)/len(br) if br else 0:.3f}")

print()
print("=== confound: bare/short replies (forced strict-retry) overlap with hedge flag? ===")
for win in ["A","B","C"]:
    wr = [r for r in rows if r["window"]==win]
    bare = [r for r in wr if r.get("raw_reply") and len(r["raw_reply"].split()) < 20]
    bare_hedged = [r for r in bare if r["hedged"]]
    print(f"[{win}] bare_n={len(bare)} bare_hedged={len(bare_hedged)} bare_hedge_share={len(bare_hedged)/len(bare) if bare else 0:.3f}")
