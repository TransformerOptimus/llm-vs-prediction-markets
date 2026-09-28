import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys
import common_root as common, scoring, _roster

rows, moments, cuts = common.load_all()
prim = [r for r in rows if r['primary']]

for w in "ABC":
    W = [r for r in prim if r["window"] == w]
    S = [r for r in W if r["sports"]]
    N = [r for r in W if not r["sports"]]
    edge_s = scoring.boot_mean(S, "brier_edge", n=1000)
    edge_n = scoring.boot_mean(N, "brier_edge", n=1000)
    diff = scoring.boot_diff(S, N, "brier_edge", n=1000)
    print("Window", w, "sports n=%d edge=%.4f [%.4f,%.4f]" % (len(S), *edge_s),
          "| nonsports n=%d edge=%.4f [%.4f,%.4f]" % (len(N), *edge_n),
          "| diff(S-N)=%.4f [%.4f,%.4f]" % diff)

print()
print("Per model, same measure (the pooled lines above stack every model's rows).")
print("A ratio of 3 means the model falls three times further behind the market on non-sports than on sports.")
for w in "ABC":
    for mdl in _roster.SCORED:
        M = [r for r in prim if r["window"] == w and r["model_name"] == mdl]
        if not M:
            continue
        S = [r for r in M if r["sports"]]
        N = [r for r in M if not r["sports"]]
        if not S or not N:
            continue
        es = scoring.boot_mean(S, "brier_edge", n=1000)
        en = scoring.boot_mean(N, "brier_edge", n=1000)
        d = scoring.boot_diff(S, N, "brier_edge", n=1000)
        ratio = (en[0] / es[0]) if es[0] < 0 and en[0] < 0 else float("nan")
        print("  window %s %-16s sports n=%d edge=%.4f [%.4f,%.4f] | nonsports n=%d edge=%.4f [%.4f,%.4f] | diff=%.4f [%.4f,%.4f] | ratio %.1f" % (
            w, _roster.SHORT[mdl], len(S), *es, len(N), *en, *d, ratio))
