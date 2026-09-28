import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys
import common_root as common
import scoring, _roster

rows, moments, cuts = common.load_all()
prim = [r for r in rows if r['primary']]

for w in "ABC":
    W = [r for r in prim if r["window"] == w]
    with_ = [r for r in W if r["direction"] == "with"]
    against = [r for r in W if r["direction"] == "against"]
    e_with = scoring.boot_mean(with_, "brier_edge", n=1000)
    e_against = scoring.boot_mean(against, "brier_edge", n=1000)
    diff = scoring.boot_diff(with_, against, "brier_edge", n=1000)
    print("Window", w, "with n=%d edge=%.4f [%.4f,%.4f]"%(len(with_), *e_with),
          "| against n=%d edge=%.4f [%.4f,%.4f]"%(len(against), *e_against),
          "| diff=%.4f [%.4f,%.4f]"%diff)
    # share of total brier deficit
    tot_def = sum(r["brier_edge"] for r in W if r["brier_edge"] < 0)
    against_def = sum(r["brier_edge"] for r in against if r["brier_edge"] < 0)
    print("   against share of rows: %.1f%%  against share of total deficit: %.1f%%" %
          (100*len(against)/len(W), 100*against_def/tot_def if tot_def else float('nan')))
    trw = [r for r in with_ if r["ret"] is not None]
    tra = [r for r in against if r["ret"] is not None]
    rw = scoring.boot_mean(trw, "ret", n=1000)
    ra = scoring.boot_mean(tra, "ret", n=1000)
    rdiff = scoring.boot_diff(trw, tra, "ret", n=1000)
    print("   with-ret n=%d %.4f [%.4f,%.4f] | against-ret n=%d %.4f [%.4f,%.4f] | diff %.4f [%.4f,%.4f]" %
          (len(trw), *rw, len(tra), *ra, *rdiff))

print()
print("=== recheck deficit share definition: net sum over ALL rows in the group (not filtered to edge<0) ===")
for w in "ABC":
    W = [r for r in prim if r["window"] == w]
    with_ = [r for r in W if r["direction"] == "with"]
    against = [r for r in W if r["direction"] == "against"]
    tot_net = -sum(r["brier_edge"] for r in W)
    against_net = -sum(r["brier_edge"] for r in against)
    print(w, "against share of rows %.1f%% | against share of NET total deficit %.1f%%" %
          (100*len(against)/len(W), 100*against_net/tot_net if tot_net else float('nan')))

print()
print("Per model, same measure (the pooled lines above stack every model's rows).")
for w in "ABC":
    for mdl in _roster.SCORED:
        M = [r for r in prim if r["window"] == w and r["model_name"] == mdl]
        if not M:
            continue
        against = [r for r in M if r["direction"] == "against"]
        with_ = [r for r in M if r["direction"] == "with"]
        if not against or not with_:
            continue
        ew = scoring.boot_mean(with_, "brier_edge", n=1000)
        ea = scoring.boot_mean(against, "brier_edge", n=1000)
        tot_def = sum(r["brier_edge"] for r in M if r["brier_edge"] < 0)
        against_def = sum(r["brier_edge"] for r in against if r["brier_edge"] < 0)
        tot_net = -sum(r["brier_edge"] for r in M)
        against_net = -sum(r["brier_edge"] for r in against)
        print("  window %s %-16s against share of rows %.1f%% | against share of the loss %.1f%% (net %.1f%%) | with edge %.4f [%.4f,%.4f] | against edge %.4f [%.4f,%.4f]" % (
            w, _roster.SHORT[mdl], 100 * len(against) / len(M),
            100 * against_def / tot_def if tot_def else float("nan"),
            100 * against_net / tot_net if tot_net else float("nan"), *ew, *ea))
