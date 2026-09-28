import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""convergence-baseline-choice: how much of the Window A drift toward the model comes from how the
shuffled baseline is built.

Per model and window (primary set, both splits, no gate, no quarantine, mode normal):
  real       the toward-model rate over the model's disagreements (|p - q| >= 0.10, rated last price)
  rule       p permuted across ALL the model's admitted rows (the rule in results.py), the whole test re-applied
  dis-only   p permuted across the model's DISAGREEMENT rows only, the whole test re-applied
Each baseline is the mean over 200 permutations (random.Random(20260902)).
Excess = real minus baseline, with a 95% interval from 1,000 event resamples that recompute both rates
on the resampled rows (seed 20260902). Last line: the single-seed permutation of results.py (rows sorted
by moment_key, one random.Random(20260902).shuffle) against the 200-permutation mean, gpt-5.2, Window A,
test split, rule baseline.
"""
import random
import convergence_common as cc

N_PERM = 200            # 200 permutations averaged
moments, last, fc = cc.load_all("normal")
print("rows: forecasts view, mode normal, status in %s, primary set (both splits, no gate); baselines = %d permutations; "
      "intervals = %d event resamples, seed %d" % (cc.STATUS_OK, N_PERM, cc.N_BOOT, cc.SEED))
print("window model            n_rows n_unrated n_disagree real | rule_baseline(all rows) excess [95%% CI] | disagree_only_baseline excess [95%% CI]")

results = {}
for w in cc.WINDOWS:
    for model in cc.ROSTER[w]:
        rows, unrated = cc.admitted_rows(moments, last, fc, model, w)
        dis = [r for r in rows if r["disagree"]]
        if not dis:
            continue
        rule = cc.boot_excess(rows, n_perm=N_PERM)
        only = cc.boot_excess(dis, n_perm=N_PERM)
        results[(w, model)] = (rule, only, len(dis))
        print("%s %-16s %5d %5d %5d %.3f | %.3f %s | %.3f %s"
              % (w, cc.SHORT[model], len(rows), unrated, len(dis), rule[3], rule[4], cc.ci(rule[:3]), only[4], cc.ci(only[:3])))

print()
print("Window A excess, rule minus disagreement-only baseline, per model:")
for model in cc.ROSTER["A"]:
    rule, only, n = results[("A", model)]
    print("   %-15s n=%d rule %s  dis-only %s  drop %+.3f" % (cc.SHORT[model], n, cc.ci(rule[:3]), cc.ci(only[:3]), rule[0] - only[0]))
for w in ("B", "C"):
    ex_rule = [results[(w, m)][0][0] for m in cc.ROSTER[w] if (w, m) in results]
    ex_only = [results[(w, m)][1][0] for m in cc.ROSTER[w] if (w, m) in results]
    clr_rule = [cc.SHORT[m] for m in cc.ROSTER[w] if (w, m) in results and (results[(w, m)][0][1] > 0 or results[(w, m)][0][2] < 0)]
    clr_only = [cc.SHORT[m] for m in cc.ROSTER[w] if (w, m) in results and (results[(w, m)][1][1] > 0 or results[(w, m)][1][2] < 0)]
    print("Window %s: rule excess %+.3f to %+.3f (clear of zero: %s); disagreement-only %+.3f to %+.3f (clear of zero: %s)"
          % (w, min(ex_rule), max(ex_rule), ", ".join(clr_rule) or "none", min(ex_only), max(ex_only), ", ".join(clr_only) or "none"))

# the single-seed permutation of results.py against the permutation mean (gpt-5.2, Window A, test split, rule baseline)
rows, _ = cc.admitted_rows(moments, last, fc, "openai/gpt-5.2", "A")
test = sorted([r for r in rows if r["m"]["split"] == "test"], key=lambda r: r["m"]["moment_key"])
print()
for pool, lab in ((test, "rule baseline (all admitted rows)"), ([r for r in test if r["disagree"]], "disagreement-only baseline")):
    ps = [r["p"] for r in pool]
    random.Random(cc.SEED).shuffle(ps)
    single = cc.real_rate([dict(r, p=p) for r, p in zip(pool, ps)])
    print("gpt-5.2, Window A, test split, %s (pool n=%d): single results.py-style shuffle = %.3f, mean of %d permutations = %.3f, "
          "real rate = %.3f" % (lab, len(pool), single, N_PERM, cc.shuffled_rate(pool, N_PERM), cc.real_rate(test)))

if _paths.want_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, w in zip(axes, cc.WINDOWS):
        names = [cc.SHORT[m] for m in cc.ROSTER[w] if (w, m) in results]
        for k, (idx, lab) in enumerate(((0, "rule baseline (all rows)"), (1, "disagreement-only baseline"))):
            v = [results[(w, m)][idx][0] for m in cc.ROSTER[w] if (w, m) in results]
            lo = [results[(w, m)][idx][0] - results[(w, m)][idx][1] for m in cc.ROSTER[w] if (w, m) in results]
            hi = [results[(w, m)][idx][2] - results[(w, m)][idx][0] for m in cc.ROSTER[w] if (w, m) in results]
            ax.errorbar([i + (k - 0.5) * 0.3 for i in range(len(names))], v, yerr=[lo, hi], fmt="o", capsize=3, label=lab)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title("Window %s" % w)
    axes[0].set_ylabel("toward-model rate minus baseline"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(_paths.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(_paths.FIG_DIR, "convergence-baseline-choice.png"), dpi=150)
