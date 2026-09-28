"""Shared reporting routine for the four reply-text scripts."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import statistics as st
from reply_text_common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYS = [("abs_dist", "distance from market |p-q|"), ("conf", "confidence |p-0.5|"), ("p", "probability given p"),
        ("with_crowd", "share on the crowd's side"), ("brier_edge", "Brier edge (market - model)"),
        ("brier_model", "model Brier")]


def prep(rows):
    for r in rows:
        r["conf"] = abs(r["p"] - 0.5)
        r["with_crowd"] = int(r["direction"] == "with")
    return rows


def report(rows, cls, label, windows="ABC", subsets=None):
    """Per window: n with/without class, within-model bootstrap difference for every key, per-model diffs."""
    results = {}
    subsets = subsets or {"all": lambda r: True}
    for sname, keep in subsets.items():
        for w in windows:
            sub = [r for r in rows if r["window"] == w and keep(r)]
            n1 = sum(r[cls] for r in sub); n0 = len(sub) - n1
            print("\n== %s | subset %s | window %s: n=%d, %s=%d (%.1f%%), other=%d" % (label, sname, w, len(sub), cls, n1, 100 * n1 / max(1, len(sub)), n0))
            for key, desc in KEYS:
                base = st.fmean(r[key] for r in sub if not r[cls]) if n0 else float("nan")
                pt, lo, hi, per, _, _ = boot_within_model(sub, cls, key)
                star = "*" if (lo > 0 or hi < 0) else ""
                print("   %-30s other-mean %+.3f  diff(with - without, within model) %+.4f [%+.4f, %+.4f]%s  per model: %s" % (
                    desc, base, pt, lo, hi, star, " ".join("%s=%+.3f(%d/%d)" % (m, v[0], v[1], v[2]) for m, v in sorted(per.items()))))
                results[(sname, w, key)] = (pt, lo, hi, n1, n0)
    return results


def figure(results, path, title, keys=("abs_dist", "conf", "brier_edge"), sname="all"):
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4))
    names = dict(KEYS)
    for ax, key in zip(axes, keys):
        ys, xs, err = [], [], []
        for i, w in enumerate("ABC"):
            if (sname, w, key) not in results:
                continue
            pt, lo, hi, n1, n0 = results[(sname, w, key)]
            xs.append(i); ys.append(pt); err.append([pt - lo, hi - pt])
        ax.errorbar(xs, ys, yerr=list(zip(*err)) if err else None, fmt="o", color="#2b6cb0", capsize=4)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Window A", "Window B", "Window C"])
        ax.set_title(names[key], fontsize=10)
        ax.set_ylabel("difference, with minus without", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("figure:", path)
