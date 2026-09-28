"""Compare two models on the rows they both forecast (comparisons between models are made on
their common rows only). Read-only on both databases.

Written for the GPT-5.2 against GPT-5.5 comparison, but it takes any two model ids. Nothing new is
measured here: the rows, the depth buckets, the trading rule with fees, the test split, the primary
analysis set and the 95% event bootstrap all come from scoring.py and the loader, exactly as
results.py uses them. The only thing this script adds is the restriction to common rows and the
paired difference (one row, two models, one subtraction), which is the honest way to compare two
models that did not forecast identical row sets.

Per window and arm it prints, on the common rows of the two models:
  - each model's Brier score, the market's Brier score on the same rows, the Brier edge
    (market minus model; positive means the model was more accurate) and the mean return per
    dollar with fees, each with its interval;
  - the paired difference (second model minus first) of Brier, of Brier edge and of return, with
    its interval, over the rows where both models traded;
  - each model's thin-minus-deep return gap (depth bucket 1 minus bucket 5) with its interval.

Usage: python3 analysis/pair_compare.py [--a openai/gpt-5.2] [--b openai/gpt-5.5]
                                        [--windows B C] [--arms news all] [--n-boot 2000]
                                        [--out analysis/out/results]
"""
import argparse
import os
import statistics as st
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_loader
import scoring


def common_rows(rows_a: List[dict], rows_b: List[dict]):
    """The rows both models forecast, paired by row id and returned in the same order."""
    by_a = {r["row_id"]: r for r in rows_a}
    by_b = {r["row_id"]: r for r in rows_b}
    ids = sorted(set(by_a) & set(by_b))
    return [by_a[i] for i in ids], [by_b[i] for i in ids]


def paired(rows_a: List[dict], rows_b: List[dict], key: str) -> List[dict]:
    """One row per market carrying the difference b minus a in `key`, for the event bootstrap.
    Rows where either side has no value (for example a market neither side could trade) are dropped."""
    out = []
    for ra, rb in zip(rows_a, rows_b):
        if ra.get(key) is None or rb.get(key) is None:
            continue
        out.append({"row_id": ra["row_id"], "event_id": ra["event_id"], "window": ra["window"],
                    "bucket": ra["bucket"], "d": rb[key] - ra[key]})
    return out


def side(ctx, rows: List[dict], label: str, view: str, arm: str, n_boot: int) -> dict:
    s = scoring.summarize(rows, n_boot=n_boot)
    b1 = [r for r in rows if r["bucket"] == 1]
    b5 = [r for r in rows if r["bucket"] == 5]
    gap, gap_lo, gap_hi = scoring.boot_diff(b1, b5, "ret", n=n_boot)
    return {"window": view, "arm": arm, "model": label, "n_common": len(rows), "n_events": s["n_events"],
            "n_traded": s["n_traded"], "brier_model": s["brier_model"], "brier_market": s["brier_market"],
            "brier_edge": s["brier_edge"], "brier_edge_lo": s["be_lo"], "brier_edge_hi": s["be_hi"],
            "market_more_accurate_clear": (s["be_hi"] < 0) if s["be_hi"] == s["be_hi"] else None,
            "ret": s["ret"], "ret_lo": s["ret_lo"], "ret_hi": s["ret_hi"], "ret_nofee": s["ret_nofee"],
            "gap_ret_b1_minus_b5": gap, "gap_lo": gap_lo, "gap_hi": gap_hi,
            "gap_excludes_zero": (gap_lo > 0 or gap_hi < 0) if gap_lo == gap_lo else None}


def f4(x):
    return "" if x is None else ("nan" if x != x else "%.4f" % x)


def pc(x):
    return "-" if x is None or x != x else "%+.1f%%" % (100 * x)


def main(argv=None):
    ap = argparse.ArgumentParser(description="compare two models on their common rows")
    ap.add_argument("--a", default="openai/gpt-5.2")
    ap.add_argument("--b", default="openai/gpt-5.5")
    ap.add_argument("--windows", nargs="+", default=["B", "C"])
    ap.add_argument("--arms", nargs="+", default=["news", "all"], choices=list(scoring.ARMS))
    ap.add_argument("--n-boot", type=int, default=scoring.N_BOOT)
    ap.add_argument("--out", default=os.path.join(scoring.OUT_DIR, "results"))
    args = ap.parse_args(argv)

    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    per_side, per_pair = [], []
    for arm in args.arms:
        rows_all, _ = harness_loader.load_pass("normal", moments, cuts, models=[args.a, args.b], arm=arm)
        for w in args.windows:
            sel = [r for r in rows_all if r["window"] == w and r["primary"] and r["split"] == "test"]
            ra, rb = common_rows([r for r in sel if r["model_name"] == args.a],
                                 [r for r in sel if r["model_name"] == args.b])
            if not ra:
                continue
            per_side.append(side(None, ra, args.a, w, arm, args.n_boot))
            per_side.append(side(None, rb, args.b, w, arm, args.n_boot))
            row = {"window": w, "arm": arm, "a": args.a, "b": args.b, "n_common": len(ra),
                   "n_events": len({r["event_id"] for r in ra})}
            for key, name in (("brier_model", "brier"), ("brier_edge", "edge"), ("ret", "ret")):
                d = paired(ra, rb, key)
                m, lo, hi = scoring.boot_mean(d, "d", n=args.n_boot)
                row["n_" + name] = len(d)
                row["d_" + name] = m
                row["d_%s_lo" % name] = lo
                row["d_%s_hi" % name] = hi
                row["d_%s_excludes_zero" % name] = (lo > 0 or hi < 0) if lo == lo else None
            per_pair.append(row)

    os.makedirs(args.out, exist_ok=True)
    stem = "pair_%s_vs_%s" % (args.a.split("/")[-1], args.b.split("/")[-1])
    scoring.write_csv(os.path.join(args.out, stem + "_sides.csv"), per_side)
    scoring.write_csv(os.path.join(args.out, stem + "_paired.csv"), per_pair)

    lines = ["# %s against %s on their common rows" % (args.a, args.b), "",
             "Test rows of the primary analysis set that both models forecast. Returns are per dollar staked "
             "with fees, over the rows that the trading rule traded. Intervals are 95%% bootstrap intervals "
             "resampling events, %d draws, seed %d." % (args.n_boot, scoring.SEED), "",
             "## Each model on the common rows", "",
             "| window | arm | model | common rows | events | traded | Brier model | Brier market | Brier edge | return with fees | thin minus deep return gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_side:
        lines.append("| %s | %s | %s | %d | %d | %d | %s | %s | %s [%s, %s] | %s [%s, %s] | %s [%s, %s] |" % (
            r["window"], r["arm"], r["model"].split("/")[-1], r["n_common"], r["n_events"], r["n_traded"],
            f4(r["brier_model"]), f4(r["brier_market"]), f4(r["brier_edge"]), f4(r["brier_edge_lo"]), f4(r["brier_edge_hi"]),
            pc(r["ret"]), pc(r["ret_lo"]), pc(r["ret_hi"]), pc(r["gap_ret_b1_minus_b5"]), pc(r["gap_lo"]), pc(r["gap_hi"])))
    lines += ["", "## Paired difference, row by row (%s minus %s)" % (args.b.split("/")[-1], args.a.split("/")[-1]), "",
              "A negative Brier difference means the second model was more accurate. A positive return "
              "difference means it made more money.", "",
              "| window | arm | rows | events | Brier difference | Brier edge difference | return difference | traded rows |",
              "|---|---|---|---|---|---|---|---|"]
    for r in per_pair:
        lines.append("| %s | %s | %d | %d | %s [%s, %s] | %s [%s, %s] | %s [%s, %s] | %d |" % (
            r["window"], r["arm"], r["n_common"], r["n_events"],
            f4(r["d_brier"]), f4(r["d_brier_lo"]), f4(r["d_brier_hi"]),
            f4(r["d_edge"]), f4(r["d_edge_lo"]), f4(r["d_edge_hi"]),
            pc(r["d_ret"]), pc(r["d_ret_lo"]), pc(r["d_ret_hi"]), r["n_ret"]))
    lines.append("")
    with open(os.path.join(args.out, stem + ".md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("wrote", os.path.join(args.out, stem + ".md"))


if __name__ == "__main__":
    main()
