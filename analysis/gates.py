"""The two Experiment B gates as pure functions (the memory-probe quarantine and the calibration
gate), with thin readers that fill their rows from runs.db or from one run id.

Buckets, sports, the primary-set
filter, q (mid_yes on Window A, price_yes on B and C) and the split label all come from
scoring.py through the loader; nothing is re-implemented here.

Memory-probe quarantine. Per (model, row) on the probe record (the normal record
stands in on rows without news, where the probe is not run):
    quarantined = 1   iff   |p - y| <= PROBE_MARGIN   and   |q - y| >= MARKET_MARGIN
A failed record (no parsed probability) is never flagged; the loader counts it as a failure
and it never reaches the rule. Output: analysis/out/quarantine_<probe run id>.jsonl in the
quarantine_flags column shape, plus a summary with the flag rate and the news-value measure.

Calibration gate. Rows: split = 'calibration' and primary, scored from the
normal pass. Regions: (window, depth bucket). Measure: reliability in at most N_BINS bins,
contiguous in stated probability, as near equal-count as the data allows, never splitting a
tied probability; ECE = count-weighted mean |mean p - outcome rate|. Verdict per (forecaster,
window, bucket): pass (n >= MIN_ROWS and ECE <= ECE_MAX), fail, or insufficient (n < MIN_ROWS,
which counts as not passed). Sports and non-sports are reported inside each region, not gated.
Output: analysis/out/calibration_gate.json and .md.
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness_loader
import scoring

OUT_DIR = scoring.OUT_DIR

PROBE_MARGIN = 0.10     # |p - y| at or under this = "confidently right with no news"
MARKET_MARGIN = 0.30    # |q - y| at or over this = "the crowd had not priced it in"
N_BINS = 3              # ceiling on reliability bins per region
MIN_ROWS = 30           # below this a region is "insufficient"
ECE_MAX = 0.10          # pass at or under this count-weighted mean bin gap
EPS = 1e-9


# ----------------------------------------------------------------------------
# The memory-probe quarantine rule, pure over scored rows
# ----------------------------------------------------------------------------

def quarantine_rule(p: float, q: float, y: float) -> Tuple[int, str]:
    """(quarantined 0/1, reason). p = probe probability, q = market probability, y = outcome."""
    confidently_right = abs(p - y) <= PROBE_MARGIN + EPS
    market_did_not_know = abs(q - y) >= MARKET_MARGIN - EPS
    if confidently_right and market_did_not_know:
        return 1, ("no-news p=%.3f within %.2f of outcome %d while the market q=%.3f was %.2f or more away"
                   % (p, PROBE_MARGIN, int(y), q, MARKET_MARGIN))
    if not confidently_right:
        return 0, "not confidently right without news (|p - outcome| = %.3f)" % abs(p - y)
    return 0, "outcome already priced in by the market (|q - outcome| = %.3f)" % abs(q - y)


def quarantine_flags(rows: List[dict], model: str, probe_run_id: str, flagged_at: Optional[str] = None) -> List[dict]:
    """Flags in the quarantine_flags column shape, one per scored probe row, sorted by moment_key.
    `rows` are scored rows (p, q, outcome_yes, moment_key, row_id, record_source); failed records
    never appear here because the loader does not score them."""
    now = flagged_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for r in sorted(rows, key=lambda x: x["moment_key"]):
        qd, reason = quarantine_rule(r["p"], r["q"], r["outcome_yes"])
        out.append({"moment_key": r["moment_key"], "row_id": r["row_id"], "model_name": model, "quarantined": qd,
                    "reason": reason, "flagged_at": now, "probe_run_id": probe_run_id,
                    "record_source": r.get("record_source")})
    return out


def news_value(probe_rows: List[dict], normal_rows: List[dict]) -> Optional[dict]:
    """News-value measure: on rows that have news, mean Brier without news minus mean Brier with news."""
    probe_by_key = {r["moment_key"]: r for r in probe_rows if r.get("record_source") == "memory_probe" and r["news_available"]}
    pairs = [(probe_by_key[r["moment_key"]]["brier_model"], r["brier_model"]) for r in normal_rows if r["moment_key"] in probe_by_key]
    if not pairs:
        return None
    return {"rows": len(pairs), "mean_brier_no_news": sum(p for p, _ in pairs) / len(pairs),
            "mean_brier_with_news": sum(q for _, q in pairs) / len(pairs), "delta": sum(p - q for p, q in pairs) / len(pairs)}


def write_quarantine(flags: List[dict], summary: dict, probe_run_id: str) -> Tuple[str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "quarantine_%s.jsonl" % probe_run_id)
    with open(out_path, "w") as f:
        for x in flags:
            f.write(json.dumps(x) + "\n")
    sum_path = os.path.join(OUT_DIR, "quarantine_%s_summary.json" % probe_run_id)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=1)
    return out_path, sum_path


def run_quarantine(probe_rows: List[dict], normal_rows: List[dict], model: str, probe_run_id: str,
                   normal_run_id: Optional[str], probe_counts: dict, flagged_at: Optional[str] = None) -> dict:
    """The rule end to end over already-scored rows; writes the two output files."""
    flags = quarantine_flags(probe_rows, model, probe_run_id, flagged_at)
    n, flagged = len(flags), sum(x["quarantined"] for x in flags)
    summary = {"probe_run_id": probe_run_id, "normal_run_id": normal_run_id, "model": model,
               "rule": {"probe_margin": PROBE_MARGIN, "market_margin": MARKET_MARGIN},
               "assessed": n, "quarantined": flagged, "flag_rate": flagged / n if n else None, "probe_counts": probe_counts}
    nv = news_value(probe_rows, normal_rows) if normal_rows else None
    if nv:
        summary["news_value"] = nv
    out_path, sum_path = write_quarantine(flags, summary, probe_run_id)
    return {"flags": flags, "summary": summary, "paths": (out_path, sum_path)}


# ----------------------------------------------------------------------------
# The calibration gate, pure over scored rows
# ----------------------------------------------------------------------------

def ece(pairs, k: int = N_BINS):
    """(ece, bins) for [(p, y), ...]: at most k bins, contiguous in stated p, as near equal-count
    as the data allows, a tied probability never split."""
    if not pairs:
        return None, []
    groups = {}
    for p, y in pairs:
        groups.setdefault(p, []).append(y)
    vals = sorted(groups)
    n = len(pairs)

    def close(chunk, cnt):
        ys = [y for v in chunk for y in groups[v]]
        mp = sum(v * len(groups[v]) for v in chunk) / cnt
        rate = sum(ys) / cnt
        return {"n": cnt, "p_lo": chunk[0], "p_hi": chunk[-1], "mean_p": mp, "outcome_rate": rate, "gap": abs(mp - rate)}

    bins, cur, cur_n, assigned, b = [], [], 0, 0, 1
    for i, v in enumerate(vals):
        cur.append(v)
        cur_n += len(groups[v])
        if b < k and assigned + cur_n >= round(n * b / k) and i + 1 < len(vals):
            bins.append(close(cur, cur_n))
            assigned += cur_n
            cur, cur_n, b = [], 0, b + 1
    if cur:
        bins.append(close(cur, cur_n))
    total = sum(x["n"] for x in bins)
    return sum(x["n"] * x["gap"] for x in bins) / total, bins


def verdict(n: int, e) -> str:
    if n < MIN_ROWS:
        return "insufficient"
    return "pass" if e <= ECE_MAX + EPS else "fail"


def region_report(rows: List[dict]) -> dict:
    pairs = [(r["p"], r["outcome_yes"]) for r in rows]
    e, bins = ece(pairs)
    out = {"n": len(rows), "ece": e, "bins": bins, "verdict": verdict(len(rows), e)}
    for name, flag in (("sports", 1), ("non_sports", 0)):          # reported, never gated
        sub = [(r["p"], r["outcome_yes"]) for r in rows if r["sports"] == flag]
        se, _ = ece(sub)
        out[name] = {"n": len(sub), "ece": se}
    return out


def build_gate(scored_rows: List[dict]) -> dict:
    """{forecaster: {window: {bucket: region entry}}} from calibration-slice primary rows."""
    cells = defaultdict(list)
    for r in scored_rows:
        if r["split"] == "calibration" and r["primary"]:
            cells[(r["forecaster"], r["window"], r["bucket"])].append(r)
    gate = {}
    for (fc, w, b), rows in sorted(cells.items()):
        gate.setdefault(fc, {}).setdefault(w, {})[str(b)] = region_report(rows)
    return gate


def passes(gate: dict, forecaster: str, window: str, bucket) -> bool:
    """True only when the region's verdict is pass; a missing region counts as not passed."""
    entry = gate.get(forecaster, {}).get(window, {}).get(str(bucket))
    return bool(entry) and entry["verdict"] == "pass"


def to_markdown(gate: dict) -> str:
    lines = ["# Calibration gate: verdict per forecaster, window and depth bucket", "",
             "Rule: pass = n >= %d and ECE <= %.2f on at most %d bins (near equal-count, a tied probability never split); "
             "insufficient (n < %d) stays closed." % (MIN_ROWS, ECE_MAX, N_BINS, MIN_ROWS), ""]
    for fc, wins in gate.items():
        lines += ["## %s" % fc, "", "| window | bucket | n | ECE | verdict | sports n / ECE | non-sports n / ECE |",
                  "|---|---|---|---|---|---|---|"]
        for w, buckets in sorted(wins.items()):
            for b, e in sorted(buckets.items(), key=lambda kv: int(kv[0])):
                fmt = lambda x: "%.3f" % x if x is not None else "-"
                lines.append("| %s | %s | %d | %s | %s | %d / %s | %d / %s |"
                             % (w, b, e["n"], fmt(e["ece"]), e["verdict"], e["sports"]["n"], fmt(e["sports"]["ece"]),
                                e["non_sports"]["n"], fmt(e["non_sports"]["ece"])))
        lines.append("")
    return "\n".join(lines)


def write_gate(gate: dict, source: dict) -> Tuple[str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"rule": {"n_bins": N_BINS, "min_rows": MIN_ROWS, "ece_max": ECE_MAX}, **source, "gate": gate}
    jp, mp = os.path.join(OUT_DIR, "calibration_gate.json"), os.path.join(OUT_DIR, "calibration_gate.md")
    with open(jp, "w") as f:
        json.dump(meta, f, indent=1)
    with open(mp, "w") as f:
        f.write(to_markdown(gate))
    return jp, mp


# ----------------------------------------------------------------------------
# Readers: the same rules fed from one run id (regression tests) or from the views (paper)
# ----------------------------------------------------------------------------

def _ctx():
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    return con, moments, cuts


def quarantine_from_run(probe_run_id: str, normal_run_id: Optional[str] = None, flagged_at: Optional[str] = None) -> dict:
    """By-run-id path (never for paper tables): the probe run's final records, with the paired
    normal run standing in on rows the probe did not cover."""
    con, moments, cuts = _ctx()
    probe = harness_loader.load_run(probe_run_id, moments, cuts, arm="all")
    man = probe["manifest"]
    if man["mode"] != "memory_probe":
        raise SystemExit("run %s is mode %s, not memory_probe" % (probe_run_id, man["mode"]))
    model = man["model"]
    rows = probe["rows"] + probe["ladder_rows"]
    for r in rows:
        r["record_source"] = "memory_probe"
    normal_rows = []
    if normal_run_id:
        normal = harness_loader.load_run(normal_run_id, moments, cuts, arm="all")
        nman = normal["manifest"]
        if nman["mode"] != "normal":
            raise SystemExit("run %s is mode %s, not normal" % (normal_run_id, nman["mode"]))
        if nman["model"] != model:
            raise SystemExit("model mismatch: probe %s vs normal %s" % (model, nman["model"]))
        normal_rows = normal["rows"] + normal["ladder_rows"]
        rows = harness_loader.stand_in(normal_rows, rows)
    return run_quarantine(rows, normal_rows, model, probe_run_id, normal_run_id, probe["counts"], flagged_at)


def quarantine_from_views(model: str, flagged_at: Optional[str] = None, window: Optional[str] = None) -> dict:
    """Scoring path: the probe_forecasts view (probe record, or the normal stand-in on rows the
    probe did not run) and the forecasts view for the news-value measure. `window` restricts
    both views to one window so each probe run gets its own flag file."""
    con, moments, cuts = _ctx()
    probe, pcounts = harness_loader.load_pass("probe", moments, cuts, models=[model], window=window, arm="all", include_ladders=True)
    normal, _ = harness_loader.load_pass("normal", moments, cuts, models=[model], window=window, arm="all", include_ladders=True)
    probe_ids = sorted({r["run_id"] for r in probe if r["record_source"] == "memory_probe"})
    normal_ids = sorted({r["run_id"] for r in normal})
    probe_run_id = "+".join(probe_ids) if probe_ids else "no_probe_run"
    counts = {rid: c for rid, c in pcounts.items()}
    return run_quarantine(probe, normal, model, probe_run_id, "+".join(normal_ids) or None, counts, flagged_at)


def gate_from_runs(run_ids: List[str]) -> dict:
    """By-run-id path: the gate from the normal runs named."""
    con, moments, cuts = _ctx()
    scored = []
    for rid in run_ids:
        r = harness_loader.load_run(rid, moments, cuts, arm="all")
        if r["manifest"]["mode"] != "normal":
            raise SystemExit("run %s is mode %s, not normal (the gate scores the normal pass)" % (rid, r["manifest"]["mode"]))
        scored.extend(r["rows"])
    gate = build_gate(scored)
    paths = write_gate(gate, {"run_ids": run_ids})
    return {"gate": gate, "paths": paths}


def gate_from_views(models: Optional[List[str]] = None) -> dict:
    """Scoring path: the forecasts view, normal pass, every non-void run."""
    con, moments, cuts = _ctx()
    rows, counts = harness_loader.load_pass("normal", moments, cuts, models=models, arm="all")
    gate = build_gate(rows)
    paths = write_gate(gate, {"source": "runs.db forecasts view", "models": models, "counts_by_run": counts})
    return {"gate": gate, "paths": paths}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="the memory-probe quarantine and the calibration gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("quarantine-run"); a.add_argument("probe_run_id"); a.add_argument("--normal-run")
    b = sub.add_parser("quarantine"); b.add_argument("model"); b.add_argument("--window")
    c = sub.add_parser("gate-runs"); c.add_argument("run_ids", nargs="+")
    d = sub.add_parser("gate"); d.add_argument("--models", nargs="*")
    args = ap.parse_args()
    if args.cmd == "quarantine-run":
        r = quarantine_from_run(args.probe_run_id, args.normal_run)
    elif args.cmd == "quarantine":
        r = quarantine_from_views(args.model, window=args.window)
    elif args.cmd == "gate-runs":
        r = gate_from_runs(args.run_ids)
    else:
        r = gate_from_views(args.models)
    if "summary" in r:
        s = r["summary"]
        print("assessed %d rows for %s: %d quarantined (rate %s)" % (s["assessed"], s["model"], s["quarantined"], s["flag_rate"]))
    else:
        g = r["gate"]
        print("gate built: %d forecasters, %d regions, %d pass" % (len(g), sum(len(b) for w in g.values() for b in w.values()),
              sum(1 for w in g.values() for b in w.values() for e in b.values() if e["verdict"] == "pass")))
    print("wrote", *r["paths"])


if __name__ == "__main__":
    main()
