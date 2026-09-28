"""Paid runs: four accounting tables over every finished run. Read-only on runs.db and
benchmark.db; writes analysis/out/first_runs_report.md and, through gates.py, the per-run
quarantine files and the calibration gate.

1. Memory-probe quarantine per probe run (flag file, summary, news-value measure).
2. Probe-validation comparison: flag rates of the three-window models (roster label "bridge") on the 500 validation rows against
   the probe-validation model, with the two diagnostics (market far off; model confident right).
3. Calibration gate for the named models: verdict per (window, bucket) with the sports split.
4. Failure and cut-off accounting per finished run, for the paper's per-model caveat line.

Usage: python3 analysis/first_runs_report.py
"""
import os
import sqlite3
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates
import harness_loader
import scoring

PROBE_RUNS = [  # (model as stored in runs.db, window): every finished probe pass
    ("openai/gpt-5.2", "A"), ("openai/gpt-5.2", "B"), ("openai/gpt-5.2", "C"),
    ("openrouter/deepseek/deepseek-v3.2", "A"), ("openrouter/deepseek/deepseek-v3.2", "B"), ("openrouter/deepseek/deepseek-v3.2", "C"),
    ("openrouter/deepseek/deepseek-v4-pro", "B"), ("openrouter/deepseek/deepseek-v4-pro", "C"),
    ("openrouter/moonshotai/kimi-k2.5", "A"), ("openrouter/moonshotai/kimi-k2.5", "B"), ("openrouter/moonshotai/kimi-k2.5", "C"),
    ("openrouter/moonshotai/kimi-k2.6", "B"), ("openrouter/moonshotai/kimi-k2.6", "C"),
    ("openrouter/openai/gpt-oss-120b", "A"), ("openrouter/openai/gpt-oss-120b", "B"), ("openrouter/openai/gpt-oss-120b", "C"),
    ("openrouter/z-ai/glm-5.1", "B"), ("openrouter/z-ai/glm-5.1", "C"),
]
VALIDATION_MODEL = "openrouter/z-ai/glm-5.3-flash"
BRIDGE_ON_VALIDATION = ["openai/gpt-5.2", "openrouter/deepseek/deepseek-v3.2", "openrouter/moonshotai/kimi-k2.5",
                        "openrouter/openai/gpt-oss-120b"]   # three-window ("bridge") models with a Window A probe
GATE_MODELS = None   # every model in the forecasts view
REPLY_LIMIT = 4000  # max_tokens in every run's request params; a reply cut there has finish_reason = length


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


def f3(x):
    return "%.3f" % x if x is not None else "-"


def quarantine_tables(out):
    out += ["## 1. Memory-probe quarantine per probe run", "",
            "| model | window | probe run | assessed | flagged | rate | news-value rows | Brier no news | Brier with news | delta (no news minus with news) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for model, w in PROBE_RUNS:
        r = gates.quarantine_from_views(model, window=w)
        s = r["summary"]
        nv = s.get("news_value") or {}
        out.append("| %s | %s | %s | %d | %d | %s | %s | %s | %s | %s |" % (
            model, w, s["probe_run_id"], s["assessed"], s["quarantined"], pct(s["quarantined"], s["assessed"]),
            nv.get("rows", "-"), f3(nv.get("mean_brier_no_news")), f3(nv.get("mean_brier_with_news")), f3(nv.get("delta"))))
    out += ["", "Assessed = every row the probe pass covers (probe record where the probe ran, normal record standing in on rows without news). "
            "News value = mean Brier of the probe record minus mean Brier of the normal record, on the with-news rows only; positive means the news helped.", ""]
    return out


def validation_tables(out, moments, cuts):
    keys = set(l.strip() for l in open(os.path.join(scoring.OUT_DIR, "validation_moment_keys.txt")))
    out += ["## 2. Probe-validation comparison on the 500 fixed Window A rows", "",
            "| model | rows found | flagged | flag rate | market far off (abs(q-y) >= 0.30) | model confident right (abs(p-y) <= 0.10) | p under 0.10 | p over 0.90 | median abs(p-y) | median abs(q-y) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    per_bucket = {}
    for model in [VALIDATION_MODEL] + BRIDGE_ON_VALIDATION:
        rows, _ = harness_loader.load_pass("probe", moments, cuts, models=[model], window="A", arm="all", include_ladders=True)
        rows = [r for r in rows if r["moment_key"] in keys and r["record_source"] == "memory_probe"]
        n = len(rows)
        flagged = sum(gates.quarantine_rule(r["p"], r["q"], r["outcome_yes"])[0] for r in rows)
        far = sum(abs(r["q"] - r["outcome_yes"]) >= gates.MARKET_MARGIN - gates.EPS for r in rows)
        conf = sum(abs(r["p"] - r["outcome_yes"]) <= gates.PROBE_MARGIN + gates.EPS for r in rows)
        lo = sum(r["p"] < 0.10 for r in rows)
        hi = sum(r["p"] > 0.90 for r in rows)
        mp = st.median(abs(r["p"] - r["outcome_yes"]) for r in rows) if rows else None
        mq = st.median(abs(r["q"] - r["outcome_yes"]) for r in rows) if rows else None
        out.append("| %s | %d | %d | %s | %d (%s) | %d (%s) | %d | %d | %s | %s |" % (
            model, n, flagged, pct(flagged, n), far, pct(far, n), conf, pct(conf, n), lo, hi, f3(mp), f3(mq)))
        b = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            e = b[r["bucket"]]
            e[0] += 1
            e[1] += gates.quarantine_rule(r["p"], r["q"], r["outcome_yes"])[0]
            e[2] += abs(r["p"] - r["outcome_yes"]) <= gates.PROBE_MARGIN + gates.EPS
        per_bucket[model] = b
    out += ["", "Per depth bucket, rows / flagged / confident right:", "",
            "| bucket | " + " | ".join(m.split("/")[-1] for m in per_bucket) + " |", "|---|" + "---|" * len(per_bucket)]
    for bk in sorted({b for pb in per_bucket.values() for b in pb}, key=lambda x: (x is None, x)):
        out.append("| %s | " % bk + " | ".join("%d / %d / %d" % tuple(per_bucket[m][bk]) for m in per_bucket) + " |")
    out.append("")
    return out


def gate_tables(out):
    r = gates.gate_from_views(GATE_MODELS)
    out += ["## 3. Calibration gate (calibration split of the primary set)", ""]
    out += gates.to_markdown(r["gate"]).split("\n")[2:]
    return out


def failure_tables(out, rcon):
    runs = rcon.execute("select run_id, model, mode, window, rows_planned from runs where dry_run=0 and smoke=0 and void=0 "
                        "and run_id not like 'fixture%%' and state='finished' order by started_at").fetchall()
    out += ["## 4. Failure and cut-off accounting per finished run", "",
            "| run | planned | called | ok | failed | failed share | failure reasons | strict retry rows | strict share | final reply cut at %d tokens | any call cut at %d tokens |" % (REPLY_LIMIT, REPLY_LIMIT),
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run_id, model, mode, w, planned in runs:
        finals = rcon.execute("select row_id, status, error_class, strict, finish_reason from calls where run_id=? and final=1", (run_id,)).fetchall()
        called = [f for f in finals if not f[1].startswith(("excluded", "ineligible", "leak_check"))]
        ok = sum(f[1] in ("ok", "ok_after_retry") for f in called)
        failed = [f for f in called if f[1] == "failed"]
        reasons = Counter(f[2] or "?" for f in failed)
        strict_rows = rcon.execute("select count(distinct row_id) from calls where run_id=? and strict=1", (run_id,)).fetchone()[0]
        cut_final = sum(f[4] == "length" for f in called)
        cut_any = rcon.execute("select count(distinct row_id) from calls where run_id=? and finish_reason='length'", (run_id,)).fetchone()[0]
        out.append("| %s | %d | %d | %d | %d | %s | %s | %d | %s | %d | %d |" % (
            run_id, planned, len(called), ok, len(failed), pct(len(failed), len(called)),
            ", ".join("%s %d" % kv for kv in sorted(reasons.items())) or "-",
            strict_rows, pct(strict_rows, len(called)), cut_final, cut_any))
    out += ["", "Called = rows the model was actually asked (planned minus rows the harness skipped by benchmark flag or eligibility). "
            "Strict retry = the harness's one stricter re-ask after a reply with no readable number. "
            "Cut = the reply stopped at the token limit (finish_reason = length); the final-record column is the record that stands, "
            "the any-call column counts rows where any attempt was cut. Over 5% failed marks the model's results incomplete.", ""]
    return out


def main():
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    rcon = harness_loader.connect_runs(harness_loader.RUNS_DB)
    out = ["# Paid runs: quarantine, validation, gate and failure accounting", "",
           "Generated by analysis/first_runs_report.py, read-only on runs.db and benchmark.db.", ""]
    out = quarantine_tables(out)
    out = validation_tables(out, moments, cuts)
    out = gate_tables(out)
    out = failure_tables(out, rcon)
    path = os.path.join(scoring.OUT_DIR, "first_runs_report.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
