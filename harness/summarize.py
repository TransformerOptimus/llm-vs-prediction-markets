"""Summarize one run: rows done, failures by class, tokens, cost, parse warnings,
finish reasons, and the host-pinning check.

  python3 harness/summarize.py <run_id or run dir>      (no argument: every run)
Also writes summary.json into the run folder. Exit status 1 if any run shows more
than one provider or served-model string (the host pin did not hold).
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
SKIPPED = ("leak_check_failed", "excluded_by_benchmark_flag", "ineligible_release_date")


def summarize(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json")) as f:
        manifest = json.load(f)
    recs = []
    with open(os.path.join(run_dir, "records.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    finals = {}
    for r in recs:                       # file order is time order: a resume's redo comes last
        if r["final"]:
            finals[r["row_id"]] = r      # the last final record for a row wins (resume redoes failed rows)
    finals = list(finals.values())
    recs.sort(key=lambda r: (r["row_id"], r.get("attempt", 1), r.get("transport_try", 1)))
    called = [r for r in recs if r["status"] not in SKIPPED]
    ok = [r for r in finals if r["parsed_probability"] is not None]
    failed = [r for r in finals if r["status"] == "failed"]
    skipped = {s: [r["row_id"] for r in finals if r["status"] == s] for s in SKIPPED}
    tin = sum(r["input_tokens"] or 0 for r in recs)
    tout = sum(r["output_tokens"] or 0 for r in recs)
    costs = [r["cost_usd"] for r in recs if r["cost_usd"] is not None]
    probs = [r["parsed_probability"] for r in ok]
    providers = sorted({str(r.get("provider")) for r in called if r.get("raw_reply") is not None})
    served = sorted({str(r.get("served_model")) for r in called if r.get("raw_reply") is not None})
    n_rows = len(ok) + len(failed) or 1
    s = {
        "run_id": manifest["run_id"], "mode": manifest["mode"], "model": manifest["model"],
        "template": manifest.get("template"), "window": manifest.get("window"), "news_only": manifest.get("news_only"),
        "rows_planned": manifest.get("rows_planned"), "rows_done": len(finals),
        "rows_not_started": manifest.get("rows_not_started"),
        "stopped_at_budget_cap": manifest.get("stopped_at_budget_cap"),
        "stopped_on_permanent_error": manifest.get("stopped_on_permanent_error"),
        "rows_ok": len(ok), "rows_failed": len(failed), "failed_row_ids": [r["row_id"] for r in failed],
        "failed_by_error_class": dict(Counter(r.get("error_class") for r in failed)),
        "rows_excluded_by_benchmark_flag": len(skipped["excluded_by_benchmark_flag"]),
        "excluded_row_ids": skipped["excluded_by_benchmark_flag"],
        "rows_blocked_by_leak_check": len(skipped["leak_check_failed"]), "blocked_row_ids": skipped["leak_check_failed"],
        "rows_ineligible_release_date": len(skipped["ineligible_release_date"]),
        "model_calls": len(called),
        "transport_retries": sum(1 for r in called if r.get("transport_try", 1) > 1),
        "strict_retries": sum(1 for r in called if r["attempt"] == 2 and r.get("transport_try", 1) == 1),
        "rows_ok_after_retry": sum(1 for r in finals if r["status"] == "ok_after_retry"),
        "records_by_error_class": dict(Counter(r.get("error_class") for r in called if r.get("error_class"))),
        "rows_with_parse_warning": sum(1 for r in finals if r.get("parse_warning")),
        "parse_warnings": dict(Counter(w for r in finals for w in (r.get("parse_warning") or []))),
        "finish_reasons": dict(Counter(str(r.get("finish_reason")) for r in called if r.get("raw_reply") is not None)),
        "providers_seen": providers, "served_models_seen": served,
        "host_pin_ok": len(providers) <= 1 and len(served) <= 1,
        "input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout,
        "avg_input_tokens_per_row": round(tin / n_rows, 1), "avg_output_tokens_per_row": round(tout / n_rows, 1),
        "cost_usd": round(sum(costs), 6) if costs else None, "cost_known_for_calls": len(costs),
        "litellm_cost_usd": manifest.get("litellm_cost_usd_accumulated"),
        "cost_gap_percent": manifest.get("cost_gap_percent"), "litellm_version": manifest.get("litellm_version"),
        "probability_mean": round(sum(probs) / len(probs), 4) if probs else None,
        "probability_min": min(probs) if probs else None, "probability_max": max(probs) if probs else None,
        "rows_with_future_date_mentions": sum(1 for r in finals if (r["leak_check"] or {}).get("future_date_mentions")),
        "rows_with_next_day_dateline": sum(1 for r in finals if (r["leak_check"] or {}).get("dateline_next_day")),
        "rows_scanner_hit_labelled_forward": sum(1 for r in finals if (r["leak_check"] or {}).get("scanner_hit_labelled_forward")),
        "settings": manifest.get("settings"), "git_commit": manifest.get("git_commit"),
        "tables_touched": manifest.get("tables_touched"), "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    return s


def main():
    targets = sys.argv[1:] or sorted(os.listdir(RUNS_DIR))
    bad = []
    for t in targets:
        run_dir = t if os.path.isdir(t) else os.path.join(RUNS_DIR, t)
        if not os.path.exists(os.path.join(run_dir, "records.jsonl")):
            continue
        s = summarize(run_dir)
        print("=" * 72)
        for k, v in s.items():
            print("%-34s %s" % (k, v))
        if not s["host_pin_ok"]:
            bad.append(s["run_id"])
            print("FAIL: more than one provider or model string served this run: %s / %s"
                  % (s["providers_seen"], s["served_models_seen"]))
    if bad:
        raise SystemExit("host pin failed for: %s" % ", ".join(bad))


if __name__ == "__main__":
    main()
