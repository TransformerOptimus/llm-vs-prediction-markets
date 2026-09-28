"""Run one model over benchmark rows in one mode, logging every call.

Examples (from the repo root or from harness/):
  python3 harness/run.py --mode normal       --model fake:0.5 --window A --limit 10
  python3 harness/run.py --mode memory_probe --model fake:0.5 --window B --news-only --workers 4
  python3 harness/run.py --mode normal --model fake:0.5 --window C --template with_price --limit 3
  python3 harness/run.py --mode normal --model litellm:openrouter/<vendor/model> --window B --workers 4 \\
          --budget-usd 50 --i-approve-paid-calls

Real models go through LiteLLM (adapters/litellm_adapter.py): ids are LiteLLM ids
(openai/..., anthropic/..., gemini/..., openrouter/...), keys from .env only.

Modes: normal (news shown) and memory_probe (news replaced by "No news is available.").
Neither shows the market price, best bid or best ask (template default). The with_price
template is kept in the code but is not used in the study.
The normal pass covers every row in play once; the calibration split lives in the
database (moment_outcomes.split, analysis-only) and is applied at scoring time, so
there is no calibration run mode. The memory probe is run with --news-only, because on
rows without news the two prompts are identical; the analysis loader reuses the normal
record there.

One run id per (model, mode, window). Output: harness/runs/<run_id>/run.json (the
manifest) and records.jsonl (one line per model call, including transport retries,
strict retries, failures, and rows skipped before any call). Nothing is discarded.
When a run finishes (also after --resume) it is indexed into harness/runs/runs.db by
load_runs.py; the folders stay the raw record.
With --workers > 1 the records are written in completion order; summarize.py sorts
them by row_id.

A non-fake run refuses to start without --i-approve-paid-calls, on a dirty git tree,
when the model is missing from prices.json or models.json, or when an openrouter/ id has
no pinned host in models.json.

--dry-run plans the run (row selection, leak check, eligibility, manifest) and makes no
model call; the manifest is written with dry_run: true and void: true so it never feeds
the views. --smoke runs the 20 rows listed in harness/smoke_rows.txt (not included in this
repository; supply your own), records tokens per second, and voids itself after loading;
any 'length' finish or parse failure fails the smoke test.
--host-override <slug> swaps in the entry's recorded backup host (recorded in the manifest).
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import adapters  # noqa: E402
import leak_check  # noqa: E402
from adapters.base import MODELS_FILE, load_table  # noqa: E402
from adapters.litellm_adapter import litellm_version  # noqa: E402
from benchmark_reader import DEFAULT_DB, WINDOWS, BenchmarkReader, window_of  # noqa: E402
from forecast import DEFAULT_BACKOFF, forecast  # noqa: E402
from gate import EXIT_PAUSED_BY_OWNER, EXIT_PAUSED_ON_NETWORK, GatedAdapter, NetworkOutage, RunGate  # noqa: E402
from prompt import MODES, TEMPLATES, build_prompt_info, template_hash  # noqa: E402

RUNS_DIR = os.path.join(HERE, "runs")
SMOKE_ROWS_FILE = os.path.join(HERE, "smoke_rows.txt")
MAX_WORKERS = 16


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    """HEAD commit, with '-dirty' appended when the working tree has any change."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
        porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=HERE,
                                            stderr=subprocess.DEVNULL).decode().splitlines()
        # dirty = any change to a tracked file, or an untracked file inside harness/; an untracked
        # file elsewhere in the repo does not count
        dirty = [ln for ln in porcelain if not ln.startswith("??") or ln[3:].startswith("harness/")]
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def read_row_ids(path: str) -> List:
    """One row per line: an integer row_id or a moment_key (venue|market id|forecast_ts).
    Prefer moment keys in files that must stay valid if the rows are renumbered."""
    ids = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line if "|" in line else int(line))
    if not ids:
        raise SystemExit("row-ids file %s is empty" % path)
    return ids


# A final record's status -> the manifest counter it belongs to. Anything else
# (ok, ok_after_retry) is a row that produced a number.
FINAL_STATUS_COUNTER = {
    "failed": "failed",
    "leak_check_failed": "blocked",
    "excluded_by_benchmark_flag": "excluded",
    "ineligible_release_date": "ineligible",
}


def tally_final_records(records_path: str) -> Dict[str, int]:
    """Count the run's rows by outcome from its own call log: one count per row id,
    taken from that row's last final record.

    This is the whole run, every sitting. A resumed sitting only works the rows it
    still has to do, so counting what that sitting did would leave the earlier
    sittings' rows out of the manifest. Reading the log back is what keeps
    rows_ok / rows_failed / rows_blocked / rows_excluded equal to the record.
    """
    finals: Dict[int, str] = {}
    if os.path.exists(records_path):
        with open(records_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("final"):
                    finals[r["row_id"]] = r["status"]
    counts = {"ok": 0, "failed": 0, "blocked": 0, "excluded": 0, "ineligible": 0}
    for status in finals.values():
        counts[FINAL_STATUS_COUNTER.get(status, "ok")] += 1
    return counts


def eligibility_bound(row: Dict) -> str:
    """The latest model-visible date by which the outcome may have been known:
    the scheduled end, else the forecast moment. (The true outcome date is
    analysis-only and the harness may not read it.)"""
    return (row.get("market_end_date") or row["forecast_ts"])[:10]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=MODES,
                    help="normal = news shown; memory_probe = news replaced by the no-news line. "
                         "There is no calibration mode: the normal pass covers every row once and the "
                         "split lives in the database.")
    ap.add_argument("--model", required=True, help="adapter spec, e.g. fake:0.5 or litellm:openai/gpt-4.1-mini")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--window", choices=sorted(WINDOWS), default=None,
                    help="A (polymarket/polybench_snapshot), B (polymarket/pmxt_archive), C (kalshi/pmxt_archive); "
                         "selects on venue AND book_source together")
    ap.add_argument("--news-only", action="store_true",
                    help="only rows with news_available = 1 (the memory probe runs this way)")
    ap.add_argument("--limit", type=int, default=None, help="first N rows by row_id after the other filters")
    ap.add_argument("--row-ids-file", "--row-list", dest="row_ids_file", default=None,
                    help="file with one row_id or moment_key per line (--row-list is the same option)")
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="default",
                    help="default = price-blind (the study's only template); with_price = adds current price, "
                         "best bid and best ask (kept in the code, not used in the study)")
    ap.add_argument("--run-id", default=None, help="default: <utc time>_<mode>_<model>[_<window>]")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing run id: skip rows already finished, redo rows whose final status is failed")
    ap.add_argument("--workers", type=int, default=1, help="parallel model calls, 1 to %d" % MAX_WORKERS)
    ap.add_argument("--rate-limit-rpm", type=float, default=None,
                    help="calls per minute for this provider, shared by all workers (0 = no limit); "
                         "default 120 for real models, off for the fake adapter")
    ap.add_argument("--reasoning-tokens", type=int, default=None,
                    help="thinking cap N; default: models.json entry, else provider default")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="openai path only; sent as reasoning_effort")
    ap.add_argument("--answer-tokens", type=int, default=None, help="reply limit sent as max_tokens (default 4000)")
    ap.add_argument("--temperature", default=None,
                    help="a number, or 'none' to omit the parameter (default 0)")
    ap.add_argument("--transport-backoff", default=",".join(str(x) for x in DEFAULT_BACKOFF),
                    help="comma-separated waits in seconds between transport retries; "
                         "tries = waits + 1 (default 6 tries over about 10 minutes); '' = one try")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="stop starting new rows once the accumulated cost reaches this")
    ap.add_argument("--ignore-eligibility", action="store_true",
                    help="run rows the release-date rule would skip (for the probe-validation model only); recorded")
    ap.add_argument("--i-approve-paid-calls", action="store_true",
                    help="required to use any adapter other than the fake one")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="let a non-fake run start on a dirty git tree; the manifest records the override")
    ap.add_argument("--forward-rows-file", default=None,
                    help="JSON list of row ids labelled forward when the benchmark was built "
                         "(default benchmark/dateline_forward_rows.json); scanner hits on them are logged, not blocked")
    ap.add_argument("--abort-on-leak", action="store_true",
                    help="stop the whole run if any row fails the prompt leak check "
                         "(default: record the row as leak_check_failed, skip the model call, continue)")
    ap.add_argument("--host-override", default=None, metavar="SLUG",
                    help="use the entry's recorded backup host instead of the pinned one (openrouter/ ids only; "
                         "must equal backup_host.host in models.json; recorded in the manifest)")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: select rows, run the leak check and eligibility, write the manifest "
                         "(dry_run: true, void: true), make no model call, write no records")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke test: the 20 rows in harness/smoke_rows.txt (not included; supply your own) at the "
                         "maximum worker count unless --workers is given, "
                         "tokens per second and 429s recorded per host with a recommended worker count, the run "
                         "voided after loading; a 'length' finish or a parse failure fails the test")
    ap.add_argument("--no-reasoning-budget", action="store_true",
                    help="smoke runs only: send no thinking budget or effort level, so the host's default thinking "
                         "is measured; recorded in the manifest")
    ap.add_argument("--host-quantization", default=None, metavar="Q",
                    help="smoke runs only: also send provider.quantizations=[Q] so a host with several endpoints "
                         "serves the named precision (e.g. bf16); recorded in the manifest")
    ap.add_argument("--network-give-up-s", type=float, default=1800.0,
                    help="how long a network outage may last before the run exits with code %d for the plan "
                         "runner to relaunch it (default 30 minutes)" % EXIT_PAUSED_ON_NETWORK)
    args = ap.parse_args()

    if args.no_reasoning_budget and not args.smoke:
        raise SystemExit("--no-reasoning-budget is for smoke runs only (pass --smoke)")
    if args.host_quantization and not args.smoke:
        raise SystemExit("--host-quantization is for smoke runs only (pass --smoke)")
    if args.smoke:
        for flag, val in (("--window", args.window), ("--limit", args.limit), ("--row-ids-file", args.row_ids_file)):
            if val is not None:
                raise SystemExit("--smoke selects its own rows; drop %s" % flag)
        if args.news_only:
            raise SystemExit("--smoke selects its own rows; drop --news-only")
        args.row_ids_file = SMOKE_ROWS_FILE
        if "--workers" not in sys.argv:
            args.workers = MAX_WORKERS

    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit("--workers must be between 1 and %d" % MAX_WORKERS)
    backoff = [float(x) for x in args.transport_backoff.split(",") if x.strip() != ""]
    adapter_probe = adapters.get_adapter(args.model)
    if args.rate_limit_rpm is None:
        args.rate_limit_rpm = 120.0 if adapter_probe.is_paid else 0.0

    adapter = adapters.get_adapter(args.model)
    models = load_table(MODELS_FILE)
    model_entry = models.get(adapter.name)
    commit = git_commit()
    if adapter.is_paid:
        if not args.i_approve_paid_calls and not args.dry_run:
            raise SystemExit("refusing paid adapter %s without --i-approve-paid-calls" % adapter.name)
        if (commit is None or commit.endswith("-dirty")) and not args.allow_dirty and not args.dry_run:
            raise SystemExit("refusing a paid run on a dirty git tree (commit first, or pass --allow-dirty); "
                             "git_commit = %s" % commit)
        if adapter.price_per_million() is None:
            raise SystemExit("refusing: no price for %s in harness/prices.json" % adapter.name)
        if not model_entry or not model_entry.get("release_date"):
            raise SystemExit("refusing: %s missing from harness/models.json (release_date, host, reasoning_tokens)" % adapter.name)
        host = model_entry.get("host")
        if adapter.provider == "openrouter" and (not host or host == "direct"):
            raise SystemExit("refusing: openrouter id %s needs a pinned host in harness/models.json" % adapter.name)
        if adapter.provider not in ("openrouter", "direct_compatible") and host not in (None, "direct"):
            raise SystemExit("refusing: %s is a direct provider; models.json host must be 'direct'" % adapter.name)
    if args.host_override is not None:
        if not hasattr(adapter, "use_backup_host"):
            raise SystemExit("--host-override applies to real openrouter/ models only")
        try:
            adapter.use_backup_host(args.host_override, allow_any=args.smoke)
        except ValueError as e:
            raise SystemExit("refusing: %s" % e)
        print("host override: calls go to backup host %s (%s), recorded in the manifest"
              % (adapter.host, adapter.host_name))
    if args.reasoning_tokens is not None:
        adapter.reasoning_tokens = args.reasoning_tokens
    elif model_entry and model_entry.get("reasoning_tokens") is not None:
        adapter.reasoning_tokens = model_entry["reasoning_tokens"]
    if args.reasoning_effort is not None:
        adapter.reasoning_effort = args.reasoning_effort
    if args.answer_tokens is not None:
        adapter.answer_tokens = args.answer_tokens
    if args.temperature is not None:
        adapter.temperature = None if args.temperature.lower() == "none" else float(args.temperature)
    if args.host_quantization:
        adapter.host_quantization = args.host_quantization
    if args.no_reasoning_budget:            # drop the budget; keep an effort level only if the flag asked for one
        adapter.reasoning_tokens = None
        if args.reasoning_effort is None:
            adapter.reasoning_effort = None
    release_date = (model_entry or {}).get("release_date")

    reader = BenchmarkReader(args.db)
    row_ids = read_row_ids(args.row_ids_file) if args.row_ids_file else None
    rows = list(reader.iter_rows(row_ids=row_ids, limit=args.limit, window=args.window, news_only=args.news_only))
    if not rows:
        raise SystemExit("no rows selected")
    for row in rows:
        if not row.get("moment_key"):
            raise SystemExit("row %s has no moment_key; refusing to write a log without it" % row["row_id"])
    allowed_windows = (model_entry or {}).get("windows")
    if adapter.is_paid and allowed_windows and not args.smoke:
        outside = sorted({window_of(r) or "?" for r in rows} - set(allowed_windows))
        if outside:
            raise SystemExit("refusing: %s belongs to the %s set, which runs on windows %s only; the selected rows "
                             "include window %s" % (adapter.name, (model_entry or {}).get("set"),
                                                    "/".join(allowed_windows), "/".join(outside)))
    pre_report, excluded, blocked = leak_check.pre_run(reader, rows, args.mode, template=args.template,
                                                       forward_rows_file=args.forward_rows_file)
    if blocked and args.abort_on_leak:
        raise SystemExit("aborting: leak check blocked rows %s" % sorted(blocked))
    ineligible = {}
    if release_date and not args.ignore_eligibility:
        for row in rows:
            bound = eligibility_bound(row)
            if bound < release_date:
                ineligible[row["row_id"]] = ("outcome-date bound %s (market_end_date, else forecast_ts) is before "
                                             "the model's release date %s" % (bound, release_date))

    model_slug = "".join(c if c.isalnum() or c in "-." else "-" for c in adapter.name)
    run_id = args.run_id or "%s%s_%s_%s%s" % ("dryrun_" if args.dry_run else "smoke_" if args.smoke else "",
                                              datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), args.mode,
                                              model_slug, ("_" + args.window) if args.window else "")
    run_dir = os.path.join(RUNS_DIR, run_id)
    records_path = os.path.join(run_dir, "records.jsonl")
    manifest_path = os.path.join(run_dir, "run.json")
    if os.path.exists(run_dir) and not args.resume:
        raise SystemExit("run dir %s exists; pass --resume to continue it or choose another --run-id" % run_dir)
    os.makedirs(run_dir, exist_ok=True)

    done = set()
    spent_before = 0.0
    if args.resume and os.path.exists(records_path):
        finals: Dict[int, Dict] = {}
        with open(records_path) as f:
            for line in f:
                r = json.loads(line)
                spent_before += r.get("cost_usd") or 0.0
                if r.get("final"):
                    finals[r["row_id"]] = r
        done = {rid for rid, r in finals.items() if r["status"] != "failed"}

    previous = None
    if args.resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            previous = json.load(f)

    manifest = {
        "run_id": run_id, "mode": args.mode, "model": adapter.name, "model_spec": args.model,
        "provider": adapter.provider, "pinned_host": getattr(adapter, "host", None),
        "pinned_host_name": getattr(adapter, "host_name", None), "host_override": args.host_override,
        "host_override_kind": getattr(adapter, "host_override_kind", None),
        "host_kind": getattr(adapter, "host_kind", None), "api_base": getattr(adapter, "api_base", None),
        "no_reasoning_budget": args.no_reasoning_budget, "host_quantization": args.host_quantization,
        "expected_served_model": getattr(adapter, "expected_served_model", None),
        "model_set": (model_entry or {}).get("set"), "precision": (model_entry or {}).get("precision"),
        "dry_run": args.dry_run, "smoke": args.smoke,
        "litellm_version": litellm_version(),
        "window": args.window, "windows_present": sorted({window_of(r) or "?" for r in rows}),
        "news_only": args.news_only, "template": args.template,
        "prompt_template_hash": template_hash(args.template),
        "template_id": "harness-prompt-" + template_hash(args.template),
        "db_path": os.path.abspath(args.db), "db_sha256": sha256_file(args.db),
        "row_ids_file": args.row_ids_file, "limit": args.limit, "rows_planned": len(rows),
        "row_ids": [r["row_id"] for r in rows], "moment_keys": [r["moment_key"] for r in rows],
        "settings": adapter.settings(), "workers": args.workers, "rate_limit_rpm": args.rate_limit_rpm,
        "transport_backoff_s": backoff, "budget_usd": args.budget_usd,
        "release_date": release_date, "ignore_eligibility": args.ignore_eligibility,
        "git_commit": commit,
        "git_tree": ("dirty, override used" if args.allow_dirty else "dirty") if (commit or "").endswith("-dirty") else "clean",
        "forward_rows_file": leak_check.forward_rows_source(),
        "started_at": now_iso(), "resumed": args.resume,
        "pre_run_leak_check": pre_report,
        "prices_known": adapter.price_per_million() is not None,
        "rows_excluded_by_benchmark_flag": excluded,
        "rows_blocked_by_leak_check": blocked,
        "rows_ineligible_release_date": ineligible,
    }
    if previous is not None:
        # the run's code history: the commit it started on stays as git_commit; every resume adds
        # its own commit beside it
        manifest["git_commit"] = previous.get("git_commit", commit)
        manifest["resume_commit"] = commit
        manifest["first_started_at"] = previous.get("first_started_at", previous.get("started_at"))
        manifest["resumes"] = (previous.get("resumes") or []) + [
            {"at": manifest["started_at"], "commit": commit, "rows_done_before": len(done),
             "previous_state": previous.get("state"), "previous_note": previous.get("paused_note")}]
        for k in ("pause_history",):
            manifest[k] = previous.get(k) or []
        if previous.get("state") == "paused_by_owner":
            manifest["pause_history"].append({"paused_at": previous.get("paused_at"), "note": previous.get("paused_note")})
    if args.dry_run:
        manifest.update(void=True, void_at=now_iso(), void_reason="dry run: no model calls were made",
                        finished_at=now_iso(), rows_ok=0, rows_failed=0, rows_blocked=0, rows_excluded=0,
                        rows_ineligible=0, rows_completed=0, rows_not_started=len(rows),
                        rows_would_be_called=len(rows) - len(set(excluded) | set(blocked) | set(ineligible)),
                        tables_touched=sorted(reader.tables_touched))
        leak_check.check_reader_after(reader)
        reader.close()
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("run_id:", run_id)
    for line in pre_report:
        print("leak check:", line)
    if ineligible:
        print("eligibility: %d rows skipped (release date %s)" % (len(ineligible), release_date))
    if args.dry_run:
        print("DRY RUN: %d rows planned, %d would be called (%d excluded by benchmark flag, %d blocked by leak "
              "check, %d ineligible); no model call made; manifest at %s"
              % (len(rows), manifest["rows_would_be_called"], len(excluded), len(blocked), len(ineligible),
                 manifest_path))
        index_run(run_dir, run_id, 0)
        return

    def base_record(row, attempt, strict, final, status, prompt, prompt_info=None):
        prompt_info = prompt_info or build_prompt_info(row, args.mode, template=args.template)[1]
        return {"run_id": run_id, "mode": args.mode, "model": adapter.name, "template": args.template,
                "row_id": row["row_id"], "moment_key": row["moment_key"], "window": window_of(row),
                "venue": row["venue"], "venue_market_id": row["venue_market_id"],
                "forecast_ts": row["forecast_ts"], "book_source": row.get("book_source"),
                "attempt": attempt, "transport_try": 1, "strict": strict,
                "final": final, "status": status, "error_class": None, "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_reply": None, "parsed_probability": None, "parse_warning": [], "timestamp": now_iso(),
                "input_tokens": None, "output_tokens": None, "cost_usd": None, "litellm_cost_usd": None,
                "request_params": None, "finish_reason": None,
                "provider": None, "served_model": None, "latency_s": None, "tokens_per_s": None,
                "rate_limit": None, "rate_limit_headers": None,
                "error": None, "error_message": None, "adapter_raw": None, "leak_check": None,
                "description_paragraphs_stripped": prompt_info["description_paragraphs_stripped"],
                "description_stripped_text": prompt_info["description_stripped_text"],
                "price_shown_from": prompt_info["price_shown_from"]}

    counts = {"ok": 0, "failed": 0, "blocked": 0, "excluded": 0, "ineligible": 0}
    state = {"spent": spent_before, "spent_litellm": 0.0, "stopped_at_budget_cap": False, "stop": False,
             "stopped_on_permanent_error": None, "paused_on_network": False, "requeued_on_outage": 0,
             "paused_by_owner": None}
    lock = threading.Lock()          # guards the log file, counts, spend, and the queue
    queue: List[int] = list(range(len(rows)))
    manifest_lock = threading.Lock()

    def write_manifest(**fields):
        """Rewrite run.json mid-run (state changes the plan runner reads while running)."""
        with manifest_lock:
            manifest.update(fields)
            tmp = manifest_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp, manifest_path)

    def on_saturation(record):
        if record is not None:
            print("HOST SATURATED: 429s outnumbered successful calls in each of the last %d minutes (%s); "
                  "the run keeps going; consider the backup host" % (record["minutes"], record["counts_per_minute"]))
            write_manifest(state="host_saturated", saturation_events=gate.saturation_events)
        else:
            print("host saturation cleared")
            write_manifest(state="running", saturation_events=gate.saturation_events)

    gate = RunGate(args.rate_limit_rpm, backoff, probe=getattr(adapter, "probe", None),
                   give_up_s=args.network_give_up_s, on_saturation=on_saturation)
    stop_file = os.path.join(run_dir, "STOP")

    def owner_stop(how):
        with lock:
            if state["paused_by_owner"] is None:
                state["paused_by_owner"] = how
                state["stop"] = True
        print("STOP requested (%s): no new rows start; rows in flight finish and are logged" % how)

    def on_signal(signum, frame):
        owner_stop("signal %s" % {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT"}.get(signum, signum))

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
    limited = GatedAdapter(adapter, gate)
    out = open(records_path, "a")

    def write(recs):
        with lock:
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    def skip(row, status, reason, key):
        prompt, info = build_prompt_info(row, args.mode, template=args.template)
        rec = base_record(row, 1, False, True, status, prompt, info)
        rec["error"] = reason
        write([rec])
        with lock:
            counts[key] += 1

    def records_for(row, attempts, final_status):
        recs = []
        for a in attempts:
            final = final_status is not None and a is attempts[-1]
            rec = base_record(row, a["attempt"], a["strict"], final,
                              final_status if final else "retry_needed", a["prompt"], a["prompt_info"])
            rec.update({k: a[k] for k in ("transport_try", "raw_reply", "parsed_probability", "parse_warning",
                                          "error_class", "input_tokens", "output_tokens", "cost_usd",
                                          "litellm_cost_usd", "request_params",
                                          "finish_reason", "provider", "served_model", "latency_s", "error",
                                          "error_message", "adapter_raw", "leak_check", "rate_limit",
                                          "rate_limit_headers")})
            if rec["raw_reply"] is not None and rec["output_tokens"] and (rec["latency_s"] or 0) > 0:
                rec["tokens_per_s"] = round(rec["output_tokens"] / rec["latency_s"], 2)   # output tokens / wall time
            recs.append(rec)
        return recs

    def work_one(i, row):
        rid = row["row_id"]
        if rid in done:
            return
        leak_check.check_row(row)
        if rid in excluded:
            skip(row, "excluded_by_benchmark_flag", excluded[rid], "excluded")
            print("[%d/%d] row %s -> EXCLUDED by benchmark flag, no model call" % (i, len(rows), rid))
            return
        if rid in blocked:
            skip(row, "leak_check_failed", blocked[rid], "blocked")
            print("[%d/%d] row %s -> BLOCKED by leak check, no model call" % (i, len(rows), rid))
            return
        if rid in ineligible:
            skip(row, "ineligible_release_date", ineligible[rid], "ineligible")
            print("[%d/%d] row %s -> INELIGIBLE (release date), no model call" % (i, len(rows), rid))
            return
        result = forecast(row, args.mode, limited, template=args.template,
                          prompt_check=leak_check.check_prompt, backoff=backoff)
        recs = records_for(row, result["attempts"], result["status"])
        write(recs)
        with lock:
            counts["ok" if result["probability"] is not None else "failed"] += 1
            state["spent"] += sum(a["cost_usd"] or 0.0 for a in result["attempts"])
            state["spent_litellm"] += sum(a["litellm_cost_usd"] or 0.0 for a in result["attempts"])
            if result["error_class"] == "permanent" and state["stopped_on_permanent_error"] is None:
                state["stopped_on_permanent_error"] = result["permanent_error"]
                state["stop"] = True        # no new rows start; rows in flight finish and are logged
                print("PERMANENT ERROR on row %s, stopping the run: %s" % (rid, result["permanent_error"][:200]))
            if args.budget_usd is not None and state["spent"] >= args.budget_usd and not state["stopped_at_budget_cap"]:
                state["stopped_at_budget_cap"] = True
                print("budget cap reached: $%.4f >= $%.4f; no new rows will start" % (state["spent"], args.budget_usd))
        print("[%d/%d] row %s -> %s (%s%s)" % (i, len(rows), rid, result["probability"], result["status"],
                                               (", " + result["error_class"]) if result["error_class"] else ""))

    errors: List[str] = []

    def worker():
        while True:
            if gate.outage is not None:          # another worker declared an outage: wait, do not take a row
                if not gate.wait_for_network():
                    with lock:
                        state["stop"] = True
                        state["paused_on_network"] = True
                    return
            if os.path.exists(stop_file):
                owner_stop("STOP file")
            with lock:
                if state["stopped_at_budget_cap"] or state["stop"] or not queue:
                    return
                i = queue.pop(0)
            try:
                work_one(i + 1, rows[i])
            except NetworkOutage as outage:
                # not a row failure: log the attempts made so far as retry_needed, put the row back
                # at the front of the queue, and wait for the network with the other workers
                write(records_for(rows[i], outage.attempts, None))
                with lock:
                    queue.insert(0, i)
                    state["requeued_on_outage"] += 1
                gate.note_row_in_flight(rows[i]["row_id"])
                print("[%d/%d] row %s -> network outage, row requeued; waiting for the network"
                      % (i + 1, len(rows), rows[i]["row_id"]))
                if not gate.wait_for_network():
                    with lock:
                        state["stop"] = True
                        state["paused_on_network"] = True
                    return
            except Exception as e:  # a leak error or a bug: stop every worker, keep the log
                with lock:
                    errors.append("row %s: %s" % (rows[i]["row_id"], e))
                    state["stop"] = True
                return

    if args.budget_usd is not None and state["spent"] >= args.budget_usd:
        state["stopped_at_budget_cap"] = True
        print("budget cap already reached by earlier records ($%.4f); nothing started" % state["spent"])
    threads = [threading.Thread(target=worker, name="worker-%d" % k) for k in range(args.workers)]
    for t in threads:
        t.start()
    write_manifest(state="running")
    for t in threads:
        while t.is_alive():
            t.join(1.0)           # short joins so a SIGTERM/SIGINT handler can run in the main thread
    out.close()

    leak_check.check_reader_after(reader)
    # totals over every sitting, from the call log; `counts` is this sitting alone
    totals = tally_final_records(records_path)
    rows_completed = sum(totals.values())
    gate_summary = gate.summary()
    smoke_report = smoke_check(records_path, manifest, gate_summary, args.workers) if args.smoke else None
    if smoke_report:
        manifest.update(tokens_per_s_median=smoke_report["tokens_per_s_median"],
                        smoke_passed=smoke_report["passed"], smoke_failures=smoke_report["failures"],
                        smoke_report=smoke_report,
                        void=True, void_at=now_iso(), void_reason="smoke test: never feeds the views")
    manifest.update(gate_summary)
    manifest.update(rate_limit_headers_last=last_rate_limit_headers(records_path),
                    rows_requeued_on_outage=state["requeued_on_outage"],
                    state=("paused_on_network" if state["paused_on_network"] else
                           "stopped_on_permanent_error" if state["stopped_on_permanent_error"] else
                           "stopped_by_error" if errors else
                           "paused_by_owner" if state["paused_by_owner"] else
                           "stopped_at_budget_cap" if state["stopped_at_budget_cap"] else "finished"))
    if state["paused_by_owner"]:
        manifest.update(paused_at=now_iso(), paused_note="stopped by the owner (%s); rows in flight finished and "
                        "were logged; relaunch this run id with --resume" % state["paused_by_owner"])
        try:
            os.remove(stop_file)
        except OSError:
            pass
    manifest.update(finished_at=now_iso(), rows_ok=totals["ok"], rows_failed=totals["failed"],
                    rows_blocked=totals["blocked"], rows_excluded=totals["excluded"],
                    rows_ineligible=totals["ineligible"], rows_completed=rows_completed,
                    rows_this_sitting=dict(counts),
                    rows_not_started=len(rows) - rows_completed,
                    cost_usd_accumulated=round(state["spent"], 6),
                    litellm_cost_usd_accumulated=round(state["spent_litellm"], 6),
                    cost_gap_percent=(round(100.0 * (state["spent_litellm"] - state["spent"]) / state["spent"], 2)
                                      if state["spent"] else None),
                    stopped_at_budget_cap=state["stopped_at_budget_cap"],
                    stopped_on_permanent_error=state["stopped_on_permanent_error"],
                    worker_errors=errors,
                    tables_touched=sorted(reader.tables_touched),
                    post_run_leak_check="only market_moments was read")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    reader.close()
    print("done: %d ok, %d failed, %d excluded by benchmark flag, %d blocked by leak check, %d ineligible; "
          "cost $%.4f%s; log at %s"
          % (totals["ok"], totals["failed"], totals["excluded"], totals["blocked"], totals["ineligible"],
             state["spent"], " (STOPPED AT BUDGET CAP)" if state["stopped_at_budget_cap"] else "", records_path))
    if args.resume:
        print("  (the line above is the whole run, every sitting; this sitting alone: "
              "%d ok, %d failed, %d excluded, %d blocked, %d ineligible)"
              % (counts["ok"], counts["failed"], counts["excluded"], counts["blocked"], counts["ineligible"]))
    index_run(run_dir, run_id, None)
    if smoke_report:
        print("smoke test on host %s: %d workers, %d 429s (Retry-After %s), median latency %s s, median %s output "
              "tokens per second over %d calls; recommended workers %s (%s); %s"
              % (smoke_report["host"], smoke_report["workers_used"], smoke_report["rate_limit_429s"],
                 smoke_report["retry_after_values"], smoke_report["latency_s_median"],
                 smoke_report["tokens_per_s_median"], smoke_report["calls_timed"],
                 smoke_report["recommended_workers"]["workers"], smoke_report["recommended_workers"]["rule"],
                 "PASSED" if smoke_report["passed"] else "FAILED"))
        if smoke_report["openai_rate_limit_headers"]:
            print("  OpenAI rate-limit headers: %s (implied %s calls per minute)"
                  % (smoke_report["openai_rate_limit_headers"], smoke_report["openai_implied_calls_per_minute"]))
        for line in smoke_report["failures"]:
            print("  smoke failure:", line)
    if state["paused_on_network"]:
        print("PAUSED ON NETWORK: the host did not answer for %.0f minutes; %d rows not started; relaunch this run "
              "id with --resume when the network is back (exit code %d)"
              % (args.network_give_up_s / 60, len(rows) - rows_completed, EXIT_PAUSED_ON_NETWORK))
        sys.exit(EXIT_PAUSED_ON_NETWORK)
    if state["paused_by_owner"]:
        print("PAUSED BY OWNER (%s): %d rows not started; relaunch this run id with --resume (exit code %d)"
              % (state["paused_by_owner"], len(rows) - rows_completed, EXIT_PAUSED_BY_OWNER))
        sys.exit(EXIT_PAUSED_BY_OWNER)
    if state["stopped_on_permanent_error"]:
        raise SystemExit("run stopped on a permanent error (the same call would fail on every row): %s"
                         % state["stopped_on_permanent_error"][:300])
    if errors:
        raise SystemExit("run stopped by an error: %s" % errors[0])
    if smoke_report and not smoke_report["passed"]:
        raise SystemExit("SMOKE TEST FAILED: %d problem(s), listed above" % len(smoke_report["failures"]))


def index_run(run_dir: str, run_id: str, n_expected: Optional[int]) -> None:
    """Load this run into runs.db and rebuild the views. The folder is the record; an
    index failure must not lose it."""
    last_error = None
    for attempt in range(5):             # two runs finishing at once collide on the views; wait and retry
        try:
            import load_runs
            con = load_runs.connect()
            _, n_calls = load_runs.load_run(con, run_dir)
            problems = load_runs.build_views(con)
            con.close()
            print("runs.db: indexed %s (%d call rows)%s" % (run_id, n_calls,
                  "; GUARD: views not built, run `python3 harness/load_runs.py --check`" if problems else ""))
            return
        except Exception as e:
            last_error = e
            time.sleep(2.0 + attempt)
    print("runs.db: could not index this run (%s); run `python3 harness/load_runs.py --load %s`" % (last_error, run_id))


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 2)


def last_rate_limit_headers(records_path: str) -> Optional[Dict]:
    """The x-ratelimit-* headers on the last successful call (direct OpenAI reports them)."""
    last = None
    with open(records_path) as f:
        for line in f:
            if '"rate_limit_headers": {' in line:
                r = json.loads(line)
                if r.get("rate_limit_headers"):
                    last = r["rate_limit_headers"]
    return last


def recommended_workers(workers_used: int, n_429: int) -> Dict:
    """MAX_WORKERS if no 429 at MAX_WORKERS; else half the workers that produced 429s, never
    below 2. A clean run at fewer than MAX_WORKERS only shows that count is safe."""
    if n_429 == 0:
        if workers_used >= MAX_WORKERS:
            return {"workers": MAX_WORKERS, "rule": "no 429 at %d workers" % workers_used}
        return {"workers": workers_used, "rule": "no 429 at %d workers; the rule's first branch needs a run at %d"
                % (workers_used, MAX_WORKERS)}
    return {"workers": max(2, workers_used // 2), "rule": "%d 429s at %d workers: half, never below 2" % (n_429, workers_used)}


def smoke_check(records_path: str, manifest: Dict, gate_summary: Dict, workers_used: int) -> Dict:
    """Smoke-test verdict from the records: per host, the workers used, 429 count and the
    Retry-After values, median latency, median tokens per second (output tokens over wall
    time, on every record), the OpenAI rate-limit headers with the calls per minute they
    imply, a recommended worker count, and the list of failures: any call that finished
    with 'length', any call whose reply had no usable number, and any row that ended
    without a probability."""
    recs = []
    with open(records_path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    rates, failures, latencies = [], [], []
    for r in recs:
        if r.get("tokens_per_s") is not None:
            rates.append(r["tokens_per_s"])
        if r.get("raw_reply") is not None and r.get("latency_s"):
            latencies.append(r["latency_s"])
        if r.get("finish_reason") == "length":
            failures.append("row %s attempt %s: finish_reason 'length' (answer cut off)" % (r["row_id"], r["attempt"]))
        if r.get("raw_reply") is not None and r.get("parsed_probability") is None:
            failures.append("row %s attempt %s: reply had no usable probability (%s)"
                            % (r["row_id"], r["attempt"], r.get("error_class")))
        if (r.get("final") and r.get("parsed_probability") is None and r.get("raw_reply") is None
                and r.get("status") not in ("excluded_by_benchmark_flag", "leak_check_failed", "ineligible_release_date")):
            failures.append("row %s: %s (%s)" % (r["row_id"], r.get("status"), (r.get("error_message") or r.get("error") or "")[:120]))
    n_429 = gate_summary.get("rate_limit_429s", 0)
    headers = last_rate_limit_headers(records_path)
    implied_rpm = None
    if headers and headers.get("x-ratelimit-limit-requests"):
        try:
            implied_rpm = int(headers["x-ratelimit-limit-requests"])   # OpenAI's request limit is per minute
        except ValueError:
            pass
    return {"host": manifest.get("pinned_host_name") or manifest.get("pinned_host"),
            "host_slug": manifest.get("pinned_host"), "workers_used": workers_used,
            "rate_limit_429s": n_429, "retry_after_values": gate_summary.get("retry_after_values_seen"),
            "rate_limit_metadata": [{k: e.get(k) for k in ("error_type", "provider_code", "provider_name", "raw")}
                                    for e in gate_summary.get("rate_limit_events", [])],
            "latency_s_median": _median(latencies), "tokens_per_s_median": _median(rates),
            "calls_timed": len(rates), "openai_rate_limit_headers": headers,
            "openai_implied_calls_per_minute": implied_rpm,
            "recommended_workers": recommended_workers(workers_used, n_429),
            "network_outages": len(gate_summary.get("network_outages", [])),
            "passed": not failures, "failures": failures}


if __name__ == "__main__":
    main()
