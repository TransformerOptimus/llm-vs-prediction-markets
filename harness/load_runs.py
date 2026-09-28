"""runs.db: a SQLite index of every run folder under harness/runs/, built from the folders.

The folders (run.json + records.jsonl) stay the raw record; run.py writes them and
calls load_run() for its own run when it finishes. Loading a run_id
that is already in the database replaces that run's rows, so a resumed run reloads cleanly.

  python3 harness/load_runs.py --rebuild                 recreate runs.db from every folder
  python3 harness/load_runs.py --load <run_id>           load or reload one run
  python3 harness/load_runs.py --void <run_id> --reason "..."   mark a run void (run.json), reload
  python3 harness/load_runs.py --check                   run the settings guard and (re)build the views

Tables, views and the guard are documented in RUNS_SCHEMA.md. Nothing here reads
benchmark.db.
"""
import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
DB_PATH = os.path.join(RUNS_DIR, "runs.db")
SCHEMA_VERSION = 1

RUN_COLUMNS = [
    "run_id", "mode", "model", "model_spec", "window", "template", "template_id", "prompt_template_hash",
    "db_sha256", "git_commit", "git_tree", "litellm_version", "started_at", "finished_at", "rows_planned",
    "rows_ok", "rows_failed", "rows_blocked", "rows_excluded", "rows_completed", "rows_not_started",
    "cost_usd", "litellm_cost_usd", "cost_gap_percent", "stopped_at_budget_cap", "stopped_on_permanent_error",
    "void", "manifest_json", "pinned_host", "host_override", "expected_served_model", "dry_run", "smoke",
    "tokens_per_s_median", "state", "rate_limit_429s", "network_outages_count", "rows_requeued_on_outage",
    "seconds_paused_total", "rows_per_min_outside_pauses", "resume_commit",
]
# (column, record key) for the scalar fields of a record; JSON columns are listed separately.
CALL_SCALARS = [
    ("run_id", "run_id"), ("mode", "mode"), ("model", "model"), ("template", "template"), ("row_id", "row_id"),
    ("moment_key", "moment_key"), ("window", "window"), ("venue", "venue"), ("venue_market_id", "venue_market_id"),
    ("forecast_ts", "forecast_ts"), ("book_source", "book_source"), ("attempt", "attempt"),
    ("transport_try", "transport_try"), ("strict", "strict"), ("final", "final"), ("status", "status"),
    ("error_class", "error_class"), ("error", "error"), ("prompt_sha256", "prompt_sha256"), ("raw_reply", "raw_reply"),
    ("parsed_probability", "parsed_probability"), ("parse_warning", "parse_warning"), ("timestamp", "timestamp"),
    ("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"), ("cost_usd", "cost_usd"),
    ("litellm_cost_usd", "litellm_cost_usd"), ("finish_reason", "finish_reason"), ("provider", "provider"),
    ("served_model", "served_model"), ("latency_s", "latency_s"), ("price_shown_from", "price_shown_from"),
    ("description_paragraphs_stripped", "description_paragraphs_stripped"), ("tokens_per_s", "tokens_per_s"),
]
CALL_JSON = ["request_params", "adapter_raw", "leak_check", "description_stripped_text", "rate_limit", "rate_limit_headers"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, mode TEXT, model TEXT, model_spec TEXT, window TEXT, template TEXT, template_id TEXT,
  prompt_template_hash TEXT, db_sha256 TEXT, git_commit TEXT, git_tree TEXT, litellm_version TEXT,
  started_at TEXT, finished_at TEXT, rows_planned INTEGER, rows_ok INTEGER, rows_failed INTEGER,
  rows_blocked INTEGER, rows_excluded INTEGER, rows_completed INTEGER, rows_not_started INTEGER,
  cost_usd REAL, litellm_cost_usd REAL, cost_gap_percent REAL, stopped_at_budget_cap INTEGER,
  stopped_on_permanent_error TEXT, void INTEGER NOT NULL DEFAULT 0, manifest_json TEXT NOT NULL,
  pinned_host TEXT, host_override TEXT, expected_served_model TEXT, dry_run INTEGER, smoke INTEGER,
  tokens_per_s_median REAL, state TEXT, rate_limit_429s INTEGER, network_outages_count INTEGER,
  seconds_paused_total REAL, rows_per_min_outside_pauses REAL, resume_commit TEXT,
  rows_requeued_on_outage INTEGER
);
CREATE TABLE IF NOT EXISTS calls (
  call_id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), line_no INTEGER NOT NULL,
  mode TEXT, model TEXT, template TEXT, row_id INTEGER, moment_key TEXT, window TEXT, venue TEXT,
  venue_market_id TEXT, forecast_ts TEXT, book_source TEXT, attempt INTEGER, transport_try INTEGER,
  strict INTEGER, final INTEGER, status TEXT, error_class TEXT, error TEXT, prompt_sha256 TEXT, raw_reply TEXT,
  parsed_probability REAL, parse_warning TEXT, timestamp TEXT, input_tokens INTEGER, output_tokens INTEGER,
  reasoning_tokens INTEGER, cost_usd REAL, litellm_cost_usd REAL, finish_reason TEXT, provider TEXT,
  served_model TEXT, latency_s REAL, price_shown_from TEXT, description_paragraphs_stripped INTEGER,
  tokens_per_s REAL, request_params TEXT, adapter_raw TEXT, leak_check TEXT, description_stripped_text TEXT,
  rate_limit TEXT, rate_limit_headers TEXT
);
CREATE INDEX IF NOT EXISTS calls_run_row ON calls(run_id, row_id);
CREATE INDEX IF NOT EXISTS calls_model_mode_key ON calls(model, mode, moment_key);
CREATE INDEX IF NOT EXISTS calls_key ON calls(moment_key);
"""

VIEWS = """
CREATE VIEW forecasts AS
  SELECT * FROM (
    SELECT c.*, r.started_at AS run_started_at,
           ROW_NUMBER() OVER (PARTITION BY c.model, c.mode, c.row_id
                              ORDER BY c.timestamp DESC, r.started_at DESC, c.call_id DESC) AS rn
    FROM calls c JOIN runs r ON r.run_id = c.run_id
    WHERE c.final = 1 AND r.void = 0
  ) WHERE rn = 1;
CREATE VIEW probe_forecasts AS
  SELECT p.*, 'memory_probe' AS source FROM forecasts p WHERE p.mode = 'memory_probe'
  UNION ALL
  SELECT n.*, 'normal_stand_in' AS source FROM forecasts n
  WHERE n.mode = 'normal' AND NOT EXISTS (
    SELECT 1 FROM forecasts p WHERE p.mode = 'memory_probe' AND p.model = n.model AND p.row_id = n.row_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    # DB_PATH is read here, not bound as a default at import time, so a caller that
    # repoints the module (a test running against a temp runs dir) really does get the
    # temp database instead of silently writing into the real one.
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for table in ("runs", "calls"):     # an older file lacks columns added since: rebuild it
        have = {r[1] for r in con.execute("PRAGMA table_info(%s)" % table)}
        want = set(RUN_COLUMNS) if table == "runs" else {c for c, _ in CALL_SCALARS} | {"call_id", "line_no", "reasoning_tokens"} | set(CALL_JSON)
        missing = sorted(want - have)
        if missing:
            con.close()
            raise RuntimeError("%s: table %s lacks columns %s; run `python3 harness/load_runs.py --rebuild`"
                               % (path, table, missing))
    if con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
        con.execute("INSERT INTO schema_version VALUES (?, ?)", (SCHEMA_VERSION, now_iso()))
    con.commit()
    return con


def _j(v) -> Optional[str]:
    return None if v is None else json.dumps(v, ensure_ascii=False)


def _scalar(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return int(v)
    return v


def load_run(con: sqlite3.Connection, run_dir: str) -> Tuple[str, int]:
    """Load (or reload) one run folder. Returns (run_id, number of call rows)."""
    with open(os.path.join(run_dir, "run.json")) as f:
        manifest_text = f.read()
    m = json.loads(manifest_text)
    run_id = m["run_id"]
    row = {k: m.get(k) for k in RUN_COLUMNS}
    row.update(cost_usd=m.get("cost_usd_accumulated"), litellm_cost_usd=m.get("litellm_cost_usd_accumulated"),
               void=1 if m.get("void") else 0, manifest_json=manifest_text,
               dry_run=1 if m.get("dry_run") else 0, smoke=1 if m.get("smoke") else 0,
               network_outages_count=len(m.get("network_outages") or []),
               stopped_at_budget_cap=int(bool(m.get("stopped_at_budget_cap"))) if m.get("stopped_at_budget_cap") is not None else None)
    con.execute("DELETE FROM calls WHERE run_id = ?", (run_id,))
    con.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    con.execute("INSERT INTO runs (%s) VALUES (%s)" % (", ".join(RUN_COLUMNS), ", ".join("?" * len(RUN_COLUMNS))),
                [_scalar(row[k]) for k in RUN_COLUMNS])
    cols = ["run_id", "line_no"] + [c for c, _ in CALL_SCALARS if c != "run_id"] + ["reasoning_tokens"] + CALL_JSON
    sql = "INSERT INTO calls (%s) VALUES (%s)" % (", ".join(cols), ", ".join("?" * len(cols)))
    n = 0
    records_path = os.path.join(run_dir, "records.jsonl")
    if os.path.exists(records_path):
        with open(records_path) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                vals = [run_id, line_no] + [_scalar(r.get(k)) for c, k in CALL_SCALARS if c != "run_id"]
                vals.append((r.get("adapter_raw") or {}).get("reasoning_tokens_used"))
                vals += [_j(r.get(k)) for k in CALL_JSON]
                con.execute(sql, vals)
                n += 1
    con.commit()
    return run_id, n


def run_dirs() -> List[str]:
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for name in sorted(os.listdir(RUNS_DIR)):
        d = os.path.join(RUNS_DIR, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "run.json")):
            out.append(d)
    return out


def _settings_signature(con: sqlite3.Connection, run_id: str, manifest: Dict) -> str:
    """The request settings that must agree across the runs feeding one (model, mode,
    window): thinking cap or effort, temperature, token limits, host or provider order.
    Taken from request_params on the calls (what was really sent) and, where no call
    carried them, from the manifest's settings."""
    seen = set()
    for (rp,) in con.execute("SELECT DISTINCT request_params FROM calls WHERE run_id = ? AND request_params IS NOT NULL", (run_id,)):
        p = json.loads(rp)
        extra = p.get("extra_body") or {}
        sig = {"thinking": p.get("thinking") or (extra.get("reasoning") or {}).get("max_tokens")
               or (extra.get("chat_template_args") or {}).get("enable_thinking"),
               "reasoning_effort": p.get("reasoning_effort") or (extra.get("reasoning") or {}).get("effort"), "temperature": p.get("temperature"),
               "max_tokens": p.get("max_tokens") or p.get("max_completion_tokens"),
               "provider_order": (extra.get("provider") or {}).get("order")}
        seen.add(json.dumps(sig, sort_keys=True))
    if not seen:
        s = manifest.get("settings") or {}
        sig = {"thinking": s.get("reasoning_tokens"), "reasoning_effort": s.get("reasoning_effort"),
               "temperature": s.get("temperature"),
               "max_tokens": (s.get("answer_tokens") or 0) + (s.get("reasoning_tokens") or 0) if s else None,
               "provider_order": [manifest["pinned_host"]] if manifest.get("pinned_host") not in (None, "direct", "fake") else None}
        seen.add(json.dumps(sig, sort_keys=True))
    return " | ".join(sorted(seen))


def guard(con: sqlite3.Connection) -> List[str]:
    """Runs whose non-void records feeding one (model, mode, window) disagree on the
    template hash, the benchmark hash, or the request settings. Empty list = fine."""
    groups = defaultdict(list)
    for run_id, model, mode, window, th, dbh, mj in con.execute(
            "SELECT run_id, model, mode, window, prompt_template_hash, db_sha256, manifest_json FROM runs WHERE void = 0"):
        groups[(model, mode, window)].append((run_id, th, dbh, _settings_signature(con, run_id, json.loads(mj))))
    problems = []
    for key, members in groups.items():
        for label, idx in (("prompt_template_hash", 1), ("db_sha256", 2), ("request settings", 3)):
            values = defaultdict(list)
            for mem in members:
                values[mem[idx]].append(mem[0])
            if len(values) > 1:
                problems.append("(model=%s, mode=%s, window=%s) disagree on %s: %s" % (
                    key[0], key[1], key[2], label,
                    "; ".join("%s -> %s" % (str(v)[:80], sorted(ids)) for v, ids in values.items())))
    return problems


def count_check(con: sqlite3.Connection) -> Tuple[List[str], int]:
    """Runs whose manifest row counts disagree with their own call log.

    rows_ok + rows_failed + rows_blocked + rows_excluded + rows_ineligible must equal
    the number of distinct row ids that have a final call. This catches a resumed run
    that wrote only its last sitting's counts. Returns (problems, runs checked).
    """
    logged = dict(con.execute(
        "SELECT run_id, COUNT(DISTINCT row_id) FROM calls WHERE final = 1 GROUP BY run_id"))
    problems = []
    checked = 0
    for run_id, ok, failed, blocked, excluded, mj in con.execute(
            "SELECT run_id, rows_ok, rows_failed, rows_blocked, rows_excluded, manifest_json FROM runs"):
        manifest = json.loads(mj)
        if manifest.get("dry_run"):
            continue                      # a dry run makes no calls and logs no rows
        checked += 1
        parts = [ok or 0, failed or 0, blocked or 0, excluded or 0, manifest.get("rows_ineligible") or 0]
        n_log = logged.get(run_id, 0)
        if sum(parts) != n_log:
            problems.append("%s: manifest counts add to %d (ok %d, failed %d, blocked %d, excluded %d, "
                            "ineligible %d) but the call log has %d rows with a final attempt"
                            % (run_id, sum(parts), parts[0], parts[1], parts[2], parts[3], parts[4], n_log))
    return problems, checked


def build_views(con: sqlite3.Connection) -> List[str]:
    con.executescript("DROP VIEW IF EXISTS probe_forecasts; DROP VIEW IF EXISTS forecasts;")
    problems = guard(con)
    if problems:
        con.commit()
        return problems
    con.executescript(VIEWS)
    con.commit()
    return []


def void_run(run_id: str, reason: str) -> str:
    path = os.path.join(RUNS_DIR, run_id, "run.json")
    with open(path) as f:
        m = json.load(f)
    m["void"] = True
    m["void_at"] = now_iso()
    m["void_reason"] = reason
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
    return path


def rebuild() -> sqlite3.Connection:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = connect()
    for d in run_dirs():
        rid, n = load_run(con, d)
        print("loaded %-40s %6d call rows" % (rid, n))
    return con


def report_guard(problems: List[str], counts: Optional[Tuple[List[str], int]] = None) -> int:
    bad = False
    if problems:
        print("GUARD: views not built; runs feeding one (model, mode, window) disagree:")
        for p in problems:
            print("  " + p)
        bad = True
    else:
        print("guard ok: views forecasts and probe_forecasts built")
    if counts is not None:
        count_problems, checked = counts
        if count_problems:
            print("ROW COUNTS: %d of %d runs disagree with their own call log:" % (len(count_problems), checked))
            for p in count_problems:
                print("  " + p)
            bad = True
        else:
            print("row counts ok: 0 of %d runs disagree with their own call log" % checked)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rebuild", action="store_true")
    g.add_argument("--load", metavar="RUN_ID")
    g.add_argument("--void", metavar="RUN_ID")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--reason", default=None, help="why the run is void (required with --void)")
    args = ap.parse_args()
    try:
        if args.rebuild:
            con = rebuild()
        elif args.load:
            con = connect()
            rid, n = load_run(con, os.path.join(RUNS_DIR, args.load))
            print("loaded %s: %d call rows" % (rid, n))
        elif args.void:
            if not args.reason:
                raise SystemExit("--void needs --reason")
            path = void_run(args.void, args.reason)
            con = connect()
            load_run(con, os.path.dirname(path))
            print("void: %s (%s); reloaded" % (args.void, args.reason))
        else:
            con = connect()
    except RuntimeError as e:          # an out-of-date runs.db file
        raise SystemExit(str(e))
    sys.exit(report_guard(build_views(con), count_check(con)))


if __name__ == "__main__":
    main()
