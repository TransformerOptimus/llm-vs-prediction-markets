"""Loader for harness output: harness/runs/runs.db -> scored rows.

Reads only; never writes to benchmark.db, runs.db or the run folders.

runs.db (see harness/RUNS_SCHEMA.md) indexes every run folder. Two read paths:

  Scoring path (paper tables): the `forecasts` view (normal pass: the latest final record
  per model and row from non-void runs) and the `probe_forecasts` view (the memory-probe
  record where one exists, else the normal record with source = normal_stand_in, the rule
  for rows without news, where the probe is not run). load_pass() reads them.

  By-run-id path (regression tests and test runs, never a paper table): the final records of
  one run from the `calls` table, void or not. load_run() reads it.

Joining: records are matched to benchmark rows by `moment_key`, which stays valid if row ids change;
`row_id` is used only for records without a key. A record whose moment_key resolves to a
row whose row_id disagrees with the record's row_id is counted as moment_key_mismatch and
skipped. If a run's db_sha256 differs from the live benchmark file's hash AND any of its
keys fails to resolve, that run is refused. A record whose `window` disagrees with the
row's window is counted as window_mismatch and skipped.

Statuses: SKIP_STATUSES are rows the harness excluded before any model call; they are
counted as skipped_<status>, never as failures. failed_share = failed / model_called.
Forecaster key = model + '@' + template (the record's template, else the run's template_id).
Two arms: arm "news" keeps rows with news_available = 1; arm "all" keeps every row.
Memory-probe quarantine: apply_quarantine=True drops rows flagged for the model in benchmark
quarantine_flags (keyed on moment_key); Experiment B tables only. Ladder rows are returned
separately; dateline-flagged rows are never scored.
"""
import hashlib
import json
import os
import sqlite3
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from scoring import DB_PATH, REPO, in_arm, load_quarantine, score_forecast

RUNS_DIR = os.path.join(REPO, "harness", "runs")
RUNS_DB = os.path.join(RUNS_DIR, "runs.db")
SKIP_STATUSES = ("excluded_by_benchmark_flag", "leak_check_failed", "ineligible_release_date")
RECORD_COLS = ("run_id", "mode", "model", "template", "row_id", "moment_key", "window", "status", "parsed_probability", "timestamp")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def connect_runs(path: str = RUNS_DB) -> sqlite3.Connection:
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    con.row_factory = sqlite3.Row
    return con


def read_manifests(rcon: sqlite3.Connection) -> Dict[str, dict]:
    """run_id -> the run's row from `runs`, with the parsed manifest under 'manifest'."""
    out = {}
    for r in rcon.execute("SELECT * FROM runs"):
        d = dict(r)
        try:
            d["manifest"] = json.loads(d.get("manifest_json") or "{}")
        except ValueError:
            d["manifest"] = {}
        out[d["run_id"]] = d
    return out


def read_run_records(rcon: sqlite3.Connection, run_id: str) -> List[dict]:
    """By-run-id path: the final records of one run from `calls`, void or not, latest per row."""
    rows = rcon.execute("SELECT %s FROM calls WHERE run_id = ? AND final = 1 ORDER BY timestamp, call_id"
                        % ", ".join(RECORD_COLS), (run_id,)).fetchall()
    latest = {}
    for r in rows:
        d = dict(r)
        latest[d["moment_key"] or ("row:%s" % d["row_id"])] = d      # the last final record for a row wins
    return list(latest.values())


def read_pass(rcon: sqlite3.Connection, pass_: str, models: Optional[List[str]] = None,
              window: Optional[str] = None) -> List[dict]:
    """Scoring path: 'normal' from the forecasts view (mode normal), 'probe' from probe_forecasts."""
    if pass_ == "normal":
        sql = "SELECT %s, 'normal' AS source FROM forecasts WHERE mode = 'normal'" % ", ".join(RECORD_COLS)
    elif pass_ == "probe":
        sql = "SELECT %s, source FROM probe_forecasts WHERE 1 = 1" % ", ".join(RECORD_COLS)
    else:
        raise ValueError(pass_)
    params: list = []
    if models:
        sql += " AND model IN (%s)" % ",".join("?" * len(models))
        params += list(models)
    if window:
        sql += " AND window = ?"
        params.append(window)
    return [dict(r) for r in rcon.execute(sql, params)]


def score_records(records: List[dict], run_meta: Dict[str, dict], moments: Dict[int, dict], cuts,
                  quarantine: Optional[Dict] = None, arm: str = "all", apply_quarantine: bool = False,
                  by_key: Optional[Dict[str, dict]] = None, live_db_sha: Optional[str] = None,
                  template_override: Optional[Dict[str, str]] = None, include_ladders: bool = False) -> dict:
    """The join, the checks and the scoring, over a list of records from either path.
    Returns rows, ladder_rows, quarantined_rows and counts (per run_id under counts_by_run)."""
    if by_key is None:
        by_key = {m["moment_key"]: m for m in moments.values()}
    quarantine = quarantine or {}
    counts_by_run: Dict[str, Counter] = {}
    rows, ladder_rows, quarantined_rows = [], [], []
    unresolved: Counter = Counter()
    resolved = []
    for rec in records:
        c = counts_by_run.setdefault(rec["run_id"], Counter())
        c["final_records"] += 1
        m = None
        if rec.get("moment_key"):
            m = by_key.get(rec["moment_key"])
            if m is not None and rec.get("row_id") is not None and m["row_id"] != rec["row_id"]:
                c["moment_key_mismatch"] += 1
                continue
        elif rec.get("row_id") is not None:
            m = moments.get(rec["row_id"])
            c["joined_by_row_id_only"] += 1
        if m is None:
            unresolved[rec["run_id"]] += 1
            c["key_not_in_benchmark"] += 1
            continue
        resolved.append((rec, m))
    if unresolved:
        live = live_db_sha or sha256_file(DB_PATH)
        for rid, n in unresolved.items():
            run_sha = (run_meta.get(rid) or {}).get("db_sha256")
            if run_sha and run_sha != live:
                raise RuntimeError("run %s: %d records do not resolve and the benchmark hash changed since the run "
                                   "(run %s, live %s); refusing to score" % (rid, n, run_sha[:12], live[:12]))
    for rec, m in resolved:
        c = counts_by_run[rec["run_id"]]
        status = rec.get("status")
        if status in SKIP_STATUSES:
            c["skipped_" + status] += 1
            continue
        if rec.get("window") and rec["window"] != m["window"]:
            c["window_mismatch"] += 1
            continue
        c["model_called"] += 1
        if rec.get("parsed_probability") is None:
            c["failed_rows"] += 1
            continue
        if m["news_dateline_after_capture"]:
            c["dateline_flagged"] += 1
            continue
        if not in_arm(m, arm):
            c["outside_arm"] += 1
            continue
        meta = run_meta.get(rec["run_id"]) or {}
        template = (template_override or {}).get(rec["run_id"]) or rec.get("template") or meta.get("template_id") or "unknown"
        model = rec["model"]
        forecaster = "%s@%s" % (model, template)
        source = rec.get("source") or rec.get("mode")
        row = score_forecast(m, float(rec["parsed_probability"]), forecaster, cuts,
                             extra={"run_id": rec["run_id"], "mode": rec.get("mode"), "template": template, "model_name": model,
                                    "arm": arm, "record_source": "normal_as_probe" if source == "normal_stand_in" else source})
        if quarantine.get((m["moment_key"], model), 0):
            c["quarantined"] += 1
            quarantined_rows.append(row)
            if apply_quarantine:
                continue
        if m["is_ladder"]:
            c["ladder"] += 1
            ladder_rows.append(row)
            if not include_ladders:
                continue
        rows.append(row)
        c["scored"] += 1
    for c in counts_by_run.values():
        c["failed_share"] = c["failed_rows"] / max(1, c["model_called"])
    return {"rows": rows, "ladder_rows": ladder_rows, "quarantined_rows": quarantined_rows,
            "counts_by_run": {k: dict(v) for k, v in counts_by_run.items()}}


def load_run(run_id: str, moments: Dict[int, dict], cuts, quarantine: Optional[Dict] = None, template_override=None,
             runs_db: str = RUNS_DB, arm: str = "all", apply_quarantine: bool = False,
             by_key: Optional[Dict[str, dict]] = None, live_db_sha: Optional[str] = None) -> dict:
    """By-run-id path (never for paper tables): score one run's final records, void or not."""
    rcon = connect_runs(runs_db)
    meta = read_manifests(rcon)
    if run_id not in meta:
        raise KeyError("run %s is not in %s" % (run_id, runs_db))
    recs = read_run_records(rcon, run_id)
    r = score_records(recs, meta, moments, cuts, quarantine, arm, apply_quarantine, by_key, live_db_sha, template_override)
    man = meta[run_id]["manifest"] or {}
    man.setdefault("run_id", run_id)
    man.setdefault("mode", meta[run_id]["mode"])
    man.setdefault("model", meta[run_id]["model"])
    return {"rows": r["rows"], "ladder_rows": r["ladder_rows"], "quarantined_rows": r["quarantined_rows"],
            "counts": r["counts_by_run"].get(run_id, {}), "manifest": man,
            "template": meta[run_id].get("template_id") or "unknown", "arm": arm, "void": bool(meta[run_id].get("void"))}


def load_pass(pass_: str, moments: Dict[int, dict], cuts, models: Optional[List[str]] = None, window: Optional[str] = None,
              arm: str = "all", apply_quarantine: bool = False, con=None, runs_db: str = RUNS_DB,
              include_ladders: bool = False) -> Tuple[List[dict], Dict[str, dict]]:
    """Scoring path: 'normal' or 'probe' pass from the views. Returns (rows, counts_by_run).
    With include_ladders the ladder rows are kept in `rows` (the quarantine rule assesses them too)."""
    rcon = connect_runs(runs_db)
    meta = read_manifests(rcon)
    recs = read_pass(rcon, pass_, models, window)
    quarantine = load_quarantine(con, moments) if con is not None else {}
    r = score_records(recs, meta, moments, cuts, quarantine, arm, apply_quarantine, live_db_sha=sha256_file(DB_PATH),
                      include_ladders=include_ladders)
    return r["rows"], r["counts_by_run"]


def load_runs(run_ids: Iterable[str], con, moments, cuts, template_override=None, mode: str = "normal",
              arm: str = "all", apply_quarantine: bool = False, runs_db: str = RUNS_DB) -> Tuple[List[dict], Dict[str, dict]]:
    """By-run-id path over several runs of one mode (regression tests and test runs only)."""
    quarantine = load_quarantine(con, moments)
    by_key = {m["moment_key"]: m for m in moments.values()}
    live_sha = sha256_file(DB_PATH)
    out, counts_by_run = [], {}
    for rid in run_ids:
        r = load_run(rid, moments, cuts, quarantine, template_override, runs_db=runs_db, arm=arm,
                     apply_quarantine=apply_quarantine, by_key=by_key, live_db_sha=live_sha)
        if r["manifest"]["mode"] != mode:
            counts_by_run[rid] = {"skipped_run_wrong_mode": r["manifest"]["mode"]}
            continue
        out.extend(r["rows"])
        counts_by_run[rid] = r["counts"]
    return out, counts_by_run


def stand_in(normal_rows: List[dict], probe_run_rows: List[dict]) -> List[dict]:
    """By-run-id path only: the normal record stands in on rows without news that the probe
    did not cover. The probe_forecasts view does this for the scoring path."""
    have = {(r["forecaster"], r["moment_key"]) for r in probe_run_rows}
    out = list(probe_run_rows)
    for r in normal_rows:
        if (r["forecaster"], r["moment_key"]) in have or r["news_available"]:
            continue
        x = dict(r)
        x["record_source"] = "normal_as_probe"
        out.append(x)
    return out
