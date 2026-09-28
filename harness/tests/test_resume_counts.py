"""A resumed run's manifest counts must be the whole run, not the last sitting.

Runs a short fake run twice over the same run id: the first sitting gets ten rows,
the second gets those ten plus ten more with --resume. One row is made to refuse
every time, so a failed row is carried across the resume as well.

Then it checks rows_ok / rows_failed / rows_blocked / rows_excluded / rows_completed
in run.json against the call log the run itself wrote.

Run it with:  .venv/bin/python harness/tests/test_resume_counts.py
No model is called and no money is spent (the fake adapter answers locally).
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.dirname(HERE)
sys.path.insert(0, HARNESS)

import load_runs            # noqa: E402
import run as run_module    # noqa: E402
from benchmark_reader import DEFAULT_DB  # noqa: E402

RUN_ID = "test_resume_counts"


def pick_rows(db_path, n):
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    ids = [r[0] for r in con.execute(
        "SELECT row_id FROM market_moments ORDER BY row_id LIMIT ?", (n,))]
    con.close()
    if len(ids) < n:
        raise SystemExit("need %d rows in market_moments, found %d" % (n, len(ids)))
    return ids


def call_log_counts(records_path):
    """The same tally, done independently of run.py: last final record per row id."""
    finals = {}
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("final"):
                finals[r["row_id"]] = r["status"]
    out = {"ok": 0, "failed": 0, "blocked": 0, "excluded": 0, "ineligible": 0}
    for status in finals.values():
        if status == "failed":
            out["failed"] += 1
        elif status == "leak_check_failed":
            out["blocked"] += 1
        elif status == "excluded_by_benchmark_flag":
            out["excluded"] += 1
        elif status == "ineligible_release_date":
            out["ineligible"] += 1
        else:
            out["ok"] += 1
    return out, len(finals)


def sitting(tmp, ids, resume, refuse_row):
    ids_file = os.path.join(tmp, "ids_%d.txt" % len(ids))
    with open(ids_file, "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    argv = ["run.py", "--mode", "normal", "--model", "fake:0.5:refuse_rows=%d" % refuse_row,
            "--db", DEFAULT_DB, "--row-ids-file", ids_file, "--run-id", RUN_ID,
            "--workers", "2", "--transport-backoff", ""]
    if resume:
        argv.append("--resume")
    old_argv, sys.argv = sys.argv, argv
    try:
        run_module.main()
    finally:
        sys.argv = old_argv


def main():
    failures = []
    tmp = tempfile.mkdtemp(prefix="resume-counts-")
    runs_dir = os.path.join(tmp, "runs")
    os.makedirs(runs_dir)
    # keep the real harness/runs untouched: the run folder and runs.db go to a temp dir
    run_module.RUNS_DIR = runs_dir
    load_runs.RUNS_DIR = runs_dir
    load_runs.DB_PATH = os.path.join(runs_dir, "runs.db")
    try:
        ids = pick_rows(DEFAULT_DB, 20)
        refuse_row = ids[0]                      # fails in the first sitting and again in the second
        sitting(tmp, ids[:10], resume=False, refuse_row=refuse_row)
        first = json.load(open(os.path.join(runs_dir, RUN_ID, "run.json")))
        sitting(tmp, ids, resume=True, refuse_row=refuse_row)

        run_dir = os.path.join(runs_dir, RUN_ID)
        manifest = json.load(open(os.path.join(run_dir, "run.json")))
        log, distinct_final_rows = call_log_counts(os.path.join(run_dir, "records.jsonl"))

        if distinct_final_rows != 20:
            failures.append("the call log has %d rows with a final record, expected 20" % distinct_final_rows)
        if first["rows_completed"] != 10:
            failures.append("the first sitting finished %d rows, expected 10" % first["rows_completed"])
        for key, column in (("ok", "rows_ok"), ("failed", "rows_failed"),
                            ("blocked", "rows_blocked"), ("excluded", "rows_excluded"),
                            ("ineligible", "rows_ineligible")):
            if manifest[column] != log[key]:
                failures.append("manifest %s = %s, call log says %d" % (column, manifest[column], log[key]))
        total = sum(manifest[c] for c in ("rows_ok", "rows_failed", "rows_blocked",
                                          "rows_excluded", "rows_ineligible"))
        if total != distinct_final_rows:
            failures.append("manifest counts add to %d, the call log has %d rows" % (total, distinct_final_rows))
        if manifest["rows_completed"] != distinct_final_rows:
            failures.append("rows_completed = %s, the call log has %d rows"
                            % (manifest["rows_completed"], distinct_final_rows))
        if manifest["rows_completed"] <= first["rows_completed"]:
            failures.append("the resumed manifest did not carry the first sitting forward")
        if manifest["rows_failed"] != 1:
            failures.append("expected the one refusing row to be failed, got %s" % manifest["rows_failed"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("FAILED:")
        for line in failures:
            print("  " + line)
        return 1
    print("ok: a resumed run's manifest counts match its call log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
