"""A reply the host refuses for running past the reply limit must not stop the run.

Direct OpenAI can do this: a row's reply runs past the 4,000 token limit and the host
sends back a 400 error instead of the truncated text. It arrives as a BadRequestError,
which the classifier calls permanent, so without special handling the whole run would
stop. It is a row-level problem, not a broken request.

The harness sends it down the same path as a truncated reply: the strict retry gets a go
(it asks for a bare number, so the reply is short) and only that row fails if the strict
retry fails too.

Three things are checked, with the fake adapter so no model is called and nothing is spent:
  1. the recognizer fires on the host's message and not on the other error kinds
  2. a row that hits the limit once is rescued by the strict retry, and the run finishes
  3. a row that hits it on both attempts fails alone, and the run still finishes

Run it with:  .venv/bin/python harness/tests/test_output_limit.py
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
from adapters.base import classify_error, is_output_limit_error  # noqa: E402
from adapters.fake import FakeBadRequest  # noqa: E402
from benchmark_reader import DEFAULT_DB   # noqa: E402

HOST_MESSAGE = ("litellm.BadRequestError: OpenAIException - Could not finish the message because "
                "max_tokens or model output limit was reached. Please try again with higher max_tokens.")


def pick_rows(db_path, n):
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    ids = [r[0] for r in con.execute("SELECT row_id FROM market_moments ORDER BY row_id LIMIT ?", (n,))]
    con.close()
    return ids


def do_run(tmp, runs_dir, ids, spec, run_id):
    ids_file = os.path.join(tmp, run_id + ".txt")
    with open(ids_file, "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    argv = ["run.py", "--mode", "normal", "--model", spec, "--db", DEFAULT_DB,
            "--row-ids-file", ids_file, "--run-id", run_id, "--workers", "2",
            "--transport-backoff", ""]
    old, sys.argv = sys.argv, argv
    code = 0
    try:
        run_module.main()
    except SystemExit as e:
        code = e.code or 0
    finally:
        sys.argv = old
    manifest = json.load(open(os.path.join(runs_dir, run_id, "run.json")))
    records = [json.loads(l) for l in open(os.path.join(runs_dir, run_id, "records.jsonl"))]
    return code, manifest, records


def main():
    failures = []

    # 1. the recognizer
    if classify_error(FakeBadRequest(HOST_MESSAGE)) != "permanent":
        failures.append("the classifier no longer calls this error permanent, so the test proves nothing")
    if not is_output_limit_error(FakeBadRequest(HOST_MESSAGE)):
        failures.append("is_output_limit_error did not fire on the host's own message")
    for other in ("This model is only available on agentic harnesses",
                  "You have no credits remaining. Add credits to continue using the API.",
                  "Invalid API key provided"):
        if is_output_limit_error(Exception(other)):
            failures.append("is_output_limit_error fired on an unrelated error: %s" % other)

    tmp = tempfile.mkdtemp(prefix="output-limit-")
    runs_dir = os.path.join(tmp, "runs")
    os.makedirs(runs_dir)
    run_module.RUNS_DIR = runs_dir
    load_runs.RUNS_DIR = runs_dir
    load_runs.DB_PATH = os.path.join(runs_dir, "runs.db")
    try:
        ids = pick_rows(DEFAULT_DB, 6)
        hit = ids[2]

        # 2. the limit is hit on the normal attempt only: the strict retry rescues the row
        code, m, recs = do_run(tmp, runs_dir, ids, "fake:0.5:outlimit_rows=%d" % hit, "test_outlimit_rescued")
        if code != 0:
            failures.append("the run exited %s; it should finish normally" % code)
        if m["state"] != "finished":
            failures.append("run state is %r, expected finished" % m["state"])
        if m.get("stopped_on_permanent_error"):
            failures.append("the run stopped on a permanent error: %s" % str(m["stopped_on_permanent_error"])[:120])
        if m["rows_ok"] != len(ids) or m["rows_failed"] != 0:
            failures.append("expected %d ok and 0 failed, got %s ok and %s failed"
                            % (len(ids), m["rows_ok"], m["rows_failed"]))
        marked = [r for r in recs if r.get("error_class") == "output_limit"]
        if len(marked) != 1 or marked[0]["row_id"] != hit:
            failures.append("expected one output_limit record on row %s, got %s"
                            % (hit, [(r["row_id"], r["error_class"]) for r in marked]))
        if marked and marked[0].get("final"):
            failures.append("the output_limit record should not be the row's final record when the retry works")
        final = {r["row_id"]: r for r in recs if r.get("final")}
        if final[hit]["status"] != "ok_after_retry" or final[hit]["parsed_probability"] is None:
            failures.append("row %s should have been rescued by the strict retry, got %s"
                            % (hit, final[hit]["status"]))

        # 3. the limit is hit on both attempts: that row fails alone and the run still finishes
        code, m, recs = do_run(tmp, runs_dir, ids, "fake:0.5:outlimit_both=%d" % hit, "test_outlimit_failed")
        if code != 0:
            failures.append("the both-attempts run exited %s; it should finish normally" % code)
        if m["state"] != "finished":
            failures.append("both-attempts run state is %r, expected finished" % m["state"])
        if m["rows_failed"] != 1 or m["rows_ok"] != len(ids) - 1:
            failures.append("expected 1 failed and %d ok, got %s failed and %s ok"
                            % (len(ids) - 1, m["rows_failed"], m["rows_ok"]))
        if m["rows_completed"] != len(ids):
            failures.append("expected all %d rows completed, got %s" % (len(ids), m["rows_completed"]))
        final = {r["row_id"]: r for r in recs if r.get("final")}
        if final[hit]["status"] != "failed" or final[hit].get("error_class") != "output_limit":
            failures.append("row %s should be failed with error_class output_limit, got %s / %s"
                            % (hit, final[hit]["status"], final[hit].get("error_class")))
        if any(r["status"] == "failed" for rid, r in final.items() if rid != hit):
            failures.append("a row other than %s failed; only the offending row should" % hit)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("FAILED:")
        for line in failures:
            print("  " + line)
        return 1
    print("ok: an output-limit refusal fails at most its own row and never stops the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
