"""Plan runner: launch a list of runs lane by lane, restart-safe, with backups.

Write your own plan file (harness/plan.example.json is a free template using the fake model)
and pass it with --plan:

  python3 harness/run_plan.py --plan my_plan.json --backup-dir /Volumes/backup/harness-runs \\
          --i-approve-paid-calls
  python3 harness/run_plan.py --plan my_plan.json --dry-run      # plan only: every run gets --dry-run

The plan file lists entries in order. Each entry: model (LiteLLM id), mode, windows
(one run per window, run id "<entry id>_<window>"), workers, budget_usd, optional
row_list, ignore_eligibility, host_override, news_only. An entry's lane is the pinned
host from models.json (direct OpenAI is its own lane); an entry may name its lane
outright. At most `max_per_lane` runs go at once per lane (default 1; `max_per_lane_by_lane` in the plan
overrides it per lane) and every lane
goes at once; a lane starts its next run when the previous one finishes.

Each run is a separate run.py process (under caffeinate on macOS so the machine
stays awake). --i-approve-paid-calls is passed on only when the runner itself got it.
State lives in harness/runs/plan_state.json: per run pending / running / done /
failed / paused, with exit codes and backup paths. On restart, unfinished runs are
relaunched with --resume and done runs are never started again. A run that exits with
code 75 (paused on network) is relaunched with --resume once the runner's own probe sees
the network back. A run that fails (permanent error, served-checkpoint mismatch, any
non-zero exit) stops its lane, also across restarts, until --retry-failed relaunches it
with --resume; the other lanes continue. Every finished run's folder is
copied to --backup-dir/<run id> and the copy recorded.

A status line per lane is printed every minute: rows done / planned, cost so far,
429 count, state.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from adapters.base import MODELS_FILE, load_table  # noqa: E402
from gate import EXIT_PAUSED_BY_OWNER, EXIT_PAUSED_ON_NETWORK  # noqa: E402

RUNS_DIR = os.path.join(HERE, "runs")
DEFAULT_PLAN = os.path.join(HERE, "plan.example.json")   # the free fake-model template
DEFAULT_STATE = os.path.join(RUNS_DIR, "plan_state.json")
PROBE_URLS = ["https://openrouter.ai/api/v1", "https://api.openai.com/v1"]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def network_up() -> bool:
    for url in PROBE_URLS:
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            continue
    return False


def lane_of(entry: Dict, models: Dict) -> str:
    if entry.get("lane"):
        return entry["lane"]
    m = models.get(entry["model"]) or {}
    host = entry.get("host_override") or m.get("host")
    if host in (None, "direct"):
        return entry["model"].split("/", 1)[0]          # openai, anthropic, ...
    return host


def expand(plan: Dict, models: Dict, dry_run: bool) -> List[Dict]:
    """One job per (entry, window)."""
    jobs = []
    for e in plan["entries"]:
        for w in e.get("windows") or [None]:
            rid = "%s%s%s" % (e["id"], ("_" + w) if w else "", "_dryrun" if dry_run else "")
            jobs.append({"run_id": rid, "entry_id": e["id"], "lane": lane_of(e, models), "window": w, "entry": e})
    return jobs


def command_for(job: Dict, args) -> List[str]:
    e = job["entry"]
    spec = e.get("model_spec") or ("litellm:" + e["model"])      # model_spec: a fake spec, for tests only
    cmd = [sys.executable, os.path.join(HERE, "run.py"), "--mode", e["mode"], "--model", spec,
           "--run-id", job["run_id"], "--workers", str(e.get("workers", 4))]
    if job["window"]:
        cmd += ["--window", job["window"]]
    if e.get("budget_usd") is not None:
        cmd += ["--budget-usd", str(e["budget_usd"])]
    if e.get("news_only") or e["mode"] == "memory_probe":
        cmd.append("--news-only")
    if e.get("row_list"):
        cmd += ["--row-list", os.path.join(os.path.dirname(HERE), e["row_list"])]
    if e.get("ignore_eligibility"):
        cmd.append("--ignore-eligibility")
    if e.get("host_override"):
        cmd += ["--host-override", e["host_override"]]
    for extra in e.get("extra_flags") or []:
        cmd.append(extra)
    if args.i_approve_paid_calls and not args.dry_run:
        cmd.append("--i-approve-paid-calls")
    if args.allow_dirty:
        cmd.append("--allow-dirty")
    if args.dry_run:
        cmd.append("--dry-run")
    if os.path.exists(os.path.join(RUNS_DIR, job["run_id"])):
        cmd.append("--resume")
    if platform.system() == "Darwin" and shutil.which("caffeinate") and not args.no_caffeinate:
        cmd = ["caffeinate", "-i"] + cmd
    return cmd


class State:
    def __init__(self, path: str):
        self.path = path
        self.data = {"jobs": {}, "started_at": now_iso(), "updated_at": None}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def job(self, run_id: str) -> Dict:
        return self.data["jobs"].setdefault(run_id, {"state": "pending", "launches": [], "backup": None})

    def save(self):
        self.data["updated_at"] = now_iso()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)


def progress(run_id: str) -> Dict:
    """Rows done / planned, cost, 429s from the run folder (the manifest is written at start
    and end; rows and cost come from the records file)."""
    d = os.path.join(RUNS_DIR, run_id)
    out = {"done": 0, "planned": None, "cost": 0.0, "n429": 0, "ok": 0, "failed": 0, "state": None}
    try:
        with open(os.path.join(d, "run.json")) as f:
            m = json.load(f)
        out["planned"] = m.get("rows_planned")
        out["state"] = m.get("state")
        finals = {}
        with open(os.path.join(d, "records.jsonl")) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                out["cost"] += r.get("cost_usd") or 0.0
                if r.get("rate_limit"):
                    out["n429"] += 1
                if r.get("final"):
                    finals[r["row_id"]] = r["status"]
        out["done"] = len(finals)
        out["ok"] = sum(1 for s in finals.values() if s in ("ok", "ok_after_retry"))
        out["failed"] = sum(1 for s in finals.values() if s == "failed")
    except Exception:
        pass
    return out


def backup(run_id: str, backup_dir: str) -> Optional[str]:
    src = os.path.join(RUNS_DIR, run_id)
    dst = os.path.join(backup_dir, run_id)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--state", default=None, help="state file (default harness/runs/plan_state.json; "
                                                  "plan_state_dryrun.json with --dry-run)")
    ap.add_argument("--backup-dir", default=None, help="copy every finished run folder here (a second disk)")
    ap.add_argument("--max-per-lane", type=int, default=None, help="runs at once per lane (default from the plan, else 1)")
    ap.add_argument("--i-approve-paid-calls", action="store_true", help="passed on to every run.py")
    ap.add_argument("--allow-dirty", action="store_true", help="passed on to every run.py")
    ap.add_argument("--dry-run", action="store_true", help="every run gets --dry-run (plans, no calls); run ids get _dryrun")
    ap.add_argument("--status-every", type=float, default=60.0, help="seconds between status lines")
    ap.add_argument("--no-caffeinate", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated entry ids to run (others are left pending)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="relaunch failed runs with --resume and reopen their lanes (default: a lane with a failed "
                         "run stays stopped across restarts until this is passed)")
    args = ap.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    models = load_table(MODELS_FILE)
    for e in plan["entries"]:
        if e["model"] not in models and not e.get("model_spec"):
            raise SystemExit("plan entry %s: model %s is not in models.json" % (e["id"], e["model"]))
    max_per_lane = args.max_per_lane or plan.get("max_per_lane", 1)
    per_lane_cap = plan.get("max_per_lane_by_lane") or {}      # e.g. {"siliconflow": 2}
    state = State(args.state or (DEFAULT_STATE.replace(".json", "_dryrun.json") if args.dry_run else DEFAULT_STATE))
    jobs = expand(plan, models, args.dry_run)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["entry_id"] in keep]
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    for j in jobs:
        js = state.job(j["run_id"])
        if js["state"] == "paused_by_owner":
            js["state"] = "pending"
            js["note"] = "paused by the owner earlier; relaunched with --resume"
    # a run that was "running" when the runner last died is relaunched with --resume;
    # a failed run keeps its lane stopped unless --retry-failed reopens it
    stopped_lanes: Dict[str, str] = {}
    for j in jobs:
        js = state.job(j["run_id"])
        if js["state"] == "running":
            js["state"] = "pending"
            js["note"] = "runner restarted; relaunched with --resume"
        elif js["state"] == "failed":
            if args.retry_failed:
                js["state"] = "pending"
                js["note"] = "failed earlier; relaunched with --resume by --retry-failed"
            else:
                stopped_lanes[j["lane"]] = "%s failed earlier (exit %s); pass --retry-failed to reopen the lane" % (
                    j["run_id"], (js["launches"][-1].get("exit_code") if js["launches"] else "?"))
    state.save()

    lanes: Dict[str, List[Dict]] = {}
    for j in jobs:
        lanes.setdefault(j["lane"], []).append(j)
    procs: Dict[str, subprocess.Popen] = {}
    logs_dir = os.path.join(RUNS_DIR, "plan_logs")
    os.makedirs(logs_dir, exist_ok=True)
    print("plan %s: %d runs in %d lanes (%s); max %d per lane%s" % (
        os.path.basename(args.plan), len(jobs), len(lanes), ", ".join(sorted(lanes)), max_per_lane,
        "; DRY RUN" if args.dry_run else ""))

    def launch(j: Dict):
        cmd = command_for(j, args)
        log = open(os.path.join(logs_dir, j["run_id"] + ".log"), "a")
        log.write("\n=== %s launch: %s\n" % (now_iso(), " ".join(cmd)))
        log.flush()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
        procs[j["run_id"]] = p
        js = state.job(j["run_id"])
        js["state"] = "running"
        js["launches"].append({"at": now_iso(), "pid": p.pid, "resume": "--resume" in cmd, "cmd": cmd})
        state.save()
        print("%s lane %-12s START %s (pid %d%s)" % (now_iso(), j["lane"], j["run_id"], p.pid,
                                                     ", resume" if "--resume" in cmd else ""))

    def finish(j: Dict, code: int):
        js = state.job(j["run_id"])
        js["launches"][-1]["exit_code"] = code
        js["launches"][-1]["ended_at"] = now_iso()
        if code == 0:
            js["state"] = "done"
            if args.backup_dir:
                try:
                    js["backup"] = {"path": backup(j["run_id"], args.backup_dir), "at": now_iso()}
                except Exception as e:
                    js["backup"] = {"error": str(e), "at": now_iso()}
            print("%s lane %-12s DONE  %s%s" % (now_iso(), j["lane"], j["run_id"],
                                                 (" -> backed up to %s" % js["backup"]["path"]) if js.get("backup") and js["backup"].get("path") else ""))
        elif code == EXIT_PAUSED_ON_NETWORK:
            js["state"] = "paused"
            js["paused_at"] = now_iso()
            print("%s lane %-12s PAUSED %s (network); will relaunch with --resume when the network is back"
                  % (now_iso(), j["lane"], j["run_id"]))
        elif code == EXIT_PAUSED_BY_OWNER:
            js["state"] = "paused_by_owner"
            js["paused_at"] = now_iso()
            stopped_lanes[j["lane"]] = "%s paused by the operator; start the runner again to resume it" % j["run_id"]
            print("%s lane %-12s PAUSED BY OWNER %s; lane stopped until the runner is started again"
                  % (now_iso(), j["lane"], j["run_id"]))
        else:
            js["state"] = "failed"
            stopped_lanes[j["lane"]] = "%s exited %d (see runs/plan_logs/%s.log)" % (j["run_id"], code, j["run_id"])
            print("%s lane %-12s FAILED %s exit %d; lane stopped, other lanes continue"
                  % (now_iso(), j["lane"], j["run_id"], code))
        state.save()

    last_status = 0.0
    last_probe, net_ok = 0.0, True
    while True:
        # reap
        for j in jobs:
            p = procs.get(j["run_id"])
            if p is not None and p.poll() is not None:
                del procs[j["run_id"]]
                finish(j, p.returncode)
        # network probe, for paused runs
        if any(state.job(j["run_id"])["state"] == "paused" for j in jobs) and time.time() - last_probe >= 30:
            last_probe = time.time()
            net_ok = network_up()
            if net_ok:
                for j in jobs:
                    if state.job(j["run_id"])["state"] == "paused":
                        state.job(j["run_id"])["state"] = "pending"
                state.save()
        # launch
        for lane, ljobs in lanes.items():
            if lane in stopped_lanes:
                continue
            running = sum(1 for j in ljobs if j["run_id"] in procs)
            for j in ljobs:
                if running >= per_lane_cap.get(lane, max_per_lane):
                    break
                if state.job(j["run_id"])["state"] == "pending":
                    launch(j)
                    running += 1
        # done?
        pending = [j for j in jobs if state.job(j["run_id"])["state"] in ("pending", "running", "paused")
                   and j["lane"] not in stopped_lanes]
        if not procs and not pending:
            break
        if time.time() - last_status >= args.status_every:
            last_status = time.time()
            for lane, ljobs in sorted(lanes.items()):
                cur = [j for j in ljobs if j["run_id"] in procs]
                done_n = sum(1 for j in ljobs if state.job(j["run_id"])["state"] == "done")
                if lane in stopped_lanes:
                    line = "STOPPED: %s" % stopped_lanes[lane]
                elif cur:
                    parts = []
                    for j in cur:
                        pr = progress(j["run_id"])
                        parts.append("%s rows %d/%s ok %d failed %d cost $%.2f 429s %d%s" % (
                            j["run_id"], pr["done"], pr["planned"], pr["ok"], pr["failed"], pr["cost"], pr["n429"],
                            " HOST SATURATED (amber: 429s outnumber successful calls for 3 minutes running; consider the backup host)"
                            if pr["state"] == "host_saturated" else ""))
                    line = "running " + "; ".join(parts)
                else:
                    paused = [j["run_id"] for j in ljobs if state.job(j["run_id"])["state"] == "paused"]
                    line = ("paused on network: %s" % ", ".join(paused)) if paused else "idle"
                print("%s lane %-12s %d/%d done | %s" % (now_iso(), lane, done_n, len(ljobs), line))
        time.sleep(2.0)

    print("plan finished: %d done, %d failed, %d pending/paused; state in %s" % (
        sum(1 for j in jobs if state.job(j["run_id"])["state"] == "done"),
        sum(1 for j in jobs if state.job(j["run_id"])["state"] == "failed"),
        sum(1 for j in jobs if state.job(j["run_id"])["state"] in ("pending", "paused")), state.path))
    if stopped_lanes:
        for lane, why in stopped_lanes.items():
            print("  lane %s stopped: %s" % (lane, why))
        sys.exit(1)


if __name__ == "__main__":
    main()
