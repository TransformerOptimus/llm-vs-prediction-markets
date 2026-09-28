"""Checks of results.py on a fake fixture.

The fixture is a small runs.db built here (in a temporary folder, never under harness/) with the
real runs.db's own table and view definitions, holding two fake models on Window A:
  fake:0.97  normal pass on every row, probe pass on the rows with news (p = 0.97 everywhere),
             manifest model_set "bridge" (the three-window roster label)
  fake:0.50  normal pass only (p = 0.50 everywhere), model_set "2026" (the Windows B and C label)
plus a quarantine flag file for fake:0.97 written from the memory-probe quarantine rule.

Every number checked here is recomputed straight from the benchmark rows with scoring.trade,
so the check is on the driver's joins, filters, splits and bucketing, not on the scoring code
(test_fees.py and test_gates.py cover that).

Usage: python3 analysis/test_results.py
"""
import csv
import json
import os
import shutil
import sqlite3
import statistics as st
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates
import harness_loader
import results
import scoring

P97, P50 = 0.97, 0.50
TMP = tempfile.mkdtemp(prefix="results_fixture_")


def build_fixture(moments, live_sha):
    """A runs.db with the harness's own DDL and views, two fake normal runs and one probe run on Window A."""
    real = harness_loader.connect_runs()
    ddl = [r[0] for r in real.execute("SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL")]
    path = os.path.join(TMP, "runs.db")
    con = sqlite3.connect(path)
    for sql in ddl:
        con.execute(sql)
    rows_a = sorted((m for m in moments.values() if m["window"] == "A"), key=lambda m: m["row_id"])

    def add_run(run_id, model, mode, p, model_set, only_news):
        man = {"run_id": run_id, "mode": mode, "model": model, "model_set": model_set, "pinned_host_name": "fake host",
               "precision": "none", "template": "default", "db_sha256": live_sha}
        con.execute("INSERT INTO runs (run_id, mode, model, window, template, template_id, db_sha256, started_at, void, "
                    "dry_run, smoke, state, rows_planned, manifest_json) VALUES (?,?,?,?,?,?,?,?,0,0,0,'finished',?,?)",
                    (run_id, mode, model, "A", "default", "harness-prompt-fake", live_sha, "2026-09-12T00:00:00Z",
                     len(rows_a), json.dumps(man)))
        n = 0
        for m in rows_a:
            if only_news and not m["news_available"]:
                continue
            n += 1
            excluded = bool(m["news_dateline_after_capture"])
            con.execute("INSERT INTO calls (run_id, line_no, mode, model, template, row_id, moment_key, window, attempt, "
                        "transport_try, strict, final, status, parsed_probability, timestamp, reasoning_tokens, served_model) "
                        "VALUES (?,?,?,?,?,?,?,?,1,1,0,1,?,?,?,?,?)",
                        (run_id, n, mode, model, "default", m["row_id"], m["moment_key"], "A",
                         "excluded_by_benchmark_flag" if excluded else "ok", None if excluded else p,
                         "2026-09-12T00:00:%02dZ" % (n % 60), 100 + (n % 3), "fake-served"))
        return n

    add_run("fake97_normal_A", "fake:0.97", "normal", P97, "bridge", False)
    add_run("fake97_probe_A", "fake:0.97", "memory_probe", P97, "bridge", True)
    add_run("fake50_normal_A", "fake:0.50", "normal", P50, "2026", False)
    con.commit()
    con.close()
    return path


def write_flags(moments):
    """Memory-probe quarantine flags for fake:0.97 on every Window A row in play, from gates.quarantine_rule."""
    qdir = os.path.join(TMP, "flags")
    os.makedirs(qdir)
    flagged = set()
    with open(os.path.join(qdir, "quarantine_fake97_probe_A.jsonl"), "w") as f:
        for m in moments.values():
            if m["window"] != "A" or m["news_dateline_after_capture"]:
                continue
            q, reason = gates.quarantine_rule(P97, m["q"], float(m["outcome_yes"]))
            if q:
                flagged.add(m["moment_key"])
            f.write(json.dumps({"moment_key": m["moment_key"], "row_id": m["row_id"], "model_name": "fake:0.97",
                                "quarantined": q, "reason": reason}) + "\n")
    return qdir, flagged


def read(out, name):
    with open(os.path.join(out, name + ".csv")) as f:
        return list(csv.DictReader(f))


def expected_rows(moments, cuts, arm, view):
    """Test-split primary rows of the view, straight from the benchmark, with the view's bucket."""
    window, cut_key, flt = results.VIEWS[view]
    out = []
    for m in moments.values():
        if m["window"] != window or m["news_dateline_after_capture"] or not m["primary"] or m["split"] != "test":
            continue
        if arm == "news" and not m["news_available"]:
            continue
        if flt == "live_book" and not m["live_book"]:
            continue
        out.append((m, scoring.bucket_of(m, cuts, cut_key)))
    return out


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main():
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    live_sha = harness_loader.sha256_file(scoring.DB_PATH)
    runs_db = build_fixture(moments, live_sha)
    qdir, flagged = write_flags(moments)
    out1, out2 = os.path.join(TMP, "out1"), os.path.join(TMP, "out2")
    argv = ["--runs-db", runs_db, "--quarantine-dir", qdir, "--n-boot", "10", "--views", "A", "A_live", "--no-figure"]
    results.main(argv + ["--out", out1])
    ctx = results.main(argv + ["--out", out2])

    # 0. rerunnable: byte-identical CSVs
    for name in sorted(os.listdir(out1)):
        if name.endswith(".csv"):
            assert open(os.path.join(out1, name), "rb").read() == open(os.path.join(out2, name), "rb").read(), name

    # 1. Experiment A bucket table: n, Brier and return per bucket match a direct recomputation
    buckets = read(out1, "expA_buckets")
    for view in ("A", "A_live"):
        for arm in ("news", "all"):
            exp = expected_rows(moments, cuts, arm, view)
            for fc, p in (("fake:0.97@default", P97), ("fake:0.50@default", P50)):
                got = {int(r["bucket"]): r for r in buckets if r["view"] == view and r["arm"] == arm and r["model"] == fc}
                assert sum(int(r["n"]) for r in got.values()) == len(exp), (view, arm, fc)
                for b in range(1, 6):
                    ms = [m for m, bb in exp if bb == b]
                    assert int(got[b]["n"]) == len(ms), (view, arm, fc, b)
                    assert close(got[b]["brier_model"], st.fmean((p - m["outcome_yes"]) ** 2 for m in ms)), (view, arm, fc, b)
                    assert close(got[b]["brier_market"], st.fmean((m["q"] - m["outcome_yes"]) ** 2 for m in ms))
                    rets = [scoring.trade(p, m).get("ret") for m in ms]
                    rets = [x for x in rets if x is not None]
                    assert int(got[b]["n_traded"]) == len(rets)
                    if rets:
                        assert close(got[b]["ret"], st.fmean(rets)), (view, arm, fc, b)
                    for name in ("always_yes", "always_no", "cheaper_side", "market_favored", "coin_flip"):
                        assert got[b]["floor_" + name] not in ("", "nan"), (name, b)
                    assert float(got[b]["floor_worst"]) <= min(float(got[b]["floor_" + n]) for n in
                                                                ("always_yes", "always_no", "cheaper_side", "market_favored", "coin_flip")) + 1e-12
                    assert int(got[b]["n_quarantined_kept"]) == (sum(1 for m in ms if m["moment_key"] in flagged) if p == P97 else 0)
    # A_live buckets use the A_live cuts: the thinnest A_live bucket holds rows that sit in several published buckets
    live = [r for r in buckets if r["view"] == "A_live" and r["arm"] == "all" and r["model"] == "fake:0.97@default"]
    pub = [r for r in buckets if r["view"] == "A" and r["arm"] == "all" and r["model"] == "fake:0.97@default"]
    assert [r["median_depth_usd"] for r in live] != [r["median_depth_usd"] for r in pub]

    # 2. gap and trend: the gap is bucket 1 minus bucket 5 of the bucket table; the pooled row stacks both models
    gaps = read(out1, "expA_gap_trend")
    for view in ("A", "A_live"):
        for arm in ("news", "all"):
            g = {r["model"]: r for r in gaps if r["view"] == view and r["arm"] == arm}
            for fc in ("fake:0.97@default", "fake:0.50@default"):
                bt = {int(r["bucket"]): r for r in buckets if r["view"] == view and r["arm"] == arm and r["model"] == fc}
                assert close(g[fc]["gap_ret"], float(bt[1]["ret"]) - float(bt[5]["ret"]), 1e-9), (view, arm, fc)
                assert close(g[fc]["gap_brier_edge"], float(bt[1]["brier_edge"]) - float(bt[5]["brier_edge"]), 1e-9)
            pooled = [r for r in g.values() if r["model"].startswith("POOLED")][0]
            assert pooled["model"] == "POOLED(2 models)"
            assert int(pooled["n"]) == int(g["fake:0.97@default"]["n"]) + int(g["fake:0.50@default"]["n"])
            for r in g.values():
                assert r["within_price_voting_bands"] == r["within_price_voting_bands"]   # present (may be empty)
    verdicts = read(out1, "expA_pooled_verdicts")
    for v in verdicts:
        assert v["n_models"] == "2" and v["roster_complete"] == "False"
        assert v["depth_effect_absent"] in ("True", "False", "")

    # 3. Experiment B: quarantine removed = flagged rows in the model's test rows; gate admitted = rows in pass regions
    gate = json.load(open(os.path.join(out1, "gate_used.json")))
    expb = read(out1, "expB_main")
    for r in expb:
        exp = expected_rows(moments, cuts, r["arm"], r["view"])
        assert int(r["n_test_rows"]) == len(exp)
        want_q = sum(1 for m, _ in exp if m["moment_key"] in flagged) if r["model"] == "fake:0.97@default" else 0
        assert int(r["n_quarantined_removed"]) == want_q, (r["view"], r["arm"], r["model"])
        kept = [m for m, _ in exp if not (r["model"] == "fake:0.97@default" and m["moment_key"] in flagged)]
        adm = sum(1 for m in kept if gates.passes(gate, r["model"], "A", scoring.bucket_of(m, cuts)))
        assert int(r["n_gate_admitted"]) == adm and int(r["n_gate_shut"]) == len(kept) - adm
        assert int(r["n_disagreements"]) <= int(r["n_gate_admitted"])
        assert int(r["n_rated"]) == int(r["n_disagreements"]) - int(r["n_dropped_no_path_point"])
    # a constant forecast makes one bin per region; ECE = |p - yes share| on the calibration rows
    for fc, p in (("fake:0.97@default", P97), ("fake:0.50@default", P50)):
        for b in range(1, 6):
            e = gate[fc]["A"][str(b)]
            cal = [m for m in moments.values() if m["window"] == "A" and m["primary"] and m["split"] == "calibration"
                   and not m["news_dateline_after_capture"] and scoring.bucket_of(m, cuts) == b]
            assert e["n"] == len(cal)
            assert close(e["ece"], abs(p - st.fmean(m["outcome_yes"] for m in cal)), 1e-9), (fc, b)
    # convergence on a direct call: rated + dropped = disagreements, and the shuffle keeps the row count
    rows = results.view_rows(results.load_normal_rows(ctx.rcon, ctx.meta, con, moments, cuts, {}, "all", True)["rows"],
                             "A", moments, cuts)
    sub = [r for r in rows if r["forecaster"] == "fake:0.50@default"][:300]
    real, shuf = results.convergence(ctx, sub, False), results.convergence(ctx, sub, True)
    assert real["n_rated"] + real["n_dropped_no_path_point"] == real["n_disagreements"]
    assert shuf["n_disagreements"] == real["n_disagreements"]      # a permutation of a constant p changes nothing
    assert shuf["n_drifted_toward_model"] == real["n_drifted_toward_model"]

    # 4. Experiment C: errors of two constant forecasts are both linear in the outcome, so r = 1 exactly
    expc = read(out1, "expC_overall")
    assert len(expc) == 4
    for r in expc:
        assert close(r["mean_r"], 1.0, 1e-9) and r["n_pairs"] == "1"
        assert int(r["n_common_rows"]) == len(expected_rows(moments, cuts, r["arm"], r["view"]))
    sets = read(out1, "expC_sets")
    assert {r["pair_kind"] for r in sets} == {"2026-bridge"}
    byb = read(out1, "expC_by_bucket")
    assert all(close(r["mean_r"], 1.0, 1e-9) for r in byb) and len(byb) == 20

    # 5. reporting line: rows in play, skips, failures, thinking tokens, host, precision, set
    rep = {r["run_id"]: r for r in read(out1, "reporting_line")}
    assert set(rep) == {"fake97_normal_A", "fake50_normal_A"}
    in_play = sum(1 for m in moments.values() if m["window"] == "A" and not m["news_dateline_after_capture"])
    for rid, r in rep.items():
        assert int(r["rows_in_play"]) == in_play and int(r["rows_skipped_before_call"]) == 2997 - in_play
        assert int(r["rows_failed"]) == 0 and r["host"] == "fake host" and r["precision"] == "none"
        assert r["thinking_tokens_median"] == "101" and r["db_sha256_matches_live"] == "True"
    assert rep["fake97_normal_A"]["model_set"] == "bridge" and rep["fake50_normal_A"]["model_set"] == "2026"
    assert int(rep["fake97_normal_A"]["rows_quarantined"]) == len(flagged)
    assert int(rep["fake50_normal_A"]["rows_quarantined"]) == 0

    # 6. secondary tables: the full set is every non-ladder test row in play; ladders are the rest
    full = {(r["view"], r["arm"], r["model"]): r for r in read(out1, "secondary_full_set")}
    lad = {(r["view"], r["arm"], r["model"]): r for r in read(out1, "secondary_ladders")}
    for arm in ("news", "all"):
        want_full = sum(1 for m in moments.values() if m["window"] == "A" and not m["news_dateline_after_capture"]
                        and m["split"] == "test" and not m["is_ladder"] and (arm == "all" or m["news_available"]))
        want_lad = sum(1 for m in moments.values() if m["window"] == "A" and not m["news_dateline_after_capture"]
                       and m["split"] == "test" and m["is_ladder"] and (arm == "all" or m["news_available"]))
        assert int(full[("A", arm, "fake:0.97@default")]["n"]) == want_full
        assert int(lad[("A", arm, "fake:0.97@default")]["n"]) == want_lad
        assert int(full[("A", arm, "fake:0.97@default")]["n"]) > int(full[("A", arm, "fake:0.97@default")]["n_outside_price_range"])

    # 7. README lists the status and every resolution
    readme = open(os.path.join(out1, "README.md")).read()
    assert "## Status" in readme and all(t in readme for t, _ in results.RESOLUTIONS)
    shutil.rmtree(TMP)
    print("results checks: all passed")


if __name__ == "__main__":
    main()
