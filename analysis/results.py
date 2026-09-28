"""The results driver: runs.db and benchmark.db in, the paper's tables out. Read-only on both databases. Every output goes to
analysis/out/results/ and is rewritten on every run.

What it produces (each table as a CSV with a row count behind every number, and a Markdown
twin), per window view and per arm, never pooled across windows or venues, on the test split
of the primary analysis set:

  views  A (Window A as published), A_live (Window A restricted to live-book rows, its own
         cuts), B, C
  arms   news (rows with news, news shown) and all (every row, no-news records included)

  1. Experiment A: per model and depth bucket, Brier of model and market, edge,
     mean return with fees, the five no-skill floors on the same rows; the thin-minus-deep
     gap (bucket 1 minus bucket 5) with its event bootstrap interval; the trend test;
     the within-price-band (test 4) and within-horizon (test 5) versions; the direction-of-
     disagreement split; the pooled gap across models and whether it shows no depth effect.
  2. Experiment B: memory-probe quarantine removed, calibration gate applied, disagreements of 0.10
     or more by direction, and the convergence rate against a shuffled baseline.
  3. Experiment C: pairwise error correlation per window, overall and per bucket, on
     the rows common to every model in the window; pairs labelled by roster set.
  4. The per-model reporting line.
  5. The central figure: model return by depth bucket, one panel per window view.
  Plus two secondary tables: the full in-play set and the ladder side set.

Inputs: the forecasts view (normal pass) and probe_forecasts view (probe pass) joined by
moment_key with the benchmark-hash check (harness_loader); quarantine flags from the
quarantine_flags table when it holds rows, else from the flag files analysis/out/quarantine_*.jsonl
written by gates.py; the calibration gate rebuilt by gates.build_gate on the calibration rows;
the split label and depth cuts from scoring; price paths through price_paths.

No scoring logic lives here: every number comes from scoring.py, no_skill.py, correlated_errors.py,
price_paths.py and gates.py. Where a rule left a choice of implementation, the choice is listed
in RESOLUTIONS below and printed into out/results/README.md.

Usage: python3 analysis/results.py [--n-boot 2000] [--runs-db PATH] [--out DIR] [--arms news all]
       [--views A A_live B C] [--no-figure]
"""
import argparse
import glob
import json
import math
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates
import harness_loader
import correlated_errors
import no_skill
import price_paths
import scoring
from scoring import N_BOOT, SEED

OUT_DIR = os.path.join(scoring.OUT_DIR, "results")

ROSTER_FILE = os.path.join(scoring.REPO, "harness", "models.json")   # the model roster, read for completeness only

VIEWS = {  # view name -> (window, cut key used for buckets, extra row filter)
    "A": ("A", None, None),
    "A_live": ("A", "A_live", "live_book"),
    "B": ("B", None, None),
    "C": ("C", None, None),
}
ARMS = list(scoring.ARMS)
DISAGREE_MIN = 0.10      # a disagreement is |p - q| of at least 0.10
DRIFT_MIN = 0.02         # drifted toward the model = closer to p than q was, by at least 0.02
CONVERGENCE_HOURS = 72.0  # convergence window
ABSENT_MAX_GAP = 0.05    # no depth effect if the pooled gap's interval includes zero and |point| < 5 points
BAND_MIN_ROWS = 30       # depth-effect tests 4 and 5

RESOLUTIONS = [
    ("Pooled gap across models and the no-effect reading",
     "Every model's test rows in the window are stacked, one observation per model-row, and bucket 1 minus "
     "bucket 5 is taken on the stack with events resampled jointly (a resampled event carries every model's rows "
     "on it). Stacking rows, rather than averaging per-model gaps, keeps the "
     "event bootstrap exact."),
    ("Half of the roster models",
     "The roster counted is the set of models with a finished, non-void normal run in that window and arm at the "
     "time of the run; the fraction is printed with its denominator and the table says the roster is incomplete "
     "until every planned model has run."),
    ("Within-horizon check (test 5)",
     "Computed with the same code as test 4 with the horizon band in place of the price band (scoring.boot_within_band "
     "with band_key \"horizon_band\"); the 30-row rule, equal weights and the "
     "joint event resample are identical."),
    ("Direction of a test",
     "Gap = bucket 1 minus bucket 5; the tested pattern is 'bucket 1 lower', so a model counts when the gap is "
     "negative with an interval excluding zero. The trend is confirmed when the return slope on log depth is "
     "positive with an interval excluding zero. No depth effect: the pooled gap's interval includes zero and |gap| < 0.05."),
    ("Model failure against the floor",
     "A bucket counts as a model failure when the upper end of the model's return interval is below the worst of "
     "the five floors' point values. The coin-flip floor is a mean over 200 draws and carries no bootstrap "
     "interval, so the comparison uses the floor's point value."),
    ("Quarantine source",
     "moment_outcomes' quarantine_flags table is used when it holds rows; while it is empty the flag files "
     "analysis/out/quarantine_<probe run>.jsonl written by gates.py are read instead, and the README says which."),
    ("Gate on the live-book view of Window A",
     "The gate's regions are (window, depth bucket) on the window's published cuts. The A_live view keeps that "
     "gate and looks each row up by its published-A bucket, because no gate is built on the live-book cuts."),
    ("One gate for both arms",
     "The gate is built once per forecaster, window and bucket on every calibration row of the primary set (news "
     "and no-news rows together, as gates.py computes it) and the same verdicts are applied in both arms; a gate "
     "built on the news rows alone would fall under the 30-row floor in most regions."),
    ("Rows entering Experiment B",
     "Test-split rows of the primary analysis set, per model, with that model's quarantined rows removed, then "
     "only rows in regions whose gate verdict is pass for that forecaster. Both removed counts are printed."),
    ("Convergence and its shuffled baseline",
     "A disagreement drifted toward the model when |last - p| <= |q - p| - 0.02, with last = the last price-path "
     "point inside the window (72 hours or the close, whichever is earlier; price_paths.close_time). The baseline "
     "takes the same admitted rows, sorts them by moment_key, permutes their p values with random.Random(20260902), "
     "and reruns the whole test (the 0.10 disagreement threshold included) on the permuted p."),
    ("Common rows for correlated errors",
     "Overall and per-bucket correlations use the rows every model present in the window forecast (the "
     "intersection), so each pair is measured on the same rows; pairs are labelled bridge-bridge, 2026-2026 or "
     "cross from the run manifests' model_set (bridge = the models run on all three windows, 2026 = the models "
     "run on Windows B and C only)."),
    ("Rows in play per model",
     "The rows the harness called plus the rows it called and failed, i.e. the planned rows minus those skipped "
     "before any call (benchmark flag, leak check, release date). Thinking tokens = the median reasoning_tokens on "
     "the run's ok final records; host and precision come from the run manifest."),
    ("Secondary tables",
     "The full-set table is every non-ladder row in play on the test split without the price or end-date filters; "
     "ladders are a separate side table. Both are summaries per model, not by bucket."),
]


# ----------------------------------------------------------------------------
# small helpers (formatting and file writing only)
# ----------------------------------------------------------------------------

def f3(x) -> str:
    if x is None:
        return "-"
    try:
        return "nan" if x != x else "%.3f" % x
    except TypeError:
        return str(x)


def pc(x) -> str:
    return "-" if x is None or x != x else "%+.1f%%" % (100 * x)


def ci(v, lo, hi) -> str:
    return "%s [%s, %s]" % (pc(v), pc(lo), pc(hi))


def ci3(v, lo, hi) -> str:
    return "%s [%s, %s]" % (f3(v), f3(lo), f3(hi))


def write_table(name: str, rows: List[dict], md_title: str, md_note: str = "") -> None:
    """One CSV (all columns) and one Markdown table (same columns) under OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, name + ".csv")
    md_path = os.path.join(OUT_DIR, name + ".md")
    if not rows:
        with open(md_path, "w") as f:
            f.write("# %s\n\nNo rows yet.\n" % md_title)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return
    cols = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    scoring.write_csv(csv_path, rows, cols)
    lines = ["# %s" % md_title, ""]
    if md_note:
        lines += [md_note, ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append("nan" if v != v else ("%.4f" % v))
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def short(model: str) -> str:
    return model.split("@")[0].split("/")[-1]


def planned_models(window: str) -> List[str]:
    """The scored models the roster says should run on this window (harness/models.json). The
    probe-validation model and the fake fixture are not scored, so they are left out. Used only to say
    whether a table covers the whole roster; it changes no number."""
    try:
        with open(ROSTER_FILE) as f:
            roster = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for model, spec in roster.items():
        if not isinstance(spec, dict) or spec.get("set") not in ("bridge", "2026"):
            continue
        if window in (spec.get("windows") or []):
            out.append(model)
    return sorted(out)


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------

def load_quarantine_flags(con, moments, flag_dir: str = scoring.OUT_DIR) -> Dict:
    """(moment_key, model) -> 1 from the table, else from the flag files. Returns (dict, source)."""
    table = scoring.load_quarantine(con, moments)
    table.pop(("_unresolved", ""), None)
    if table:
        return {k: v for k, v in table.items() if v}, "quarantine_flags table"
    out = {}
    files = sorted(glob.glob(os.path.join(flag_dir, "quarantine_*.jsonl")))
    for path in files:
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("quarantined"):
                    out[(d["moment_key"], d["model_name"])] = 1
    return out, "flag files (%d): %s" % (len(files), ", ".join(os.path.basename(p) for p in files))


def load_normal_rows(rcon, meta, con, moments, cuts, quarantine, arm: str, include_ladders: bool):
    recs = harness_loader.read_pass(rcon, "normal")
    r = harness_loader.score_records(recs, meta, moments, cuts, quarantine, arm, apply_quarantine=False,
                                     live_db_sha=harness_loader.sha256_file(scoring.DB_PATH),
                                     include_ladders=include_ladders)
    for row in r["rows"]:
        row["quarantined"] = 1 if quarantine.get((row["moment_key"], row["model_name"])) else 0
    return r


def view_rows(rows: List[dict], view: str, moments, cuts, split: Optional[str] = "test",
              primary_only: bool = True, ladders: Optional[bool] = False) -> List[dict]:
    """The rows of one window view: window filter, live-book filter and A_live rebucketing, split, primary."""
    window, cut_key, flt = VIEWS[view]
    out = []
    for r in rows:
        if r["window"] != window:
            continue
        if flt == "live_book" and not r["live_book"]:
            continue
        if split and r["split"] != split:
            continue
        if primary_only and not r["primary"]:
            continue
        if ladders is not None and bool(r["is_ladder"]) != ladders:
            continue
        if cut_key:
            r = dict(r)
            r["bucket"] = scoring.bucket_of(moments[r["row_id"]], cuts, cut_key)
            r["bucket_published"] = scoring.bucket_of(moments[r["row_id"]], cuts)
        else:
            r = dict(r)
            r["bucket_published"] = r["bucket"]
        out.append(r)
    return out


# ----------------------------------------------------------------------------
# Experiment A
# ----------------------------------------------------------------------------

class Ctx:
    def __init__(self, args):
        self.args = args
        self.n_boot = args.n_boot
        self.con = scoring.connect()
        self.moments = scoring.load_moments(self.con)
        self.cuts = scoring.load_or_compute_cuts(self.moments)
        self.rcon = harness_loader.connect_runs(args.runs_db)
        self.meta = harness_loader.read_manifests(self.rcon)
        self.quarantine, self.quarantine_source = load_quarantine_flags(self.con, self.moments, args.quarantine_dir)
        self._floor_cache = {}
        self._path_cache = {}
        self.status = []       # (table, state, note)
        self.model_set = {}    # forecaster -> roster label: bridge (all three windows) / 2026 (Windows B and C) / validation

    def floors(self, row_ids: List[int], cut_key: Optional[str]) -> Dict[int, dict]:
        key = (tuple(sorted(row_ids)), cut_key)
        if key not in self._floor_cache:
            ns = no_skill.no_skill_rows(self.moments, self.cuts, row_ids=list(row_ids), cut_key=cut_key)
            self._floor_cache[key] = no_skill.floor_by_bucket(ns)
        return self._floor_cache[key]

    def path(self, row_id: int) -> List[dict]:
        if row_id not in self._path_cache:
            self._path_cache[row_id] = price_paths.load_price_path(self.con, self.moments[row_id])
        return self._path_cache[row_id]


def bucket_table(ctx: Ctx, rows: List[dict], view: str, arm: str, forecaster: str, cut_key) -> List[dict]:
    """Per bucket: accuracy, return with fees, horizon, and the five floors on the same rows."""
    out = []
    by_bucket = scoring.summarize_by(rows, "bucket", n_boot=ctx.n_boot, order=[1, 2, 3, 4, 5])
    floors = ctx.floors([r["row_id"] for r in rows], cut_key)
    for b in [1, 2, 3, 4, 5]:
        d = by_bucket.get(b)
        if not d:
            continue
        brs = [r for r in rows if r["bucket"] == b]
        hz = [r["horizon_days"] for r in brs if r["horizon_days"] is not None]
        fl = floors.get(b, {})
        floor = fl.get("floor", {})
        row = {"view": view, "arm": arm, "model": forecaster, "bucket": b,
               "n": d["n"], "n_events": d["n_events"], "n_traded": d["n_traded"], "n_no_trade": d["n_no_trade"],
               "n_one_sided": d["n_one_sided"], "n_degenerate": d["n_degenerate"],
               "n_quarantined_kept": sum(r["quarantined"] for r in brs),
               "n_in_play_flag": sum(r["in_play"] for r in brs),
               "median_depth_usd": round(d["median_depth_usd"], 2),
               "median_horizon_days": round(st.median(hz), 2) if hz else None,
               "brier_model": d["brier_model"], "brier_market": d["brier_market"],
               "brier_edge": d["brier_edge"], "brier_edge_lo": d["be_lo"], "brier_edge_hi": d["be_hi"],
               "market_more_accurate_clear": (d["be_hi"] < 0) if d["be_hi"] == d["be_hi"] else None,
               "ret": d["ret"], "ret_lo": d["ret_lo"], "ret_hi": d["ret_hi"], "ret_nofee": d["ret_nofee"],
               "n_fee_charged": d["n_fee_charged"], "win_rate": d["win_rate"]}
        for name in no_skill.STRATEGIES:
            e = fl.get(name, {})
            row["floor_" + name] = e.get("ret")
            row["floor_" + name + "_n_traded"] = e.get("n_traded")
        row["floor_worst"] = floor.get("ret")
        row["floor_worst_strategy"] = floor.get("strategy")
        row["model_failure"] = (d["ret_hi"] < floor["ret"]) if floor and d["ret_hi"] == d["ret_hi"] else None
        out.append(row)
    return out


def gap_and_trend(ctx: Ctx, rows: List[dict], view: str, arm: str, forecaster: str) -> dict:
    """Depth-effect tests 1 to 5 for one set of rows (one model, or the pooled stack)."""
    nb = ctx.n_boot
    b1 = [r for r in rows if r["bucket"] == 1]
    b5 = [r for r in rows if r["bucket"] == 5]
    g, lo, hi = scoring.boot_diff(b1, b5, "ret", n=nb)
    ge, ge_lo, ge_hi = scoring.boot_diff(b1, b5, "brier_edge", n=nb)
    noflag = [r for r in rows if not r["in_play"]]
    gf, gf_lo, gf_hi = scoring.boot_diff([r for r in noflag if r["bucket"] == 1], [r for r in noflag if r["bucket"] == 5], "ret", n=nb)
    traded = [r for r in rows if r["ret"] is not None]
    s_ret = scoring.boot_slope(traded, "ret", n=nb)
    s_be = scoring.boot_slope(rows, "brier_edge", n=nb)
    out = {"view": view, "arm": arm, "model": forecaster,
           "n": len(rows), "n_events": len({r["event_id"] for r in rows}),
           "n_traded_b1": sum(1 for r in b1 if r["ret"] is not None), "n_traded_b5": sum(1 for r in b5 if r["ret"] is not None),
           "gap_ret": g, "gap_ret_lo": lo, "gap_ret_hi": hi,
           "gap_ret_excludes_zero": (lo > 0 or hi < 0) if lo == lo else None,
           "test1_bucket1_lower_clear": (g < 0 and hi < 0) if lo == lo else None,
           "gap_ret_no_in_play_flag": gf, "gap_ret_no_in_play_flag_lo": gf_lo, "gap_ret_no_in_play_flag_hi": gf_hi,
           "n_in_play_flag_removed": len(rows) - len(noflag),
           "gap_brier_edge": ge, "gap_brier_edge_lo": ge_lo, "gap_brier_edge_hi": ge_hi,
           "slope_ret": s_ret["slope"], "slope_ret_lo": s_ret["lo"], "slope_ret_hi": s_ret["hi"],
           "slope_ret_n": s_ret["n"], "slope_dropped_zero_depth": s_ret["dropped_zero_depth"],
           "trend_confirmed": (s_ret["slope"] > 0 and s_ret["lo"] > 0) if s_ret["lo"] == s_ret["lo"] else None,
           "slope_brier_edge": s_be["slope"], "slope_brier_edge_lo": s_be["lo"], "slope_brier_edge_hi": s_be["hi"],
           "slope_brier_edge_n": s_be["n"]}
    for band_key, tag in (("band", "price"), ("horizon_band", "horizon")):
        wd = scoring.boot_within_band(traded, "ret", "diff", n=nb, min_rows=BAND_MIN_ROWS, band_key=band_key)
        ws = scoring.boot_within_band(traded, "ret", "slope", n=nb, min_rows=BAND_MIN_ROWS, band_key=band_key)
        wb = scoring.boot_within_band(rows, "brier_edge", "slope", n=nb, min_rows=BAND_MIN_ROWS, band_key=band_key)
        out["within_%s_gap_ret" % tag] = wd["value"]
        out["within_%s_gap_ret_lo" % tag] = wd["lo"]
        out["within_%s_gap_ret_hi" % tag] = wd["hi"]
        out["within_%s_voting_bands" % tag] = "|".join(wd["voting_bands"])
        out["within_%s_slope_ret" % tag] = ws["value"]
        out["within_%s_slope_ret_lo" % tag] = ws["lo"]
        out["within_%s_slope_ret_hi" % tag] = ws["hi"]
        out["within_%s_slope_brier_edge" % tag] = wb["value"]
        out["within_%s_slope_brier_edge_lo" % tag] = wb["lo"]
        out["within_%s_slope_brier_edge_hi" % tag] = wb["hi"]
        same = None
        if g == g and wd["value"] == wd["value"]:
            same = (g < 0) == (wd["value"] < 0)
        out["within_%s_agrees_with_plain" % tag] = same
        out["_per_band_%s" % tag] = wd["per_band"]
    return out


def per_band_rows(g: dict, view, arm, forecaster) -> List[dict]:
    out = []
    for tag, band_key in (("price", "band"), ("horizon", "horizon_band")):
        per = g.pop("_per_band_%s" % tag)
        voting = set(g["within_%s_voting_bands" % tag].split("|"))
        for band, v in per.items():
            out.append({"view": view, "arm": arm, "model": forecaster, "band_kind": tag, "band": band,
                        "n_traded_b1": v[1] if v else 0, "n_traded_b5": v[2] if v else 0,
                        "gap_ret": v[0] if v else None, "votes": band in voting})
    return out


def direction_table(ctx: Ctx, rows: List[dict], view, arm, forecaster) -> List[dict]:
    """Returns and Brier edge by direction of disagreement and by price band."""
    out = []
    for d in ("with", "against", "no lean"):
        for band in [None] + scoring.BAND_NAMES:
            sub = [r for r in rows if r["direction"] == d and (band is None or r["band"] == band)]
            if not sub:
                continue
            s = scoring.summarize(sub, n_boot=ctx.n_boot)
            out.append({"view": view, "arm": arm, "model": forecaster, "direction": d, "band": band or "all",
                        "n": s["n"], "n_events": s["n_events"], "n_traded": s["n_traded"],
                        "brier_edge": s["brier_edge"], "brier_edge_lo": s["be_lo"], "brier_edge_hi": s["be_hi"],
                        "ret": s["ret"], "ret_lo": s["ret_lo"], "ret_hi": s["ret_hi"]})
    return out


def experiment_a(ctx: Ctx, rows_by_arm: Dict[str, List[dict]], views: List[str]) -> dict:
    buckets, gaps, bands, dirs, pooled = [], [], [], [], []
    per_model_gap = {}
    for view in views:
        window, cut_key, _ = VIEWS[view]
        for arm in ctx.args.arms:
            rows = view_rows(rows_by_arm[arm], view, ctx.moments, ctx.cuts)
            models = sorted({r["forecaster"] for r in rows})
            if not rows:
                ctx.status.append(("experiment_a %s/%s" % (view, arm), "waiting", "no finished normal run on this window"))
                continue
            for fc in models:
                mrows = [r for r in rows if r["forecaster"] == fc]
                buckets += bucket_table(ctx, mrows, view, arm, fc, cut_key)
                g = gap_and_trend(ctx, mrows, view, arm, fc)
                bands += per_band_rows(g, view, arm, fc)
                gaps.append(g)
                per_model_gap[(view, arm, fc)] = g
                dirs += direction_table(ctx, mrows, view, arm, fc)
            # pooled stack across models (pooled gap, no-effect reading)
            g = gap_and_trend(ctx, rows, view, arm, "POOLED(%d models)" % len(models))
            bands += per_band_rows(g, view, arm, g["model"])
            gaps.append(g)
            planned = planned_models(window)
            present = sorted({r["model_name"] for r in rows})
            missing = [m for m in planned if m not in present]
            n_clear = sum(1 for fc in models if per_model_gap[(view, arm, fc)]["test1_bucket1_lower_clear"])
            includes_zero = (g["gap_ret_lo"] <= 0 <= g["gap_ret_hi"]) if g["gap_ret_lo"] == g["gap_ret_lo"] else None
            absent = (includes_zero and abs(g["gap_ret"]) < ABSENT_MAX_GAP) if includes_zero is not None else None
            pooled.append({"view": view, "arm": arm, "n_models": len(models), "models": "|".join(short(m) for m in models),
                           "n_rows": g["n"], "n_events": g["n_events"],
                           "models_test1_clear": n_clear, "test1_at_least_half": (n_clear * 2 >= len(models)) if models else None,
                           "pooled_gap_ret": g["gap_ret"], "pooled_gap_ret_lo": g["gap_ret_lo"], "pooled_gap_ret_hi": g["gap_ret_hi"],
                           "test2_pooled_excludes_zero": g["gap_ret_excludes_zero"],
                           "depth_effect_absent": absent,
                           "trend_confirmed_pooled": g["trend_confirmed"],
                           "within_price_agrees": g["within_price_agrees_with_plain"],
                           "within_horizon_agrees": g["within_horizon_agrees_with_plain"],
                           "roster_complete": not missing,
                           "models_missing": "|".join(short(m) for m in missing) or "-",
                           "note": ("every planned model for this window has run" if not missing else
                                    "roster incomplete: verdicts are provisional until every planned model has run")})
            ctx.status.append(("experiment_a %s/%s" % (view, arm), "computed on %d model(s)" % len(models),
                               "whole roster" if not missing else "waiting for " + ", ".join(short(m) for m in missing)))
    write_table("expA_buckets", buckets, "Experiment A: accuracy and return per depth bucket, with the no-skill floors",
                "Test split of the primary analysis set. Returns are per dollar staked with fees, over traded rows; "
                "brier_* over all rows. Intervals: 95% event bootstrap. model_failure = return interval entirely below the worst floor.")
    write_table("expA_gap_trend", gaps, "Experiment A: thin-minus-deep gap, trend test, within-band and within-horizon checks",
                "gap = bucket 1 minus bucket 5. POOLED rows stack every model's rows in the view and arm.")
    write_table("expA_bands", bands, "Experiment A: per-band gaps behind tests 4 and 5",
                "A band votes only if both its bucket-1 and bucket-5 traded counts reach 30.")
    write_table("expA_direction", dirs, "Experiment A: returns and Brier edge by direction of disagreement and price band")
    write_table("expA_pooled_verdicts", pooled, "Experiment A: pooled tests and the no-effect reading per view and arm",
                "depth_effect_absent: pooled gap interval includes zero and |gap| < 0.05. roster_complete says whether every model "
                "the roster plans for this window has run; models_missing names the ones still outstanding.")
    return {"buckets": buckets, "gaps": gaps, "pooled": pooled}


# ----------------------------------------------------------------------------
# Experiment B
# ----------------------------------------------------------------------------

def convergence(ctx: Ctx, rows: List[dict], shuffled: bool) -> dict:
    """Convergence over admitted rows. With shuffled=True, p is permuted across the rows first."""
    rows = sorted(rows, key=lambda r: r["moment_key"])
    ps = [r["p"] for r in rows]
    if shuffled:
        random.Random(SEED).shuffle(ps)
    n_dis = n_drift = n_drop = 0
    for r, p in zip(rows, ps):
        if abs(p - r["q"]) < DISAGREE_MIN:
            continue
        n_dis += 1
        m = ctx.moments[r["row_id"]]
        last = price_paths.last_price_in_window(ctx.path(r["row_id"]), m, CONVERGENCE_HOURS)
        if last is None:
            n_drop += 1
            continue
        if abs(last - p) <= abs(r["q"] - p) - DRIFT_MIN:
            n_drift += 1
    rated = n_dis - n_drop
    return {"n_disagreements": n_dis, "n_dropped_no_path_point": n_drop, "n_rated": rated,
            "n_drifted_toward_model": n_drift, "rate": n_drift / rated if rated else None}


def experiment_b(ctx: Ctx, rows_by_arm: Dict[str, List[dict]], views: List[str], gate: dict) -> None:
    main, regions, dirs = [], [], []
    for view in views:
        window = VIEWS[view][0]
        for arm in ctx.args.arms:
            rows = view_rows(rows_by_arm[arm], view, ctx.moments, ctx.cuts)
            for fc in sorted({r["forecaster"] for r in rows}):
                mrows = [r for r in rows if r["forecaster"] == fc]
                kept = [r for r in mrows if not r["quarantined"]]
                n_q = len(mrows) - len(kept)
                admitted = [r for r in kept if gates.passes(gate, fc, window, r["bucket_published"])]
                shut = len(kept) - len(admitted)
                for b in [1, 2, 3, 4, 5]:
                    reg = [r for r in kept if r["bucket_published"] == b]
                    entry = gate.get(fc, {}).get(window, {}).get(str(b), {})
                    regions.append({"view": view, "arm": arm, "model": fc, "region_bucket_published": b,
                                    "verdict": entry.get("verdict", "missing"), "gate_n": entry.get("n"), "gate_ece": entry.get("ece"),
                                    "rows": len(reg), "admitted": len(reg) if entry.get("verdict") == "pass" else 0,
                                    "shut": 0 if entry.get("verdict") == "pass" else len(reg)})
                dis = [r for r in admitted if abs(r["p"] - r["q"]) >= DISAGREE_MIN]
                for d in ("with", "against", "no lean"):
                    sub = [r for r in dis if r["direction"] == d]
                    if not sub:
                        continue
                    s = scoring.summarize(sub, n_boot=ctx.n_boot)
                    dirs.append({"view": view, "arm": arm, "model": fc, "direction": d, "n": s["n"], "n_events": s["n_events"],
                                 "n_traded": s["n_traded"], "brier_edge": s["brier_edge"], "brier_edge_lo": s["be_lo"],
                                 "brier_edge_hi": s["be_hi"], "ret": s["ret"], "ret_lo": s["ret_lo"], "ret_hi": s["ret_hi"],
                                 "share_market_moved_toward_model": None})
                real = convergence(ctx, admitted, shuffled=False)
                base = convergence(ctx, admitted, shuffled=True)
                main.append({"view": view, "arm": arm, "model": fc, "n_test_rows": len(mrows),
                             "n_quarantined_removed": n_q, "n_after_quarantine": len(kept),
                             "n_gate_admitted": len(admitted), "n_gate_shut": shut,
                             "regions_pass": sum(1 for b in range(1, 6) if gates.passes(gate, fc, window, b)),
                             "n_disagreements": real["n_disagreements"],
                             "n_disagree_with": sum(1 for r in dis if r["direction"] == "with"),
                             "n_disagree_against": sum(1 for r in dis if r["direction"] == "against"),
                             "n_dropped_no_path_point": real["n_dropped_no_path_point"], "n_rated": real["n_rated"],
                             "n_drifted_toward_model": real["n_drifted_toward_model"], "convergence_rate": real["rate"],
                             "shuffled_n_disagreements": base["n_disagreements"], "shuffled_n_rated": base["n_rated"],
                             "shuffled_rate": base["rate"],
                             "rate_minus_shuffled": (real["rate"] - base["rate"]) if real["rate"] is not None and base["rate"] is not None else None})
    write_table("expB_main", main, "Experiment B: quarantine, gate, disagreements and convergence per model",
                "Test split of the primary analysis set. Memory-probe quarantine removed first, then only regions whose calibration-gate verdict is pass. "
                "Convergence: share of disagreements (|p - q| >= 0.10) whose last in-window price moved toward p by at least 0.02; "
                "shuffled = the same test with p permuted across the admitted rows (seed 20260902).")
    write_table("expB_regions", regions, "Experiment B: rows admitted and shut per gate region")
    write_table("expB_direction", dirs, "Experiment B: admitted disagreements by direction")
    ctx.status.append(("experiment_b", "computed for every model with a normal run", "gate and quarantine grow with each probe run"))


# ----------------------------------------------------------------------------
# Experiment C
# ----------------------------------------------------------------------------

def experiment_c(ctx: Ctx, rows_by_arm: Dict[str, List[dict]], views: List[str]) -> None:
    overall, per_bucket, pairs_out, sets_out = [], [], [], []
    for view in views:
        for arm in ctx.args.arms:
            rows = view_rows(rows_by_arm[arm], view, ctx.moments, ctx.cuts)
            models = sorted({r["forecaster"] for r in rows})
            if len(models) < 2:
                ctx.status.append(("experiment_c %s/%s" % (view, arm), "waiting", "needs two models; has %d" % len(models)))
                continue
            common = None
            for fc in models:
                ids = {r["row_id"] for r in rows if r["forecaster"] == fc}
                common = ids if common is None else common & ids
            crow = [r for r in rows if r["row_id"] in common]
            res = correlated_errors.pairwise(crow, n_boot=ctx.n_boot)
            overall.append({"view": view, "arm": arm, "n_models": len(models), "n_common_rows": len(common),
                            "n_events": len({r["event_id"] for r in crow}), "n_pairs": res["n_pairs"],
                            "mean_r": res["mean_r"], "r_lo": res.get("r_lo"), "r_hi": res.get("r_hi"),
                            "min_r": res["min_r"], "max_r": res["max_r"],
                            "correlated_errors_confirmed": (res["mean_r"] > 0.5 and res.get("r_lo", float("nan")) > 0.5) if res["n_pairs"] else None,
                            "both_wrong": res["both_wrong"], "both_wrong_expected": res["both_wrong_expected"]})
            for p in res["pairs"]:
                sa, sb = ctx.model_set.get(p["a"], "?"), ctx.model_set.get(p["b"], "?")
                pairs_out.append({"view": view, "arm": arm, "a": p["a"], "b": p["b"], "set_a": sa, "set_b": sb,
                                  "pair_kind": "%s-%s" % tuple(sorted((sa, sb))), "n_common": p["n_common"], "r": p["r"],
                                  "both_wrong": p["both_wrong"], "both_wrong_expected": p["both_wrong_expected"]})
            kinds = defaultdict(list)
            for p in pairs_out:
                if p["view"] == view and p["arm"] == arm:
                    kinds[p["pair_kind"]].append(p["r"])
            for k, rs in sorted(kinds.items()):
                sets_out.append({"view": view, "arm": arm, "pair_kind": k, "n_pairs": len(rs), "n_common_rows": len(common),
                                 "mean_r": st.fmean(rs) if rs else None})
            for b, d in correlated_errors.by_bucket(crow, n_boot=ctx.n_boot).items():
                per_bucket.append({"view": view, "arm": arm, "bucket": b, "n_rows": d["n_rows"], "n_markets": d["n_markets"],
                                   "n_pairs": d["n_pairs"], "mean_r": d["mean_r"], "r_lo": d.get("r_lo"), "r_hi": d.get("r_hi"),
                                   "both_wrong": d["both_wrong"], "both_wrong_expected": d["both_wrong_expected"]})
            sets_here = sorted({ctx.model_set.get(m, "?") for m in models})
            ctx.status.append(("experiment_c %s/%s" % (view, arm), "computed on %d models" % len(models),
                               "sets present: " + ", ".join(sets_here)))
    write_table("expC_overall", overall, "Experiment C: mean pairwise error correlation on common rows",
                "Common rows = rows every model in the view forecast. Confirmed if mean r > 0.5 with the interval above 0.5.")
    write_table("expC_by_bucket", per_bucket, "Experiment C: pairwise error correlation per depth bucket")
    write_table("expC_pairs", pairs_out, "Experiment C: every pair, labelled by roster set")
    write_table("expC_sets", sets_out, "Experiment C: three-window set (bridge) versus Windows B and C set (2026) inside each window, on common rows")


# ----------------------------------------------------------------------------
# Reporting line, secondary tables, figure
# ----------------------------------------------------------------------------

def reporting_line(ctx: Ctx, counts_by_run: Dict[str, dict]) -> None:
    out = []
    for run_id, m in sorted(ctx.meta.items()):
        if m.get("mode") != "normal" or m.get("dry_run") or m.get("smoke") or m.get("void"):
            continue
        man = m.get("manifest") or {}
        c = counts_by_run.get(run_id, {})
        fc = "%s@%s" % (m["model"], m.get("template") or man.get("template") or "default")
        toks = [r[0] for r in ctx.rcon.execute(
            "SELECT reasoning_tokens FROM calls WHERE run_id = ? AND final = 1 AND status IN ('ok','ok_after_retry') "
            "AND reasoning_tokens IS NOT NULL", (run_id,))]
        served = ctx.rcon.execute("SELECT served_model, COUNT(*) FROM calls WHERE run_id = ? AND final = 1 GROUP BY 1 ORDER BY 2 DESC",
                                  (run_id,)).fetchone()
        called = c.get("model_called", 0)
        in_play_q = sum(1 for (mk, model), v in ctx.quarantine.items() if model == m["model"] and v
                        and any(mm["window"] == m["window"] for mm in [ctx.moments[rid] for rid in [] ]))  # placeholder, replaced below
        q_rows = 0
        by_key = {mm["moment_key"]: mm for mm in ctx.moments.values()}
        for (mk, model), v in ctx.quarantine.items():
            if v and model == m["model"] and mk in by_key and by_key[mk]["window"] == m["window"]:
                q_rows += 1
        out.append({"run_id": run_id, "model": m["model"], "forecaster": fc, "window": m["window"],
                    "model_set": man.get("model_set"), "state": m.get("state"),
                    "rows_planned": m.get("rows_planned"),
                    "rows_skipped_before_call": sum(v for k, v in c.items() if k.startswith("skipped_")),
                    "rows_in_play": called, "rows_failed": c.get("failed_rows", 0),
                    "failed_share": c.get("failed_share"), "incomplete_over_5pct": (c.get("failed_share", 0) > 0.05),
                    "rows_scored": c.get("scored", 0), "rows_quarantined": q_rows,
                    "thinking_tokens_median": int(st.median(toks)) if toks else None, "thinking_tokens_n": len(toks),
                    "host": man.get("pinned_host_name") or man.get("pinned_host"),
                    "precision": man.get("precision") or man.get("host_quantization") or "-",
                    "served_model": served[0] if served else None,
                    "release_check_off": bool(man.get("ignore_eligibility")),
                    "db_sha256_matches_live": (m.get("db_sha256") == harness_loader.sha256_file(scoring.DB_PATH))})
    write_table("reporting_line", out, "Per-model reporting line: rows in play, failures, quarantine, thinking, host, precision",
                "rows_in_play = rows the harness called (ok plus failed) after skipping benchmark-flagged, leak-blocked and ineligible rows. "
                "Quarantine source: %s." % ctx.quarantine_source)


def secondary_tables(ctx: Ctx, full_by_arm: Dict[str, List[dict]], views: List[str]) -> None:
    full, ladders = [], []
    for view in views:
        for arm in ctx.args.arms:
            base = view_rows(full_by_arm[arm], view, ctx.moments, ctx.cuts, split="test", primary_only=False, ladders=None)
            for fc in sorted({r["forecaster"] for r in base}):
                for kind, sel in (("full_set_non_ladder", [r for r in base if r["forecaster"] == fc and not r["is_ladder"]]),
                                  ("ladders", [r for r in base if r["forecaster"] == fc and r["is_ladder"]])):
                    if not sel:
                        continue
                    s = scoring.summarize(sel, n_boot=ctx.n_boot)
                    row = {"view": view, "arm": arm, "model": fc, "n": s["n"], "n_events": s["n_events"], "n_traded": s["n_traded"],
                           "n_outside_price_range": sum(1 for r in sel if not r["in_price_range"]),
                           "n_end_date_revised": sum(1 for r in sel if r["end_date_revised"]),
                           "brier_model": s["brier_model"], "brier_market": s["brier_market"],
                           "brier_edge": s["brier_edge"], "brier_edge_lo": s["be_lo"], "brier_edge_hi": s["be_hi"],
                           "ret": s["ret"], "ret_lo": s["ret_lo"], "ret_hi": s["ret_hi"]}
                    (full if kind == "full_set_non_ladder" else ladders).append(row)
    write_table("secondary_full_set", full, "Secondary: every non-ladder row in play on the test split, no price or end-date filter")
    write_table("secondary_ladders", ladders, "Side set: ladder rows on the test split")


def figure(ctx: Ctx, buckets: List[dict], views: List[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        ctx.status.append(("figure", "skipped", "matplotlib not installed"))
        return
    pts = []
    all_models = sorted({b["model"] for b in buckets})
    colors = {fc: plt.get_cmap("tab10")(i % 10) for i, fc in enumerate(all_models)}   # one color per model in every panel
    for arm in ctx.args.arms:
        fig, axes = plt.subplots(1, len(views), figsize=(4.2 * len(views), 3.8), sharey=True)
        if len(views) == 1:
            axes = [axes]
        for ax, view in zip(axes, views):
            sub = [b for b in buckets if b["view"] == view and b["arm"] == arm]
            models = sorted({b["model"] for b in sub})
            for i, fc in enumerate(models):
                bs = sorted([b for b in sub if b["model"] == fc], key=lambda b: b["bucket"])
                xs = [b["bucket"] + (i - (len(models) - 1) / 2) * 0.08 for b in bs]
                ys = [b["ret"] for b in bs]
                err = [[b["ret"] - b["ret_lo"] for b in bs], [b["ret_hi"] - b["ret"] for b in bs]]
                ax.errorbar(xs, ys, yerr=err, marker="o", capsize=2, label=short(fc), color=colors[fc])
                for b in bs:
                    pts.append({"arm": arm, "view": view, "model": fc, "bucket": b["bucket"], "n_traded": b["n_traded"],
                                "ret": b["ret"], "ret_lo": b["ret_lo"], "ret_hi": b["ret_hi"], "floor_worst": b["floor_worst"]})
            if sub:
                fl = sorted({(b["bucket"], b["floor_worst"]) for b in sub if b["floor_worst"] is not None})
                by_b = defaultdict(list)
                for b, v in fl:
                    by_b[b].append(v)
                ax.plot(sorted(by_b), [min(by_b[b]) for b in sorted(by_b)], "k--", label="worst no-skill floor")
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title("Window %s" % view.replace("_live", " (live book)"))
            ax.set_xlabel("depth bucket (1 = thinnest)")
            ax.set_xticks([1, 2, 3, 4, 5])
        axes[0].set_ylabel("mean return per dollar, with fees")
        handles, labels = {}, []
        for ax in axes:                      # one legend over every panel's models
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in handles:
                    handles[l] = h
        axes[-1].legend(list(handles.values()), list(handles), fontsize=7)
        fig.suptitle("Model return minus market (market = 0) by depth, %s arm, test split" % arm)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, "figure_returns_by_depth_%s.png" % arm)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    write_table("figure_points", pts, "Points plotted in the central figure")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def write_readme(ctx: Ctx, gate_paths, views) -> None:
    lines = ["# Results", "",
             "Generated by analysis/results.py. Read-only on runs.db and benchmark.db. Every table is a CSV with a "
             "row count behind every number and a Markdown twin. Views: A (as published), A_live (live-book rule, own cuts), "
             "B, C. Arms: news (rows with news) and all. Test split of the primary analysis set unless the table says otherwise.", "",
             "Bootstrap draws: %d (seed %d, events resampled). Quarantine source: %s." % (ctx.n_boot, SEED, ctx.quarantine_source), "",
             "## Status", "", "| table | state | note |", "|---|---|---|"]
    for t, s, n in ctx.status:
        lines.append("| %s | %s | %s |" % (t, s, n))
    lines += ["", "## Where a rule left a choice, and what this script does", ""]
    for title, text in RESOLUTIONS:
        lines.append("- **%s.** %s" % (title, text))
    lines += ["", "## Files", ""]
    for f in sorted(os.listdir(OUT_DIR)):
        lines.append("- `%s`" % f)
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


class _FigCtx:
    """Just enough context for figure(): the arms and a status list."""
    def __init__(self, args):
        self.args = args
        self.status = []


def main(argv=None):
    global OUT_DIR
    ap = argparse.ArgumentParser(description="results driver")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--runs-db", default=harness_loader.RUNS_DB)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--quarantine-dir", default=scoring.OUT_DIR, help="where the quarantine_*.jsonl flag files are")
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from out/results/expA_buckets.csv")
    args = ap.parse_args(argv)
    OUT_DIR = args.out
    if args.figure_only:
        import csv
        with open(os.path.join(OUT_DIR, "expA_buckets.csv")) as f:
            buckets = []
            for r in csv.DictReader(f):
                for k in ("bucket",):
                    r[k] = int(r[k])
                for k in ("ret", "ret_lo", "ret_hi", "floor_worst"):
                    r[k] = float(r[k]) if r[k] not in ("", "nan") else None
                r["n_traded"] = int(r["n_traded"])
                buckets.append(r)
        figure(_FigCtx(args), buckets, args.views)
        print("redrew the figure in", OUT_DIR)
        return None
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = Ctx(args)
    views = args.views

    rows_by_arm, full_by_arm, counts_by_run = {}, {}, {}
    for arm in args.arms:
        r = load_normal_rows(ctx.rcon, ctx.meta, ctx.con, ctx.moments, ctx.cuts, ctx.quarantine, arm, include_ladders=True)
        full_by_arm[arm] = r["rows"]
        rows_by_arm[arm] = [x for x in r["rows"] if not x["is_ladder"]]
        counts_by_run = r["counts_by_run"] if arm == "all" else counts_by_run or r["counts_by_run"]
    for run_id, m in ctx.meta.items():
        fc = "%s@%s" % (m["model"], m.get("template") or "default")
        ctx.model_set[fc] = (m.get("manifest") or {}).get("model_set", "?")

    # the calibration gate: one gate per forecaster, window and bucket, built on every calibration row of the
    # primary set (the 'all' arm, as gates.py builds it), whichever arms are being reported
    if "all" in full_by_arm:
        gate_rows = full_by_arm["all"]
    else:
        gate_rows = load_normal_rows(ctx.rcon, ctx.meta, ctx.con, ctx.moments, ctx.cuts, ctx.quarantine, "all", include_ladders=True)["rows"]
    gate = gates.build_gate([r for r in gate_rows if not r["is_ladder"]])
    gate_paths = None
    with open(os.path.join(OUT_DIR, "gate_used.json"), "w") as f:
        json.dump(gate, f, indent=1)

    a = experiment_a(ctx, rows_by_arm, views)
    experiment_b(ctx, rows_by_arm, views, gate)
    experiment_c(ctx, rows_by_arm, views)
    reporting_line(ctx, counts_by_run)
    secondary_tables(ctx, full_by_arm, views)
    if not args.no_figure:
        figure(ctx, a["buckets"], views)
    write_readme(ctx, gate_paths, views)
    print("wrote", OUT_DIR)
    for t, s, n in ctx.status:
        print("  %-28s %-40s %s" % (t, s, n))
    return ctx


if __name__ == "__main__":
    main()
