"""Shared loader for the venue-and-window scripts. Read-only on both databases.

Loads every normal-mode final forecast (status ok / ok_after_retry, parsed probability present),
joins it to the benchmark by moment_key, scores it with analysis/scoring.py and keeps the
primary analysis set (is_ladder 0, end_date_revised 0, dateline flag 0, 0.05 <= q <= 0.95).
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import os, sqlite3, sys
import scoring

REPO = _paths.REPO
RUNS_DB = os.path.join(REPO, "harness", "runs", "runs.db")
EXCLUDE = set(_roster.NOT_SCORED)  # the probe-validation model and the fixture
BRIDGE = _roster.BRIDGE            # the three-window set (roster label "bridge"), from harness/models.json
SET2026 = _roster.SET2026          # the Windows B and C set (roster label "2026"), from harness/models.json
MODELS = _roster.SCORED            # every scored model in the roster
N_BOOT = 1000
SEED = scoring.SEED


def short(model):
    return model.split("/")[-1]


def load(mode="normal", primary_only=True, split=None):
    con = scoring.connect()
    moments = scoring.load_moments(con)
    by_key = {m["moment_key"]: m for m in moments.values()}
    cuts = scoring.load_or_compute_cuts(moments)
    r = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    r.row_factory = sqlite3.Row
    rows = []
    q = ("SELECT model, mode, window, row_id, moment_key, parsed_probability, status, output_tokens, reasoning_tokens "
         "FROM forecasts WHERE mode = ? AND parsed_probability IS NOT NULL AND status IN ('ok','ok_after_retry')")
    for rec in r.execute(q, (mode,)):
        if rec["model"] in EXCLUDE:
            continue
        m = by_key.get(rec["moment_key"])
        if m is None or m["row_id"] != rec["row_id"] or m["window"] != rec["window"]:
            continue
        if primary_only and not m["primary"]:
            continue
        if split and m["split"] != split:
            continue
        s = scoring.score_forecast(m, float(rec["parsed_probability"]), rec["model"], cuts,
                                   extra={"model": rec["model"], "short": short(rec["model"]),
                                          "output_tokens": rec["output_tokens"], "reasoning_tokens": rec["reasoning_tokens"],
                                          "topic_tags": m["topic_tags"], "fee_rate": m.get("fee_rate") or 0.0,
                                          "spread_yes": m.get("spread_yes"), "in_play": m.get("in_play_start_before_forecast") or 0})
        rows.append(s)
    return moments, cuts, rows


def bm(rows, key, n=N_BOOT):
    return scoring.boot_mean(rows, key, n=n, seed=SEED, cluster="event")


def bd(a, b, key, n=N_BOOT):
    return scoring.boot_diff(a, b, key, n=n, seed=SEED, cluster="event")
