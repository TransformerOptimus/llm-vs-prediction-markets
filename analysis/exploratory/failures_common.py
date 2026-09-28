"""Shared loader for the failures-and-cutoffs scripts. Read-only on both databases."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import json, sqlite3, sys, random, statistics as st
from collections import defaultdict
import scoring

REPO = _paths.REPO
RUNS = REPO + "/harness/runs/runs.db"
EXCLUDE = _roster.VALIDATION       # the probe-validation model, never scored
SEED, NBOOT = 20260902, 1000
SHORT = _roster.SHORT              # display names from harness/models.json
def load_moments():
    con = scoring.connect()
    moments = scoring.load_moments(con)
    for r in con.execute("SELECT row_id, length(question) ql, length(description) dl, length(news_text) nl, "
                         "news_article_count nc, length(coalesce(question_detail,'')) qd FROM market_moments"):
        m = moments[r["row_id"]]
        m["q_len"], m["desc_len"], m["news_len"], m["news_n"], m["detail_len"] = r["ql"], r["dl"] or 0, r["nl"] or 0, r["nc"] or 0, r["qd"]
    cuts = scoring.load_or_compute_cuts(moments)
    return moments, cuts


def load_forecasts(mode="normal"):
    """Final record per (model, mode, row) plus whether any call for it hit the token cap."""
    con = sqlite3.connect("file:%s?mode=ro" % RUNS, uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT model, mode, window, row_id, moment_key, parsed_probability, output_tokens, reasoning_tokens, "
        "finish_reason, status, parse_warning, attempt, strict, length(raw_reply) reply_len FROM forecasts "
        "WHERE model != ? AND mode = ? AND status IN ('ok','ok_after_retry','failed')", (EXCLUDE, mode))]
    capped = set()
    for r in con.execute("SELECT DISTINCT c.model, c.mode, c.row_id FROM calls c JOIN runs r USING(run_id) "
                         "WHERE r.void=0 AND c.finish_reason='length' AND c.mode=? AND c.model != ?", (mode, EXCLUDE)):
        capped.add((r[0], r[2]))
    for r in rows:
        r["capped"] = int((r["model"], r["row_id"]) in capped)
        r["retried"] = int(r["attempt"] == 2 or r["status"] == "failed")   # first answer unusable
        r["short"] = SHORT[r["model"]]
    return rows


def scored(rows, moments, cuts):
    """Score every parsed forecast under scoring.score_forecast; carries the harness fields along."""
    out = []
    for r in rows:
        m = moments.get(r["row_id"])
        if m is None or r["parsed_probability"] is None:
            continue
        s = scoring.score_forecast(m, r["parsed_probability"], r["short"], cuts,
                                   extra={k: r[k] for k in ("capped", "retried", "status", "attempt", "finish_reason",
                                                            "output_tokens", "reasoning_tokens", "parse_warning", "reply_len")})
        s["q_len"], s["desc_len"], s["news_len"], s["news_n"] = m["q_len"], m["desc_len"], m["news_len"], m["news_n"]
        out.append(s)
    return out


def boot_mean(rows, key, seed=SEED, n=NBOOT):
    return scoring.boot_mean(rows, key, n=n, seed=seed)


def boot_diff(a, b, key, seed=SEED, n=NBOOT):
    return scoring.boot_diff(a, b, key, n=n, seed=seed)


def boot_paired(rows, key, seed=SEED, n=NBOOT):
    """Mean of a per-row difference with an interval resampling events."""
    return scoring.boot_mean(rows, key, n=n, seed=seed)


def fmt(t):
    return "%+.4f [%+.4f, %+.4f]" % t
