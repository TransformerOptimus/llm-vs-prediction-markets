"""Shared loader for the news-effect angle: one paired row per (model, window, market) with
the with-news (normal) and no-news (memory probe) forecast on the same market-moment.
Read-only on both databases. d_brier = brier(no news) - brier(with news); positive = news helped."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import os, re, sys, sqlite3, json, random, statistics as st
from collections import defaultdict
REPO = _paths.REPO
import scoring, harness_loader

EXCLUDE = {"openrouter/z-ai/glm-5.3-flash"}
SEED, N_BOOT = 20260902, 1000
CITE_RE = re.compile(r"\b(news|article|articles|report|reports|reported|reporting|headline|headlines|coverage|according to|source|sources)\b", re.I)


def load_pairs(split=None, primary_only=True):
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    by_key = {m["moment_key"]: m for m in moments.values()}
    rcon = harness_loader.connect_runs()
    sql = ("SELECT model, mode, window, moment_key, row_id, parsed_probability, raw_reply, status "
           "FROM forecasts WHERE mode IN ('normal','memory_probe') AND parsed_probability IS NOT NULL")
    recs = defaultdict(dict)
    for r in rcon.execute(sql):
        if r["model"] in EXCLUDE:
            continue
        recs[(r["model"], r["moment_key"])][r["mode"]] = dict(r)
    out = []
    for (model, key), d in recs.items():
        if "normal" not in d or "memory_probe" not in d:
            continue
        m = by_key.get(key)
        if m is None or m["row_id"] != d["normal"]["row_id"]:
            continue
        if not m["news_available"] or m["news_dateline_after_capture"]:
            continue
        if primary_only and not m["primary"]:
            continue
        if split and m["split"] != split:
            continue
        if _paths.test_split_only() and m["split"] != "test":
            continue
        pn, pp = float(d["normal"]["parsed_probability"]), float(d["memory_probe"]["parsed_probability"])
        q, y = m["q"], float(m["outcome_yes"])
        n_art = con.execute("SELECT news_article_count FROM market_moments WHERE row_id=?", (m["row_id"],)).fetchone()[0]
        reply = d["normal"]["raw_reply"] or ""
        tn, tp = scoring.trade(pn, m), scoring.trade(pp, m)
        out.append({
            "model": model.split("/")[-1], "window": m["window"], "row_id": m["row_id"], "moment_key": key,
            "event_id": m["event_id"], "split": m["split"], "p_normal": pn, "p_probe": pp, "q": q, "y": y,
            "brier_normal": (pn - y) ** 2, "brier_probe": (pp - y) ** 2, "brier_market": (q - y) ** 2,
            "d_brier": (pp - y) ** 2 - (pn - y) ** 2,           # positive = news helped
            "move": pn - pp,                                     # how far news moved the forecast
            "abs_move": abs(pn - pp),
            "move_toward_outcome": (pn - pp) * (1 if y == 1 else -1),   # positive = moved toward the truth
            "move_toward_market": (abs(pp - q) - abs(pn - q)),          # positive = news pulled model toward market
            "n_articles": n_art, "cites_news": int(bool(CITE_RE.search(reply))),
            "band": scoring.band_of(q), "horizon_band": m["horizon_band"], "horizon_days": m["horizon_days"],
            "sports": m["sports"], "bucket": scoring.bucket_of(m, cuts), "depth_usd": m["depth_usd"],
            "ret_normal": tn.get("ret"), "ret_probe": tp.get("ret"),
            "side_normal": tn.get("side"), "side_probe": tp.get("side"),
            "topic_tags": m["topic_tags"], "question": m["question"],
        })
    return out


def boot_mean(rows, key, seed=SEED, n=N_BOOT):
    """Mean with a 95% interval resampling events (cluster bootstrap), 1,000 draws."""
    return scoring.boot_mean(rows, key, n=n, seed=seed, cluster="event")


def boot_diff(a, b, key, seed=SEED, n=N_BOOT):
    return scoring.boot_diff(a, b, key, n=n, seed=seed, cluster="event")


def fmt(t):
    return "%+.4f [%+.4f, %+.4f]" % t
