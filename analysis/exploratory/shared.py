import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sqlite3, os, random, math
from collections import defaultdict

REPO = _paths.REPO
RUNS_DB = "file:%s?mode=ro" % _paths.RUNS_DB
BENCH_DB = "file:%s?mode=ro" % _paths.BENCH_DB

WINDOWS = {("polymarket", "polybench_snapshot"): "A",
           ("polymarket", "pmxt_archive"): "B",
           ("kalshi", "pmxt_archive"): "C"}

EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"

def moments():
    c = sqlite3.connect(BENCH_DB, uri=True)
    c.row_factory = sqlite3.Row
    cur = c.cursor()
    cur.execute("""
      select m.row_id, m.venue, m.book_source, m.moment_key, m.price_yes, m.mid_yes,
             m.spread_yes, m.book_one_sided, m.is_ladder, m.end_date_revised,
             m.news_dateline_after_capture, m.news_available, m.depth_usd,
             o.outcome_yes, o.split, o.source_event_id, o.topic_tags
      from market_moments m join moment_outcomes o on o.row_id = m.row_id
    """)
    out = {}
    for r in cur.fetchall():
        w = WINDOWS.get((r["venue"], r["book_source"]))
        if w is None:
            continue
        q = r["mid_yes"] if (w == "A" and r["mid_yes"] is not None) else r["price_yes"]
        out[r["row_id"]] = dict(row_id=r["row_id"], window=w, moment_key=r["moment_key"],
                                 q=q, outcome=r["outcome_yes"], is_ladder=r["is_ladder"],
                                 end_date_revised=r["end_date_revised"],
                                 news_dateline_after_capture=r["news_dateline_after_capture"],
                                 news_available=r["news_available"], depth_usd=r["depth_usd"],
                                 split=r["split"], event_id=r["source_event_id"] or r["moment_key"],
                                 topic_tags=r["topic_tags"])
    return out

def forecasts(mode=None):
    c = sqlite3.connect(RUNS_DB, uri=True)
    c.row_factory = sqlite3.Row
    cur = c.cursor()
    q = "select model, mode, window, row_id, moment_key, parsed_probability, status, finish_reason from forecasts where model != ?"
    args = [EXCLUDE_MODEL]
    if mode:
        q += " and mode = ?"
        args.append(mode)
    cur.execute(q, args)
    return [dict(r) for r in cur.fetchall()]

def primary_ok(m):
    return (m["is_ladder"] == 0 and m["end_date_revised"] == 0
            and m["news_dateline_after_capture"] == 0 and 0.05 <= m["q"] <= 0.95)

def brier(p, y):
    return (p - y) ** 2

def bootstrap_ci(values, weights_key_fn, n=1000, seed=20260902):
    """values: list of (cluster_key, val). Cluster bootstrap over clusters, mean of vals."""
    rnd = random.Random(seed)
    clusters = defaultdict(list)
    for k, v in values:
        clusters[k].append(v)
    keys = list(clusters.keys())
    if not keys:
        return (float('nan'), float('nan'), float('nan'))
    point = sum(v for vs in clusters.values() for v in vs) / sum(len(vs) for vs in clusters.values())
    boot = []
    for _ in range(n):
        sample_keys = [keys[rnd.randrange(len(keys))] for _ in range(len(keys))]
        vals = []
        for k in sample_keys:
            vals.extend(clusters[k])
        if vals:
            boot.append(sum(vals) / len(vals))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot))]
    return point, lo, hi
