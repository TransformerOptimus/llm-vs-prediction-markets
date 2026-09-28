import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, sqlite3, json, random
import scoring as sc

RUNS_DB = _paths.RUNS_DB

con_b = sc.connect()
moments = sc.load_moments(con_b)

con_r = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
con_r.row_factory = sqlite3.Row

normal = {}
probe = {}
for r in con_r.execute("SELECT model, row_id, parsed_probability, status FROM forecasts WHERE mode='normal' AND model != 'openrouter/z-ai/glm-5.3-flash'"):
    if r["status"] in ("ok", "ok_after_retry") and r["parsed_probability"] is not None:
        normal[(r["model"], r["row_id"])] = r["parsed_probability"]
for r in con_r.execute("SELECT model, row_id, parsed_probability, status FROM forecasts WHERE mode='memory_probe' AND model != 'openrouter/z-ai/glm-5.3-flash'"):
    if r["status"] in ("ok", "ok_after_retry") and r["parsed_probability"] is not None:
        probe[(r["model"], r["row_id"])] = r["parsed_probability"]

pairs = []
for k, p_news in normal.items():
    if k in probe:
        model, row_id = k
        m = moments.get(row_id)
        if m is None or not m["primary"]:
            continue
        pairs.append({"model": model, "row_id": row_id, "moment": m, "p_news": p_news, "p_noNews": probe[k]})

print("total pairs (primary set):", len(pairs))

def bucket(q):
    if q < 0.05:
        return "lo"
    if q > 0.95:
        return "hi"
    return "mid"

# NOTE: primary set requires 0.05<=q<=0.95, so q<0.05 / q>0.95 pairs above are NOT in `pairs`.
# Redo without the primary price filter, keeping ladder/dateline filters.
pairs = []
for k, p_news in normal.items():
    if k in probe:
        model, row_id = k
        m = moments.get(row_id)
        if m is None:
            continue
        if m["is_ladder"] or m["news_dateline_after_capture"] or (m.get("end_date_revised") or 0):
            continue
        pairs.append({"model": model, "row_id": row_id, "moment": m, "p_news": p_news, "p_noNews": probe[k]})

print("total pairs (ladder/dateline/end-date filtered):", len(pairs))

def stats_for(window, qlo, qhi, label):
    rows = [pr for pr in pairs if pr["moment"]["window"] == window and qlo <= pr["moment"]["q"] <= qhi]
    if not rows:
        print(window, label, "n=0")
        return
    y = lambda pr: float(pr["moment"]["outcome_yes"])
    diffs = [ (pr["p_news"]-y(pr))**2 - (pr["p_noNews"]-y(pr))**2 for pr in rows ]
    pdiffs = [ pr["p_news"] - pr["p_noNews"] for pr in rows ]
    events = {pr["moment"]["event_id"] for pr in rows}
    markets = {pr["row_id"] for pr in rows}

    # cluster bootstrap by event
    rng = random.Random(20260902)
    ev_to_idx = {}
    for i, pr in enumerate(rows):
        ev_to_idx.setdefault(pr["moment"]["event_id"], []).append(i)
    ev_list = list(ev_to_idx.keys())
    boot_means = []
    boot_pmeans = []
    for _ in range(1000):
        sample_idx = []
        for _ in range(len(ev_list)):
            ev = rng.choice(ev_list)
            sample_idx.extend(ev_to_idx[ev])
        boot_means.append(sum(diffs[i] for i in sample_idx)/len(sample_idx))
        boot_pmeans.append(sum(pdiffs[i] for i in sample_idx)/len(sample_idx))
    boot_means.sort(); boot_pmeans.sort()
    lo_i, hi_i = int(0.025*1000), int(0.975*1000)
    mean_diff = sum(diffs)/len(diffs)
    mean_pdiff = sum(pdiffs)/len(pdiffs)
    print(f"{window} {label}: n={len(rows)} events={len(events)} markets={len(markets)} "
          f"BrierDiff(news-noNews)={mean_diff:+.4f} [{boot_means[lo_i]:+.4f},{boot_means[hi_i]:+.4f}] "
          f"pDiff={mean_pdiff:+.4f} [{boot_pmeans[lo_i]:+.4f},{boot_pmeans[hi_i]:+.4f}] "
          f"mean_p_news={sum(pr['p_news'] for pr in rows)/len(rows):.3f} mean_p_noNews={sum(pr['p_noNews'] for pr in rows)/len(rows):.3f}")
    return rows, diffs

for w in ("A","B","C"):
    stats_for(w, 0.0, 0.05-1e-9, "q<0.05")
    stats_for(w, 0.05, 0.95, "0.05-0.95")
    stats_for(w, 0.95+1e-9, 1.0, "q>0.95")

print()
print("--- per-model breakdown on q<0.05 ---")
for w in ("A","B","C"):
    print(f"window {w}:")
    rows = [pr for pr in pairs if pr["moment"]["window"] == w and pr["moment"]["q"] < 0.05]
    by_model = {}
    for pr in rows:
        by_model.setdefault(pr["model"], []).append(pr)
    for model, mrows in sorted(by_model.items()):
        y = lambda pr: float(pr["moment"]["outcome_yes"])
        diffs = [ (pr["p_news"]-y(pr))**2 - (pr["p_noNews"]-y(pr))**2 for pr in mrows ]
        print(f"  {model}: n={len(mrows)} meanDiff={sum(diffs)/len(diffs):+.4f}")
