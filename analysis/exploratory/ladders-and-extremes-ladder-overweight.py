import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, sqlite3, math, random, statistics as st
from collections import defaultdict
import scoring

REPO = _paths.REPO
bcon = scoring.connect(REPO + "/benchmark/benchmark.db")
moments = scoring.load_moments(bcon)

rcon = sqlite3.connect("file:%s/harness/runs/runs.db?mode=ro" % REPO, uri=True)
EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"

fc = rcon.execute("""SELECT model, mode, window, row_id, moment_key, parsed_probability, status
                      FROM forecasts WHERE mode='normal' AND model != ?""", (EXCLUDE_MODEL,)).fetchall()

def eligible(m):
    return (not m["news_dateline_after_capture"]) and (not (m.get("end_date_revised") or 0))

def boot_ci(diffs_a, diffs_b, cluster_a, cluster_b, seed=20260902, n=1000):
    # diffs_a, diffs_b: lists of values; cluster_*: parallel list of cluster ids
    rng = random.Random(seed)
    clusters_a = defaultdict(list)
    for v, c in zip(diffs_a, cluster_a):
        clusters_a[c].append(v)
    clusters_b = defaultdict(list)
    for v, c in zip(diffs_b, cluster_b):
        clusters_b[c].append(v)
    ka, kb = list(clusters_a.keys()), list(clusters_b.keys())
    means = []
    for _ in range(n):
        ra = [random.Random((seed, i)).choice(ka) for i in range(len(ka))]
        # simpler: sample with replacement using rng
        sa = [rng.choice(ka) for _ in range(len(ka))]
        sb = [rng.choice(kb) for _ in range(len(kb))]
        va = [x for c in sa for x in clusters_a[c]]
        vb = [x for c in sb for x in clusters_b[c]]
        if not va or not vb:
            continue
        means.append(st.mean(va) - st.mean(vb))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return lo, hi

print("=== Reproduce: mean p for q<0.05, ladder vs stand-alone, by window ===")
for win in ["A", "B", "C"]:
    ladder_p, sa_p = [], []
    ladder_cl, sa_cl = [], []
    ladder_events, sa_events = set(), set()
    ladder_markets, sa_markets = set(), set()
    for row in fc:
        if row[2] != win:  # window
            continue
        rid = row[3]
        m = moments.get(rid)
        if m is None or not eligible(m):
            continue
        q = m["q"]
        if q is None or q >= 0.05:
            continue
        p = row[5]
        if p is None:
            continue
        cl = m["event_id"]
        if m["is_ladder"]:
            ladder_p.append(p); ladder_cl.append(cl)
            ladder_events.add(cl); ladder_markets.add(rid)
        else:
            sa_p.append(p); sa_cl.append(cl)
            sa_events.add(cl); sa_markets.add(rid)
    if not ladder_p or not sa_p:
        print(win, "insufficient data", len(ladder_p), len(sa_p)); continue
    lo, hi = boot_ci(ladder_p, sa_p, ladder_cl, sa_cl)
    print("%s: ladder mean=%.3f n=%d events=%d markets=%d | stand-alone mean=%.3f n=%d events=%d markets=%d | diff=%.3f [%.3f,%.3f]" % (
        win, st.mean(ladder_p), len(ladder_p), len(ladder_events), len(ladder_markets),
        st.mean(sa_p), len(sa_p), len(sa_events), len(sa_markets),
        st.mean(ladder_p) - st.mean(sa_p), lo, hi))

print()
print("=== Confound check 1: does the effect survive dropping the single largest event cluster (ladder side)? ===")
for win in ["A", "B", "C"]:
    ladder_p, sa_p = [], []
    ladder_cl, sa_cl = [], []
    per_event = defaultdict(list)
    for row in fc:
        if row[2] != win:
            continue
        rid = row[3]
        m = moments.get(rid)
        if m is None or not eligible(m):
            continue
        q = m["q"]
        if q is None or q >= 0.05:
            continue
        p = row[5]
        if p is None:
            continue
        cl = m["event_id"]
        if m["is_ladder"]:
            ladder_p.append(p); ladder_cl.append(cl)
            per_event[cl].append(p)
        else:
            sa_p.append(p); sa_cl.append(cl)
    if not per_event:
        continue
    biggest = max(per_event, key=lambda k: len(per_event[k]))
    ladder_p2 = [p for p, c in zip(ladder_p, ladder_cl) if c != biggest]
    ladder_cl2 = [c for c in ladder_cl if c != biggest]
    if not ladder_p2 or not sa_p:
        continue
    print("%s: biggest ladder event=%s (n=%d of %d); mean p excluding it=%.3f vs stand-alone %.3f (orig ladder mean %.3f)" % (
        win, biggest, len(per_event[biggest]), len(ladder_p), st.mean(ladder_p2), st.mean(sa_p), st.mean(ladder_p)))

print()
print("=== Confound check 2: sports share in ladder vs stand-alone (q<0.05) ===")
for win in ["A", "B", "C"]:
    ladder_sports, ladder_n = 0, 0
    sa_sports, sa_n = 0, 0
    seen_rows = set()
    for row in fc:
        if row[2] != win:
            continue
        rid = row[3]
        if rid in seen_rows:
            continue
        seen_rows.add(rid)
        m = moments.get(rid)
        if m is None or not eligible(m):
            continue
        q = m["q"]
        if q is None or q >= 0.05:
            continue
        if m["is_ladder"]:
            ladder_n += 1; ladder_sports += m["sports"]
        else:
            sa_n += 1; sa_sports += m["sports"]
    if ladder_n and sa_n:
        print("%s: ladder sports share=%.2f (n=%d) | stand-alone sports share=%.2f (n=%d)" % (
            win, ladder_sports/ladder_n, ladder_n, sa_sports/sa_n, sa_n))

print()
print("=== Confound check 3: is this just the 0.90-1.1 sum-of-rungs figure driven by huge outlier models? Per-model mean p ladder vs stand-alone, window A ===")
per_model = defaultdict(lambda: [[], []])
for row in fc:
    if row[2] != "A":
        continue
    model = row[0]
    rid = row[3]
    m = moments.get(rid)
    if m is None or not eligible(m):
        continue
    q = m["q"]
    if q is None or q >= 0.05:
        continue
    p = row[5]
    if p is None:
        continue
    idx = 0 if m["is_ladder"] else 1
    per_model[model][idx].append(p)
for model, (lad, sa) in sorted(per_model.items()):
    if lad and sa:
        print("%s: ladder mean=%.3f (n=%d) stand-alone mean=%.3f (n=%d)" % (model, st.mean(lad), len(lad), st.mean(sa), len(sa)))
