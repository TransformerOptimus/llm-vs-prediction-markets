import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, statistics as st
from common_ins import load_moments, load_forecasts
import scoring as sc

moments = load_moments()
fc = load_forecasts("normal")

def eligible(m):
    return m["is_ladder"] == 0 and (m.get("end_date_revised") or 0) == 0 and m["news_dateline_after_capture"] == 0

rows_by_window = {"A": [], "B": [], "C": []}
for f in fc:
    m = moments.get(f["row_id"])
    if m is None or not eligible(m):
        continue
    q = m["q"]
    if q >= 0.05:
        continue
    w = m["window"]
    if w not in rows_by_window:
        continue
    rows_by_window[w].append({"p": f["parsed_probability"], "q": q, "row_id": f["row_id"],
                               "event_id": m["event_id"], "model": f["model"], "outcome": m["outcome_yes"]})

for w, rows in rows_by_window.items():
    n = len(rows)
    if n == 0:
        continue
    pmean = st.fmean(r["p"] for r in rows)
    qmean = st.fmean(r["q"] for r in rows)
    share_pgtq = sum(1 for r in rows if r["p"] > r["q"]) / n
    share_p_ge_05 = sum(1 for r in rows if r["p"] >= 0.05) / n
    n_markets = len(set(r["event_id"] for r in rows))
    print(f"Window {w}: n={n} markets~{n_markets} mean p={pmean:.3f} mean q={qmean:.3f} share p>q={share_pgtq:.2f} share p>=0.05={share_p_ge_05:.2f}")

print()
print("=== Per-model breakdown (share p>q) on these rows ===")
for w, rows in rows_by_window.items():
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    print(f"-- window {w} --")
    for model, rs in sorted(by_model.items()):
        share = sum(1 for r in rs if r["p"] > r["q"]) / len(rs)
        print(f"  {model}: n={len(rs)} share p>q={share:.2f} mean p={st.fmean(r['p'] for r in rs):.3f} mean q={st.fmean(r['q'] for r in rs):.3f}")

print()
print("=== CONFOUND CHECK: is there a general 'floor' regardless of q? ===")
print("Look at ALL primary-set rows (q in full range), bucket q into deciles, report mean p - q per decile.")
all_rows = {"A": [], "B": [], "C": []}
for f in fc:
    m = moments.get(f["row_id"])
    if m is None or not m["primary"]:
        continue
    w = m["window"]
    if w not in all_rows:
        continue
    all_rows[w].append({"p": f["parsed_probability"], "q": m["q"]})

for w, rows in all_rows.items():
    print(f"-- window {w} n={len(rows)} --")
    deciles = [(0,0.05),(0.05,0.15),(0.15,0.25),(0.25,0.35),(0.35,0.45),(0.45,0.55),
               (0.55,0.65),(0.65,0.75),(0.75,0.85),(0.85,0.95),(0.95,1.01)]
    for lo, hi in deciles:
        bucket = [r for r in rows if lo <= r["q"] < hi]
        if len(bucket) < 5:
            continue
        diff = st.fmean(r["p"] - r["q"] for r in bucket)
        pmean = st.fmean(r["p"] for r in bucket)
        print(f"  q in [{lo:.2f},{hi:.2f}): n={len(bucket)} mean p={pmean:.3f} mean(p-q)={diff:+.3f}")

print()
print("=== Does the low-p floor hold generally? min p seen per window on ALL rows ===")
for w in ["A","B","C"]:
    ps = [f["parsed_probability"] for f in fc if moments.get(f["row_id"]) and moments[f["row_id"]]["window"]==w]
    ps_sorted = sorted(ps)
    print(f"window {w}: n={len(ps)} min={ps_sorted[0]:.4f} p1={ps_sorted[len(ps)//100]:.4f} p5={ps_sorted[len(ps)//20]:.4f} p10={ps_sorted[len(ps)//10]:.4f}")
