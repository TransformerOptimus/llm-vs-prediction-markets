import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, sqlite3, math, random
import scoring as S

BENCH = _paths.BENCH_DB
RUNS = _paths.RUNS_DB

conb = S.connect(BENCH)
moments = S.load_moments(conb)

conr = sqlite3.connect("file:%s?mode=ro" % RUNS, uri=True)
conr.row_factory = sqlite3.Row

EXCLUDE = "openrouter/z-ai/glm-5.3-flash"

normal = {}
probe = {}
for r in conr.execute("SELECT model, row_id, mode, parsed_probability, status FROM forecasts WHERE model != ?", (EXCLUDE,)):
    if r["status"] not in ("ok", "ok_after_retry") or r["parsed_probability"] is None:
        continue
    key = (r["model"], r["row_id"])
    if r["mode"] == "normal":
        normal[key] = r["parsed_probability"]
    elif r["mode"] == "memory_probe":
        probe[key] = r["parsed_probability"]

# Build paired rows: model, row_id present in both normal and memory_probe
pairs = []
for key in normal:
    if key in probe:
        model, row_id = key
        m = moments.get(row_id)
        if not m or not m["primary"]:
            continue
        if not m["news_available"]:
            continue
        pairs.append({
            "model": model, "row_id": row_id, "window": m["window"],
            "sports": m["sports"], "event_id": m["event_id"],
            "move": abs(normal[key] - probe[key]),
            "p_news": normal[key], "p_nonews": probe[key],
            "q": m["q"], "y": m["outcome_yes"],
            "brier_news": (normal[key]-m["outcome_yes"])**2,
            "brier_nonews": (probe[key]-m["outcome_yes"])**2,
        })

print("total paired primary+news_available rows:", len(pairs))

for w in ("A", "B", "C"):
    wr = [r for r in pairs if r["window"] == w]
    sp = [r for r in wr if r["sports"] == 1]
    ns = [r for r in wr if r["sports"] == 0]
    mv_ns, lo_ns, hi_ns = S.boot_mean(ns, "move")
    mv_sp, lo_sp, hi_sp = S.boot_mean(sp, "move")
    diff, dlo, dhi = S.boot_diff(ns, sp, "move")
    print(f"\nWindow {w}: n_nonsports={len(ns)} n_sports={len(sp)}")
    print(f"  move nonsports={mv_ns:.3f} [{lo_ns:.3f},{hi_ns:.3f}]  sports={mv_sp:.3f} [{lo_sp:.3f},{hi_sp:.3f}]")
    print(f"  diff (nonsports-sports) = {diff:+.3f} [{dlo:+.3f},{dhi:+.3f}]")
    # brier improvement: brier_nonews - brier_news (positive = news helped)
    for grp_name, grp in (("nonsports", ns), ("sports", sp)):
        g2 = [dict(r, imp=r["brier_nonews"]-r["brier_news"]) for r in grp]
        im, ilo, ihi = S.boot_mean(g2, "imp")
        print(f"    Brier improvement {grp_name}: {im:+.4f} [{ilo:+.4f},{ihi:+.4f}]")
