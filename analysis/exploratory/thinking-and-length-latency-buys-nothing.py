import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, sqlite3, statistics as st
import scoring as S

BENCH = _paths.BENCH_DB
RUNS = _paths.RUNS_DB

conb = S.connect(BENCH)
moments = S.load_moments(conb)
cuts = S.load_or_compute_cuts(moments)
conr = sqlite3.connect("file:%s?mode=ro" % RUNS, uri=True)
conr.row_factory = sqlite3.Row
EXCLUDE = "openrouter/z-ai/glm-5.3-flash"

rows_by_mw = {}
for r in conr.execute("""SELECT model, row_id, mode, window, parsed_probability, status,
                         latency_s, reasoning_tokens, output_tokens
                         FROM forecasts WHERE model != ? AND mode='normal'""", (EXCLUDE,)):
    if r["status"] not in ("ok", "ok_after_retry") or r["parsed_probability"] is None:
        continue
    m = moments.get(r["row_id"])
    if not m or not m["primary"]:
        continue
    row = S.score_forecast(m, r["parsed_probability"], r["model"], cuts,
                            extra={"latency_s": r["latency_s"], "reasoning_tokens": r["reasoning_tokens"]})
    rows_by_mw.setdefault((r["model"], m["window"]), []).append(row)

print("Model, window, n, median latency, median reasoning tokens, mean Brier[95%CI]")
for w in ("A", "B", "C"):
    print(f"\n--- Window {w} ---")
    wmodels = sorted({k[0] for k in rows_by_mw if k[1] == w})
    lat_by_model = {}
    for mdl in wmodels:
        rows = rows_by_mw[(mdl, w)]
        lat = st.median(r["latency_s"] for r in rows)
        rt = st.median(r["reasoning_tokens"] for r in rows if r["reasoning_tokens"] is not None) if any(r["reasoning_tokens"] is not None for r in rows) else float("nan")
        be, lo, hi = S.boot_mean(rows, "brier_model")
        lat_by_model[mdl] = lat
        print(f"  {mdl:45s} n={len(rows):5d} lat={lat:6.1f}s rt={rt:7.0f} Brier={be:.3f} [{lo:.3f},{hi:.3f}]")
    market_b = st.fmean(r["brier_market"] for r in rows)
    print(f"  {'market':45s}              Brier={market_b:.3f}")
    fastest = min(lat_by_model.values()); slowest = max(lat_by_model.values())
    print(f"  latency ratio slowest/fastest = {slowest/fastest:.1f}")
