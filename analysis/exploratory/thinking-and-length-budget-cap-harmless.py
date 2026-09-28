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

MODEL = "openrouter/deepseek/deepseek-v4-pro"
CAP = 1000

rows_by_w = {"B": [], "C": []}
for r in conr.execute("""SELECT row_id, window, parsed_probability, status, reasoning_tokens,
                         output_tokens, finish_reason FROM forecasts
                         WHERE model=? AND mode='normal'""", (MODEL,)):
    if r["status"] not in ("ok", "ok_after_retry") or r["parsed_probability"] is None:
        continue
    m = moments.get(r["row_id"])
    if not m or not m["primary"] or m["window"] not in ("B", "C"):
        continue
    capped = int((r["reasoning_tokens"] or 0) >= CAP)
    ans_tok = (r["output_tokens"] or 0) - (r["reasoning_tokens"] or 0)
    row = S.score_forecast(m, r["parsed_probability"], MODEL, cuts,
                            extra={"capped": capped, "ans_tok": ans_tok, "reasoning_tokens": r["reasoning_tokens"],
                                   "abs_edge": abs(r["parsed_probability"] - m["q"])})
    rows_by_w[m["window"]].append(row)

for w in ("B", "C"):
    rows = rows_by_w[w]
    capped = [r for r in rows if r["capped"]]
    free = [r for r in rows if not r["capped"]]
    print(f"\nWindow {w}: capped n={len(capped)} free n={len(free)}")
    at, alo, ahi = S.boot_diff(capped, free, "ans_tok")
    bt, blo, bhi = S.boot_diff(capped, free, "brier_model")
    et, elo, ehi = S.boot_diff(capped, free, "brier_edge")
    print(f"  ans_tok diff={at:+.0f} [{alo:+.0f},{ahi:+.0f}]")
    print(f"  Brier diff={bt:+.4f} [{blo:+.4f},{bhi:+.4f}]")
    print(f"  Brier edge diff={et:+.4f} [{elo:+.4f},{ehi:+.4f}]")
