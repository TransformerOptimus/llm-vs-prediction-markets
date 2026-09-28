"""Hypothesis: the plateau is worst for gpt-oss-120b and least bad for DeepSeek V4 Pro, on the same markets.

Per window, primary set, Yes-favored band q in 0.75-0.95. For each pair (model X, reference model
gpt-5.2) the paired per-market difference in shortfall (q - p) and in Brier cost (brier_model -
brier_market) on the markets both forecast. Bootstrap: 1,000 resamples of events, seed 20260902."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import statistics as st
from calibration_shape_common import *

rows, cuts = load()
REF = "openai/gpt-5.2"
for w in "ABC":
    print("=== Window", w, "Yes band 0.75-0.95; paired against", SHORT[REF])
    ref = {r["row_id"]: r for r in rows if r["window"] == w and r["forecaster"] == REF and r["band"] == "0.75-0.95"}
    for mdl in MODELS:
        if mdl == REF: continue
        s = []
        for r in rows:
            if r["window"] == w and r["forecaster"] == mdl and r["band"] == "0.75-0.95" and r["row_id"] in ref:
                o = ref[r["row_id"]]
                s.append({"event_id": r["event_id"], "row_id": r["row_id"],
                          "d_short": (o["p"] - r["p"]), "d_cost": (r["brier_model"] - o["brier_model"]),
                          "d_wrong": float(r["p"] < 0.5) - float(o["p"] < 0.5)})
        if not s: continue
        print("  %-16s n=%d  extra shortfall vs ref %s  extra Brier cost vs ref %s  extra wrong-side share %s" % (
            SHORT[mdl], len(s), fmt(boot_mean(s, "d_short")), fmt(boot_mean(s, "d_cost")), fmt(boot_mean(s, "d_wrong"))))
    # gpt-oss vs deepseek-v4-pro head to head where both exist
    a = {r["row_id"]: r for r in rows if r["window"] == w and r["forecaster"] == "openrouter/openai/gpt-oss-120b" and r["band"] == "0.75-0.95"}
    b = {r["row_id"]: r for r in rows if r["window"] == w and r["forecaster"] == "openrouter/deepseek/deepseek-v4-pro" and r["band"] == "0.75-0.95"}
    s = [{"event_id": a[k]["event_id"], "row_id": k, "d_short": b[k]["p"] - a[k]["p"], "d_cost": a[k]["brier_model"] - b[k]["brier_model"]} for k in a if k in b]
    if s:
        print("  gpt-oss-120b vs deepseek-v4-pro head to head n=%d: gpt-oss lower p by %s, higher Brier by %s" % (len(s), fmt(boot_mean(s, "d_short")), fmt(boot_mean(s, "d_cost"))))
    # whole-window Brier cost per model for context
    for mdl in MODELS:
        s = [r for r in rows if r["window"] == w and r["forecaster"] == mdl]
        if s: print("  whole window %-16s Brier model-minus-market %.4f (n %d)" % (SHORT[mdl], st.fmean(r["brier_model"] - r["brier_market"] for r in s), len(s)))
