import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, os
import scoring
import harness_loader as hl

EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"

def load_all(arm="all"):
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    rows, counts = hl.load_pass("normal", moments, cuts, arm=arm)
    rows = [r for r in rows if r["model_name"] != EXCLUDE_MODEL]
    if _paths.test_split_only():
        rows = [r for r in rows if r.get("split") == "test"]
    return rows, moments, cuts

def primary(rows, moments):
    out = []
    for r in rows:
        m = moments.get(r["row_id"])
        if m is None:
            # try moment_key
            continue
        if _paths.test_split_only() and m["split"] != "test":
            continue
        if m["primary"]:
            out.append(r)
    return out
