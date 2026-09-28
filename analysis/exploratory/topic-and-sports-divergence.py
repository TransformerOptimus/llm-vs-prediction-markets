import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys
import common_topic as common, scoring

# Rows: common_topic.load, scored rows (scoring.score_forecast) for every normal-mode forecast with a
# parsed probability, primary set only, no status filter (the row counts, 5087 sports / 1096 non-sports
# on A, include the ok_after_retry rows; common_scripts.build_rows drops them and gives 4687 / 1016).
rows = common.load("normal")
for r in rows:
    r["absdev"] = abs(r["p"] - r["q"])
rows = [r for r in rows if r["primary"]]

for w in ["A", "B", "C"]:
    rw = [r for r in rows if r["window"] == w]
    sp = [r for r in rw if r["sports"]]
    ns = [r for r in rw if not r["sports"]]
    ms, slo, shi = scoring.boot_mean(sp, "absdev")
    mn, nlo, nhi = scoring.boot_mean(ns, "absdev")
    d, dlo, dhi = scoring.boot_diff(ns, sp, "absdev")
    print("Window %s: n_sports=%d n_nonsports=%d sports=%.3f [%.3f,%.3f] nonsports=%.3f [%.3f,%.3f] diff=%.3f [%.3f,%.3f]" %
          (w, len(sp), len(ns), ms, slo, shi, mn, nlo, nhi, d, dlo, dhi))
