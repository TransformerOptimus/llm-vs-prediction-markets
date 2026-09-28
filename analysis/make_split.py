"""Draw the calibration split. Read-only on the benchmark.

Rows in play: every market-moment with news_dateline_after_capture = 0 (all windows, ladders
included, since the split has to cover every row a model may be run on). Cells: window x
depth bucket (the window's fixed cuts from out/depth_cuts.json) x is_sports. Within each
cell the rows are sorted by moment_key and round(0.2 x n) rows, at least 1, are drawn with
Python random.Random(20260902).sample over the sorted list. Every cell gets its own draw
from a fresh generator seeded the same way, so adding a cell never changes another.

Writes out/split.csv (moment_key, window, bucket, sports, split) and
out/calibration_moment_keys.txt, to be written into moment_outcomes.split.

Usage: python3 analysis/make_split.py
"""
import csv
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring

SPLIT_SEED = 20260902
SHARE = 0.2


def draw(moments, cuts):
    cells = defaultdict(list)
    for m in moments.values():
        if m["news_dateline_after_capture"]:
            continue
        b = scoring.bucket_of(m, cuts)
        cells[(m["window"], b, m["sports"])].append(m["moment_key"])
    rows, table = [], []
    for key in sorted(cells):
        keys = sorted(cells[key])
        n = len(keys)
        k = max(1, int(round(SHARE * n)))
        chosen = set(random.Random(SPLIT_SEED).sample(keys, k))
        for mk in keys:
            rows.append({"moment_key": mk, "window": key[0], "bucket": key[1], "sports": key[2],
                         "split": "calibration" if mk in chosen else "test"})
        table.append({"window": key[0], "bucket": key[1], "sports": key[2], "rows": n, "calibration": k, "test": n - k})
    return rows, table


def main():
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    rows, table = draw(moments, cuts)
    os.makedirs(scoring.OUT_DIR, exist_ok=True)
    with open(os.path.join(scoring.OUT_DIR, "split.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["moment_key", "window", "bucket", "sports", "split"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(scoring.OUT_DIR, "calibration_moment_keys.txt"), "w") as f:
        for r in rows:
            if r["split"] == "calibration":
                f.write(r["moment_key"] + "\n")
    print("| window | bucket | sports | rows | calibration | test |")
    print("|---|---|---|---|---|---|")
    for t in table:
        print("| %s | %s | %s | %d | %d | %d |" % (t["window"], t["bucket"], t["sports"], t["rows"], t["calibration"], t["test"]))
    n_cal = sum(1 for r in rows if r["split"] == "calibration")
    print("rows in play %d, calibration %d (%.1f%%), test %d, cells %d, smallest cell %d rows"
          % (len(rows), n_cal, 100 * n_cal / len(rows), len(rows) - n_cal, len(table), min(t["rows"] for t in table)))


if __name__ == "__main__":
    main()
