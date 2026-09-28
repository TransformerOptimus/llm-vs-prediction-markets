"""Draw the probe-validation sample. Read-only on the benchmark.

The probe-validation model runs in memory-probe mode on a fixed sample of 500 Window A rows
and is never scored; the memory-probe quarantine rule's flag rate on it is compared with the
rate of the three-window models (roster label "bridge") on the same rows.

Population: Window A rows in play (news_dateline_after_capture = 0) that have news
(news_available = 1), which are exactly the rows the memory probe runs on. The keys are
sorted by moment_key and 500 are drawn with Python random.Random(20260902).sample, the same
seed and sorting rule as make_split.py.

Writes out/validation_moment_keys.txt (one moment_key per line, sorted, same format as
out/calibration_moment_keys.txt, for the harness row-list option) and out/validation_sample.md
(population rule and size, seed, draw date, counts per depth bucket, sports flag and split
label, and the benchmark file's sha256 at draw time). The draw date is a constant below so
that running the script twice gives byte-identical files.

Usage: python3 analysis/make_validation_sample.py
"""
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_loader
import scoring

SEED = 20260902          # same seed as make_split.py
SAMPLE_SIZE = 500        # probe-validation sample size
WINDOW = "A"
DRAW_DATE = "2026-09-11"  # the draw date; fixed so reruns are identical


def population(moments):
    """Window A rows in play with news: the rows the memory probe runs on."""
    return {m["moment_key"]: m for m in moments.values()
            if m["window"] == WINDOW and not m["news_dateline_after_capture"] and m["news_available"]}


def draw(pop):
    keys = sorted(pop)
    return sorted(random.Random(SEED).sample(keys, SAMPLE_SIZE))


def counts(chosen, pop, cuts):
    by_bucket = Counter(scoring.bucket_of(pop[k], cuts) for k in chosen)
    by_sports = Counter("sports" if pop[k]["sports"] else "non-sports" for k in chosen)
    by_split = Counter(pop[k]["split"] or "(none)" for k in chosen)
    return by_bucket, by_sports, by_split


def main():
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    db_sha = harness_loader.sha256_file(scoring.DB_PATH)
    pop = population(moments)
    chosen = draw(pop)
    by_bucket, by_sports, by_split = counts(chosen, pop, cuts)

    os.makedirs(scoring.OUT_DIR, exist_ok=True)
    with open(os.path.join(scoring.OUT_DIR, "validation_moment_keys.txt"), "w") as f:
        for k in chosen:
            f.write(k + "\n")

    lines = [
        "# Probe-validation sample",
        "",
        "A fixed sample of Window A rows on which the probe-validation model runs in memory-probe mode. "
        "It is never scored; only the memory-probe quarantine flag rate on these rows is compared with the rate of the three-window models (roster label \"bridge\") on the same rows.",
        "",
        "- Population rule: Window A rows in play (news_dateline_after_capture = 0) with news (news_available = 1), "
        "the rows the memory probe runs on.",
        "- Population size: %d rows" % len(pop),
        "- Sample size: %d rows" % len(chosen),
        "- Draw: keys sorted by moment_key, then random.Random(%d).sample (same seed and sorting rule as make_split.py)" % SEED,
        "- Draw date: %s" % DRAW_DATE,
        "- Benchmark file sha256 at draw time: %s" % db_sha,
        "- Key list: out/validation_moment_keys.txt, one moment_key per line, sorted",
        "",
        "## Drawn rows per depth bucket (Window A cuts from out/depth_cuts.json)",
        "",
        "| bucket | rows |",
        "|---|---|",
    ]
    lines += ["| %s | %d |" % (b, by_bucket[b]) for b in sorted(by_bucket, key=lambda x: (x is None, x))]
    lines += ["", "## Drawn rows by sports flag", "", "| group | rows |", "|---|---|"]
    lines += ["| %s | %d |" % (g, by_sports[g]) for g in sorted(by_sports)]
    lines += ["", "## Drawn rows by calibration split label (moment_outcomes.split)", "", "| split | rows |", "|---|---|"]
    lines += ["| %s | %d |" % (s, by_split[s]) for s in sorted(by_split)]
    lines.append("")
    with open(os.path.join(scoring.OUT_DIR, "validation_sample.md"), "w") as f:
        f.write("\n".join(lines))

    print("population %d, drawn %d" % (len(pop), len(chosen)))
    print("by bucket:", dict(sorted(by_bucket.items(), key=lambda x: (x[0] is None, x[0]))))
    print("by sports:", dict(by_sports))
    print("by split:", dict(by_split))
    print("db sha256:", db_sha)


if __name__ == "__main__":
    main()
