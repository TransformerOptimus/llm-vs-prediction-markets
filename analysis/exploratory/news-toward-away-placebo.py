import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""news-toward-away-placebo: how much of the "news that moves a forecast toward the market helps"
result is true by construction?

News-bearing rows can be split by whether the news moved the forecast toward the market price or away
from it: toward is worth about +0.05 in Brier and away costs about -0.04. It is tempting to read that as
the model misreading news that moves it away from the consensus.

That reading needs a check. "Toward" is defined using the market price, and the market price
predicts the outcome. So ANY change that happens to land closer to the price will improve the Brier
score on average, whether it carried information or not. The split is partly a selection effect.

The placebo: take each model's no-news forecast on the same rows, move it by the same distance the
news actually moved it, but in a randomly chosen direction, and split the result by toward and away
exactly as the real analysis does. Nothing in the placebo knows anything. Whatever gap it produces is
the part of the real gap that the definition creates by itself.

Reported per window: the real toward and away effects and their difference, the placebo's, and the
share of the real difference the placebo reproduces. A placebo share near 100 per cent would mean the
result is an artefact; near zero would mean the split is carrying real information. Draws: 200 random
sign assignments, seed 20260902, so the placebo has its own range rather than a single number.

Convention, matching news-effect-toward-market: the value of the news is the Brier score WITHOUT the
news minus the Brier score WITH it, so a positive number means the news helped; movements smaller than
0.02 count as no movement and fall in neither group.
"""
import random

import numpy as np
from calibration_shape_common import MODELS, SHORT, load

MOVE_MIN = 0.02
N_DRAW = 200
SEED = 20260902


def split_by(pairs, toward_flag):
    tw, aw = [], []
    for (p1, p0, y), t in zip(pairs, toward_flag):
        if t is None:
            continue
        gain = (p0 - y) ** 2 - (p1 - y) ** 2
        (tw if t else aw).append(gain)
    return tw, aw


def mean(v):
    return float(np.mean(v)) if v else float("nan")


normal, cuts = load("normal")
probe, _ = load("memory_probe")
probe_by = {(r["forecaster"], r["moment_key"]): r for r in probe}
print("rows: news-bearing primary rows where both the news pass and the memory-probe pass parsed; "
      "value of news = Brier without minus Brier with, so positive means the news helped; moves under "
      "%.2f are not counted; placebo = the same move in a random direction, %d draws, seed %d"
      % (MOVE_MIN, N_DRAW, SEED))

for w in "ABC":
    print("=== window %s" % w)
    for which in ("test split", "both splits"):
        pooled = []
        for mdl in MODELS:
            for r in normal:
                if r["window"] != w or r["forecaster"] != mdl or not r["news_available"]:
                    continue
                if which == "test split" and r["split"] != "test":
                    continue
                b = probe_by.get((mdl, r["moment_key"]))
                if b is None:
                    continue
                pooled.append((r["p"], b["p"], r["outcome_yes"], r["q"]))
        if len(pooled) < 200:
            continue

        # Real split: toward means the news forecast is closer to the price than the no-news one.
        real_flags = []
        for p1, p0, y, q in pooled:
            real_flags.append(None if abs(p1 - p0) < MOVE_MIN else abs(p1 - q) < abs(p0 - q))
        tw, aw = split_by([(a, b, c) for a, b, c, _ in pooled], real_flags)
        real_diff = mean(tw) - mean(aw)
        print("  %-12s n=%-6d real   toward %+.4f (n=%d)  away %+.4f (n=%d)  difference %+.4f"
              % (which, len(pooled), mean(tw), len(tw), mean(aw), len(aw), real_diff))

        # Placebo: same move size from the no-news forecast, random direction.
        rng = random.Random(SEED)
        diffs, tws, aws = [], [], []
        for _ in range(N_DRAW):
            fake, flags = [], []
            for p1, p0, y, q in pooled:
                d = abs(p1 - p0)
                pf = min(max(p0 + (d if rng.random() < 0.5 else -d), 0.0), 1.0)
                fake.append((pf, p0, y))
                flags.append(None if d < MOVE_MIN else abs(pf - q) < abs(p0 - q))
            t, a = split_by(fake, flags)
            tws.append(mean(t)); aws.append(mean(a)); diffs.append(mean(t) - mean(a))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        share = 100 * float(np.mean(diffs)) / real_diff if real_diff else float("nan")
        print("  %-12s %-8s placebo toward %+.4f            away %+.4f            difference %+.4f "
              "[%+.4f, %+.4f]  = %.0f%% of the real difference"
              % ("", "", float(np.mean(tws)), float(np.mean(aws)), float(np.mean(diffs)), lo, hi, share))
