import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""decomposition-why-reliability-is-small: the facts behind the small binned reliability term.

The Murphy decomposition splits a Brier score into reliability minus resolution plus uncertainty.
"Reliability" (also called calibration error) sorts the forecasts into ten equal-width probability
bins and, in each bin, squares the gap between the bin's mean forecast and the share of those rows
that actually resolved Yes, then averages those squared gaps weighted by how many rows sit in the
bin. "Resolution" measures how far the bins' outcome frequencies sit from the overall base rate: it
is the sorting term. Both are in squared-probability units.

Section 6.2 of the paper says the reliability term stays small "because these two errors partly
offset within a bin", the two errors being overconfidence in the model's own ordering and
underconfidence relative to the market. An alternative reading is that this is not how binned
reliability works, and that reliability is small because the models rarely leave the 0.2-0.7
range and because reliability is squared and measured against a much larger resolution gap. This
script does not try to confirm either reading. It prints the facts a correct sentence needs.

Per window, on the test split of the primary analysis set:

  1. WHERE THE FORECASTS SIT. Per model, the share of forecasts in 0.2-0.7, and the 5th, 50th and
     95th percentiles. The market's own prices are printed the same way for contrast.
  2. THE TEN BINS. Per model and pooled over models: each bin's population, its mean forecast, the
     share of its rows that resolved Yes, the signed gap between the two, and that bin's contribution
     to reliability (population share times the squared gap). The contributions sum to the total.
  3. THE MAGNITUDES SIDE BY SIDE. Per model: total reliability, total resolution, their ratio, and
     the market's reliability and resolution on the same rows.
  4. DOES ANYTHING OFFSET INSIDE A BIN? Each bin is cut in two by the market price -- rows the market
     prices ABOVE the bin's mean forecast and rows it prices BELOW -- and the outcome frequency of
     each half is printed. If the model is too low on one half and too high on the other by similar
     amounts, the two gaps cancel when the bin is averaged, which is what "offset within a bin" would
     mean. The size of that cancellation is printed as: the reliability the bin would contribute if
     the two halves were scored as separate bins, against what the whole bin contributes. Cutting any
     bin in two raises binned reliability a little even when the cut is meaningless, so the same cut
     made at RANDOM, with the same half sizes, is printed as the control.

Read-only on both databases; the test split is selected inside the script, so no flag is needed.
"""
import random
import statistics as st
from collections import defaultdict

from calibration_shape_common import MODELS, SHORT, load, murphy

NB = 10
LO, HI = 0.2, 0.7
SEED = 20260902
N_RAND = 200


def pct(v, f):
    v = sorted(v)
    if not v:
        return float("nan")
    return v[min(int(f * len(v)), len(v) - 1)]


def bin_of(p):
    return min(int(p * NB), NB - 1)


def bin_table(ps, ys):
    """[(bin index, n, mean forecast, outcome frequency, signed gap, contribution to reliability)]."""
    n = len(ps)
    b = defaultdict(list)
    for p, y in zip(ps, ys):
        b[bin_of(p)].append((p, y))
    out = []
    for k in sorted(b):
        pp = [p for p, _ in b[k]]
        yy = [y for _, y in b[k]]
        gap = st.fmean(pp) - st.fmean(yy)
        out.append((k, len(pp), st.fmean(pp), st.fmean(yy), gap, len(pp) * gap * gap / n))
    return out


rows_all, cuts = load("normal")
rows_all = [r for r in rows_all if r["split"] == "test"]
print("rows: primary set, forecasts view, mode normal, TEST SPLIT (selected inside this script). "
      "Ten equal-width probability bins, the same binning calibration_shape_common.murphy uses. "
      "Reliability and resolution are in squared-probability units; a gap of 0.05 in probability "
      "contributes 0.0025.")

for w in "ABC":
    rows_w = [r for r in rows_all if r["window"] == w]
    if not rows_w:
        continue
    models = [m for m in MODELS if sum(1 for r in rows_w if r["forecaster"] == m) >= 200]
    print("")
    print("=== window %s: %d test rows, %d models" % (w, len(rows_w), len(models)))

    print("  -- (1) where the forecasts sit")
    print("     %-16s %6s  %10s  %6s %6s %6s" % ("forecaster", "n", "in 0.2-0.7", "p05", "p50", "p95"))
    for m in models:
        ps = [r["p"] for r in rows_w if r["forecaster"] == m]
        share = sum(1 for p in ps if LO <= p <= HI) / len(ps)
        print("     %-16s %6d  %9.1f%%  %6.3f %6.3f %6.3f"
              % (SHORT[m], len(ps), 100 * share, pct(ps, 0.05), pct(ps, 0.50), pct(ps, 0.95)))
    qs = [r["q"] for r in rows_w if r["forecaster"] == models[0]]
    print("     %-16s %6d  %9.1f%%  %6.3f %6.3f %6.3f  (the market, same rows)"
          % ("market price", len(qs), 100 * sum(1 for q in qs if LO <= q <= HI) / len(qs),
             pct(qs, 0.05), pct(qs, 0.50), pct(qs, 0.95)))
    pooled = [r["p"] for r in rows_w]
    print("     %-16s %6d  %9.1f%%  %6.3f %6.3f %6.3f  (every model's forecasts together)"
          % ("all models", len(pooled), 100 * sum(1 for p in pooled if LO <= p <= HI) / len(pooled),
             pct(pooled, 0.05), pct(pooled, 0.50), pct(pooled, 0.95)))

    print("  -- (2) the ten bins, pooled over models")
    print("     %-11s %7s %9s %9s %8s %12s" % ("bin", "n", "mean fcst", "Yes freq", "gap", "contributes"))
    for k, n, mp, my, gap, contrib in bin_table([r["p"] for r in rows_w], [r["outcome_yes"] for r in rows_w]):
        print("     %.1f-%.1f    %7d %9.3f %9.3f %+8.3f %12.5f"
              % (k / NB, (k + 1) / NB, n, mp, my, gap, contrib))
    rel, res, unc = murphy([r["p"] for r in rows_w], [r["outcome_yes"] for r in rows_w])
    print("     pooled totals: reliability %.5f, resolution %.5f, uncertainty %.5f" % (rel, res, unc))

    print("  -- (2) the ten bins, per model")
    for m in models:
        s = [r for r in rows_w if r["forecaster"] == m]
        print("     %s (n=%d)" % (SHORT[m], len(s)))
        for k, n, mp, my, gap, contrib in bin_table([r["p"] for r in s], [r["outcome_yes"] for r in s]):
            print("       %.1f-%.1f  %7d  mean fcst %.3f  Yes freq %.3f  gap %+.3f  contributes %.5f"
                  % (k / NB, (k + 1) / NB, n, mp, my, gap, contrib))

    print("  -- (3) magnitudes side by side")
    print("     %-16s %12s %12s %10s %14s" % ("forecaster", "reliability", "resolution", "rel/res", "Brier"))
    for m in models:
        s = [r for r in rows_w if r["forecaster"] == m]
        rel, res, unc = murphy([r["p"] for r in s], [r["outcome_yes"] for r in s])
        brier = st.fmean(r["brier_model"] for r in s)
        print("     %-16s %12.5f %12.5f %10.3f %14.5f" % (SHORT[m], rel, res, rel / res if res else float("nan"), brier))
    s0 = [r for r in rows_w if r["forecaster"] == models[0]]
    relq, resq, _ = murphy([r["q"] for r in s0], [r["outcome_yes"] for r in s0])
    print("     %-16s %12.5f %12.5f %10.3f %14.5f  (the market, same rows)"
          % ("market", relq, resq, relq / resq if resq else float("nan"),
             st.fmean(r["brier_market"] for r in s0)))
    print("     for scale: the largest per-model reliability here is a mean gap of about %.3f in "
          "probability units (square root of reliability)"
          % max(murphy([r["p"] for r in rows_w if r["forecaster"] == m],
                       [r["outcome_yes"] for r in rows_w if r["forecaster"] == m])[0] ** 0.5
                for m in models))

    print("  -- (4) does anything offset inside a bin? each bin cut by the market price")
    print("     'high half' = rows the market prices above the bin's mean forecast; 'low half' = below.")
    print("     %-11s %6s %8s %8s | %6s %8s %8s | %11s %11s"
          % ("bin", "n hi", "mean q hi", "Yes hi", "n lo", "mean q lo", "Yes lo", "whole bin", "halves"))
    N = len(rows_w)
    b = defaultdict(list)
    for r in rows_w:
        b[bin_of(r["p"])].append(r)
    tot_whole = tot_halves = 0.0
    for k in sorted(b):
        rs = b[k]
        mp = st.fmean(r["p"] for r in rs)
        hi_rows = [r for r in rs if r["q"] > mp]
        lo_rows = [r for r in rs if r["q"] <= mp]
        gap = mp - st.fmean(r["outcome_yes"] for r in rs)
        whole = len(rs) * gap * gap / N
        halves = 0.0
        for half in (hi_rows, lo_rows):
            if half:
                g = st.fmean(r["p"] for r in half) - st.fmean(r["outcome_yes"] for r in half)
                halves += len(half) * g * g / N
        tot_whole += whole
        tot_halves += halves
        print("     %.1f-%.1f    %6d %8.3f %8.3f | %6d %8.3f %8.3f | %11.5f %11.5f"
              % (k / NB, (k + 1) / NB,
                 len(hi_rows), st.fmean(r["q"] for r in hi_rows) if hi_rows else float("nan"),
                 st.fmean(r["outcome_yes"] for r in hi_rows) if hi_rows else float("nan"),
                 len(lo_rows), st.fmean(r["q"] for r in lo_rows) if lo_rows else float("nan"),
                 st.fmean(r["outcome_yes"] for r in lo_rows) if lo_rows else float("nan"),
                 whole, halves))
    # Cutting each bin in two always raises binned reliability a little, even when the cut carries no
    # information, because each half's gap is now fitted on fewer rows. The same cut made at random,
    # with the same half sizes, measures that mechanical part.
    rng = random.Random(SEED)
    rand_tot = []
    for _ in range(N_RAND):
        tot = 0.0
        for k in sorted(b):
            rs = list(b[k])
            mp = st.fmean(r["p"] for r in rs)
            n_hi = sum(1 for r in rs if r["q"] > mp)
            rng.shuffle(rs)
            for half in (rs[:n_hi], rs[n_hi:]):
                if half:
                    g = st.fmean(r["p"] for r in half) - st.fmean(r["outcome_yes"] for r in half)
                    tot += len(half) * g * g / N
        rand_tot.append(tot)
    print("     totals: reliability with whole bins %.5f, with each bin cut in two by the market "
          "price %.5f; the same cut made at random with the same half sizes gives %.5f on average "
          "over %d draws (seed %d), which is the part that is only finer binning"
          % (tot_whole, tot_halves, st.fmean(rand_tot), N_RAND, SEED))
