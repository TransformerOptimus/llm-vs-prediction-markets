import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""baselines-whole-window-edge: each model's whole-window Brier edge against the market, with its
interval.

The market is more accurate than every model. Figure 2 shows that per depth bucket, with intervals;
this script gives each model's whole-window edge with its interval, so the distance of each overall
shortfall from zero can be read directly.

Rows: primary set, test split (the rows of the result tables), forecasts view, mode normal, every
scored model's final forecasts with a parsed probability. Brier edge = market Brier minus model Brier
per row, averaged over the model's rows, so the market is always scored on exactly the rows the model
answered; negative means the market was more accurate. Intervals: 2,000 event resamples, seed
20260902, the same count the main tables use. Read-only on both databases.
"""
from calibration_shape_common import MODELS, SHORT, boot_stat, load

N = 2000

rows_all, _ = load("normal")
rows_all = [r for r in rows_all if r["split"] == "test"]
print("rows: primary set, test split, forecasts view, mode normal; Brier edge = market Brier minus "
      "model Brier on the model's own rows (negative = market more accurate); intervals = %d event "
      "resamples, seed 20260902" % N)

for w in "ABC":
    print("=== window %s" % w)
    for mdl in MODELS:
        s = [r for r in rows_all if r["window"] == w and r["forecaster"] == mdl]
        if not s:
            continue
        pt, lo, hi = boot_stat(s, lambda rs: sum(r["brier_edge"] for r in rs) / len(rs), n=N)
        mb = sum(r["brier_model"] for r in s) / len(s)
        qb = sum(r["brier_market"] for r in s) / len(s)
        print("  %-16s n=%-5d model Brier %.4f  market Brier %.4f  edge %+.4f [%+.4f, %+.4f]%s"
              % (SHORT[mdl], len(s), mb, qb, pt, lo, hi, "  clear of zero" if hi < 0 else ""))
