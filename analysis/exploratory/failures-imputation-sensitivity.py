import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""failures-imputation-sensitivity: 79 replies never produced a readable probability and are dropped
from every score. Does dropping them change any headline?

A row is recorded as failed when the model's reply has no usable PROBABILITY line, the stricter re-ask
also fails, and no number can be parsed. Those rows are simply absent from the scores. If a model
tends to fail exactly on the rows it would have got wrong, dropping them quietly improves its score.
Does the choice matter? This script answers that by scoring the failed rows instead of dropping them, two ways:

  drop (current)   the failed rows are left out. This reproduces the paper's headline Brier edge.
  filled with 0.5  every failed row is scored as if the model had answered even money. This is the
                   neutral filling: 0.5 is the least informative answer a forecaster can give, and it
                   is what a forecaster who refused to answer has effectively said.
  worst case       every failed row is scored as if the model had given the answer furthest from what
                   actually happened -- 0 when the outcome was Yes, 1 when it was No. This is the
                   worst that filling the gap could possibly do, so if the headline survives this it
                   survives any filling.

Brier edge is the market's Brier minus the model's Brier on the same rows: positive means the model
beat the price, negative means the price won. The market's score is included for the failed rows too,
so the comparison stays like for like -- the market always has a price, whether or not the model
answered.

Reported per model and window: how many rows were scored, how many failed, the Brier edge under each
of the three treatments, and the shift from the current number. The test split is the headline; both
splits follow for reference. Intervals are 1,000 event resamples, seed 20260902, resampling events.
Read-only on both databases.
"""
from collections import defaultdict

import _roster
import scoring
from calibration_shape_common import MODELS, SHORT, SEED, N_BOOT, boot_stat

WINDOWS = ("A", "B", "C")
STATUS_OK = ("ok", "ok_after_retry")


def load_rows():
    """Every primary-set final forecast, whether or not a probability could be parsed.

    Rows carry q, the outcome, the event id (the bootstrap cluster) and the model's probability, which
    is None when the reply failed to parse."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    con.close()
    bykey = {m["moment_key"]: m for m in moments.values()}
    rc = _paths.runs_con()
    out = []
    sql = ("SELECT model, moment_key, status, parsed_probability p FROM forecasts "
           "WHERE mode = 'normal' AND model != ?")
    for r in rc.execute(sql, (_roster.VALIDATION,)):
        m = bykey.get(r["moment_key"])
        if m is None or not m["primary"] or r["status"] not in STATUS_OK + ("failed",):
            continue
        out.append({"forecaster": r["model"], "window": m["window"], "split": m["split"],
                    "event_id": m["event_id"], "q": m["q"], "y": float(m["outcome_yes"]),
                    "p": None if r["p"] is None else float(r["p"]),
                    "failed": int(r["status"] == "failed")})
    rc.close()
    return out


def edge(rows, fill):
    """Mean Brier edge, with failed rows treated per `fill`: None drops them."""
    tot = n = 0.0
    for r in rows:
        p = r["p"]
        if p is None:
            if fill is None:
                continue
            p = 0.5 if fill == "half" else (0.0 if r["y"] >= 0.5 else 1.0)
        tot += (r["q"] - r["y"]) ** 2 - (p - r["y"]) ** 2
        n += 1
    return tot / n if n else float("nan")


def fmt(t):
    return "%+.4f [%+.4f, %+.4f]" % t


rows_all = load_rows()
print("rows: primary set, forecasts view, mode normal, status in %s or 'failed', probe-validation "
      "model excluded. 'failed' = the reply produced no parseable probability after the stricter "
      "re-ask. Brier edge = market Brier minus model Brier on the same rows; positive means the model "
      "beat the price. Intervals = %d event resamples, seed %d." % (STATUS_OK, N_BOOT, SEED))

rc = _paths.runs_con()
all_failed = rc.execute("SELECT count(*) FROM forecasts WHERE mode = 'normal' AND status = 'failed' "
                        "AND model != ?", (_roster.VALIDATION,)).fetchone()[0]
rc.close()
tot_failed = sum(r["failed"] for r in rows_all)
print("failed rows across all scored normal-mode runs: %d. The rest of this script is about the "
      "primary analysis set only, which is where the headlines are scored; the others sit on rows no "
      "headline uses (ladder rows, out-of-price-range rows and the other primary-set exclusions)."
      % all_failed)
per = defaultdict(int)
for r in rows_all:
    if r["failed"]:
        per[(r["forecaster"], r["window"])] += 1
print("failed rows in the primary set: %d in total, from %d model-window cells: %s"
      % (tot_failed, len(per), "; ".join("%s %s %d" % (SHORT[m], w, n) for (m, w), n in sorted(per.items()))))

for w in WINDOWS:
    for which, keep in (("test split", lambda r: r["split"] == "test"), ("both splits", lambda r: True)):
        base = [r for r in rows_all if r["window"] == w and keep(r)]
        if not base:
            continue
        print("== window %s, %s" % (w, which))
        for mdl in MODELS:
            s = [r for r in base if r["forecaster"] == mdl]
            if len(s) < 50:
                continue
            nf = sum(r["failed"] for r in s)
            e_drop = boot_stat(s, lambda rs: edge(rs, None))
            if nf == 0:
                print("  %-16s scored %-5d failed %-3d | edge %s  (no failed rows: nothing to impute)"
                      % (SHORT[mdl], len(s) - nf, nf, fmt(e_drop)))
                continue
            e_half = boot_stat(s, lambda rs: edge(rs, "half"))
            e_worst = boot_stat(s, lambda rs: edge(rs, "worst"))
            print("  %-16s scored %-5d failed %-3d (%.2f%% of the cell)"
                  % (SHORT[mdl], len(s) - nf, nf, 100.0 * nf / len(s)))
            print("      drop (current)   edge %s" % fmt(e_drop))
            print("      filled with 0.5  edge %s   shift %+.5f" % (fmt(e_half), e_half[0] - e_drop[0]))
            print("      worst case       edge %s   shift %+.5f" % (fmt(e_worst), e_worst[0] - e_drop[0]))
