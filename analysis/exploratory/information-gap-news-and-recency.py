import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
"""information-gap-news-and-recency: is the sorting gap a matter of the model knowing less, or of the
model judging worse?

A forecaster with no current information must score worse on resolution than one with full information,
whatever its judgement is like. Market prices carry same-day information; our models do not, and most
rows carry no news at all. So before reading the resolution gap as a statement about reasoning, we have
to ask how much of it moves when the information changes. Three contrasts, weakest confound first:

  1. WITHIN ROW. Every news-bearing row was also asked with the news removed (the memory-probe pass).
     Same model, same question, same moment, same outcome: only the information differs. Murphy
     decomposition of both passes on those rows, and the change in resolution. This is the only
     contrast here with nothing else moving, so it leads.

  2. BETWEEN ROWS. Rows that carry news against rows that do not, within one window and venue, per
     model. Confounded by question mix: newsworthy markets are not a random sample.

  3. SPORTS. Sports against non-sports rows. Sports prices track same-day information hardest
     (injuries, line-ups, betting lines), and sports is most of the sample, so if information drives
     the gap it should be widest here.

  4. RECENCY. Each model's gap against the days between its pinned checkpoint's release date and the
     forecast. Reported last and read with care: each window is a different venue, date range and
     question mix, so a model's recency and its window are tangled together and this contrast cannot
     carry the argument by itself.

Throughout, RES is resolution (higher is better: the ability to sort events that happen from events
that do not) and REL is reliability (lower is better: calibration error). The market's own RES is
reported beside the model's, because a subset with a different question mix changes what there is to
resolve. Ten equal-width bins, as elsewhere in this analysis. Intervals are 1,000 event resamples,
seed 20260902. Test split and both splits are both reported.
"""
import json
from collections import defaultdict
from datetime import datetime

from calibration_shape_common import (MODELS, SHORT, boot_stat, load, murphy)

WINDOWS = ("A", "B", "C")


def dec(rows, key):
    return murphy([r[key] for r in rows], [r["outcome_yes"] for r in rows])


def line(rows, label, width=34):
    """One decomposition line: model REL/RES, market REL/RES, and the bootstrapped RES gap."""
    if len(rows) < 30:
        print("  %-*s n=%-5d (fewer than 30 rows; not reported)" % (width, label, len(rows)))
        return
    rel_m, res_m, _ = dec(rows, "p")
    rel_q, res_q, unc = dec(rows, "q")
    gap = boot_stat(rows, lambda rs: dec(rs, "q")[1] - dec(rs, "p")[1])
    edge = boot_stat(rows, lambda rs: sum(r["brier_edge"] for r in rs) / len(rs))
    print("  %-*s n=%-5d model REL %.4f RES %.4f | market REL %.4f RES %.4f | UNC %.4f | "
          "market-minus-model RES %+.4f [%+.4f, %+.4f] | Brier edge %+.4f [%+.4f, %+.4f]"
          % (width, label, len(rows), rel_m, res_m, rel_q, res_q, unc, *gap, *edge))


def split_sets(rows):
    return (("test split", [r for r in rows if r["split"] == "test"]),
            ("both splits", rows))


normal, cuts = load("normal")
probe, _ = load("memory_probe")
print("rows: primary set, forecasts view; RES = resolution (higher better), REL = reliability (lower "
      "better), UNC = outcome variance; 10 equal-width bins; intervals = 1,000 event resamples, seed 20260902")

# --- 1. Within row: same rows, news removed ---------------------------------
print()
print("### 1. WITHIN ROW: news-bearing rows, with the news against the same rows with the news removed")
print("### (normal pass against memory-probe pass; same model, question, moment and outcome)")
probe_by = {(r["forecaster"], r["moment_key"]): r for r in probe}
for w in WINDOWS:
    for which, rs in split_sets([r for r in normal if r["window"] == w and r["news_available"]]):
        print("== window %s, %s" % (w, which))
        for mdl in MODELS:
            with_news = [r for r in rs if r["forecaster"] == mdl]
            paired = [(r, probe_by.get((mdl, r["moment_key"]))) for r in with_news]
            paired = [(a, b) for a, b in paired if b is not None]
            if len(paired) < 30:
                continue
            a_rows = [a for a, _ in paired]
            b_rows = [b for _, b in paired]
            rel_a, res_a, _ = dec(a_rows, "p")
            rel_b, res_b, _ = dec(b_rows, "p")
            _, res_q, _ = dec(a_rows, "q")
            # Bootstrap the change in resolution over the same event clusters.
            bykey = {r["moment_key"]: r for r in b_rows}
            d = boot_stat(a_rows, lambda rs: dec(rs, "p")[1]
                          - dec([bykey[r["moment_key"]] for r in rs], "p")[1])
            print("  %-16s n=%-5d RES with news %.4f  without %.4f  change %+.4f [%+.4f, %+.4f] | "
                  "market RES %.4f | REL with %.4f without %.4f"
                  % (SHORT[mdl], len(a_rows), res_a, res_b, *d, res_q, rel_a, rel_b))

# --- 2 and 3. Between rows: news, and sports --------------------------------
for title, field in (("2. BETWEEN ROWS: rows with news against rows without", "news_available"),
                     ("3. SPORTS against non-sports", "sports")):
    print()
    print("### " + title)
    on, off = ("with news", "no news") if field == "news_available" else ("sports", "non-sports")
    for w in WINDOWS:
        for which, rs in split_sets([r for r in normal if r["window"] == w]):
            print("== window %s, %s" % (w, which))
            for mdl in MODELS:
                s = [r for r in rs if r["forecaster"] == mdl]
                if not s:
                    continue
                for lab, sel in ((on, [r for r in s if r[field]]), (off, [r for r in s if not r[field]])):
                    line(sel, "%s, %s" % (SHORT[mdl], lab))

# --- 4. Recency -------------------------------------------------------------
print()
print("### 4. RECENCY: days from the model's pinned checkpoint release to the forecast, against its gap")
print("### (confounded with window: each window is a different venue, date range and question mix)")
roster = json.load(open(os.path.join(_roster.REPO, "harness", "models.json")))
release = {m: v.get("release_date") for m, v in roster.items() if isinstance(v, dict)}
con = __import__("scoring").connect()
ts = {r[0]: r[1] for r in con.execute("SELECT moment_key, forecast_ts FROM market_moments")}
con.close()


def days(mdl, r):
    rel, when = release.get(mdl), ts.get(r["moment_key"])
    if not rel or not when:
        return None
    try:
        return (datetime.fromisoformat(str(when)[:10]) - datetime.fromisoformat(str(rel)[:10])).days
    except ValueError:
        return None


for w in WINDOWS:
    for which, rs in split_sets([r for r in normal if r["window"] == w]):
        print("== window %s, %s" % (w, which))
        for mdl in MODELS:
            s = [r for r in rs if r["forecaster"] == mdl]
            if len(s) < 30:
                continue
            ds = [d for d in (days(mdl, r) for r in s) if d is not None]
            if not ds:
                continue
            rel_m, res_m, _ = dec(s, "p")
            _, res_q, _ = dec(s, "q")
            edge = sum(r["brier_edge"] for r in s) / len(s)
            print("  %-16s release %s  median days to forecast %5d  RES %.4f  market RES %.4f  "
                  "market-minus-model RES %+.4f  Brier edge %+.4f"
                  % (SHORT[mdl], release.get(mdl, "?"), sorted(ds)[len(ds) // 2], res_m, res_q,
                     res_q - res_m, edge))
