"""Shape of a window's primary sample before any model is scored.

For one window and one arm: rows per depth bucket, the crowd's price band by bucket, news
coverage by bucket, events by bucket and rows per event, book status counts. Plus the
depth-versus-price check: the correlation between log depth and how far the price sits from
0.5, overall and per bucket, for two windows side by side.

Usage: python3 analysis/window_shape.py            (prints the tables for A and B)
"""
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scoring
from scoring import BANDS, ARMS, assign_bucket, band_of, book_status, in_arm

BAND_LABELS = ["%.2f-%.2f" % b for b in BANDS]


def primary_rows(moments: Dict[int, dict], window: str, arm: str = "all") -> List[dict]:
    return [m for m in moments.values() if m["primary"] and m["window"] == window and in_arm(m, arm)]


def with_buckets(rows: List[dict], cuts: List[float]) -> List[dict]:
    out = []
    for m in rows:
        d = dict(m)
        d["bucket"] = assign_bucket(m["depth_usd"], cuts)
        d["band"] = band_of(m["q"])
        d["dist"] = abs(m["q"] - 0.5)
        out.append(d)
    return out


def _pearson(x, z) -> float:
    if len(x) < 3:
        return float("nan")
    mx, mz = st.fmean(x), st.fmean(z)
    vx = sum((a - mx) ** 2 for a in x)
    vz = sum((c - mz) ** 2 for c in z)
    if vx == 0 or vz == 0:
        return float("nan")
    return sum((a - mx) * (c - mz) for a, c in zip(x, z)) / math.sqrt(vx * vz)


def _spearman(x, z) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    return _pearson(ranks(x), ranks(z))


def shape_tables(rows: List[dict], w) -> None:
    """Write the sample-shape tables for one (already bucketed) row set through `w`."""
    n = len(rows)
    by_b = defaultdict(list)
    for r in rows:
        by_b[r["bucket"]].append(r)
    w("| Bucket | rows | share | median depth_usd | events | rows per event (mean, max) | with news | news share | one-sided | degenerate | zero depth |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in sorted(by_b):
        s = by_b[b]
        ev = Counter(r["event_id"] for r in s)
        st_ = Counter(book_status(r) for r in s)
        w("| %d | %d | %.0f%% | $%.0f | %d | %.2f, %d | %d | %.0f%% | %d | %d | %d |"
          % (b, len(s), 100 * len(s) / n, st.median(r["depth_usd"] for r in s), len(ev), len(s) / len(ev), max(ev.values()),
             sum(r["news_available"] for r in s), 100 * st.fmean(r["news_available"] for r in s),
             st_["one_sided_book"], st_["degenerate_book"], sum(1 for r in s if r["depth_usd"] == 0)))
    ev_all = Counter(r["event_id"] for r in rows)
    w("| all | %d | 100%% | $%.0f | %d | %.2f, %d | %d | %.0f%% | %d | %d | %d |"
      % (n, st.median(r["depth_usd"] for r in rows), len(ev_all), n / len(ev_all), max(ev_all.values()),
         sum(r["news_available"] for r in rows), 100 * st.fmean(r["news_available"] for r in rows),
         sum(1 for r in rows if book_status(r) == "one_sided_book"), sum(1 for r in rows if book_status(r) == "degenerate_book"),
         sum(1 for r in rows if r["depth_usd"] == 0)))
    w("")
    w("Price band (crowd probability q: the mid on A, `price_yes` on B and C) by depth bucket, share of the bucket's rows:")
    w("")
    w("| Bucket | " + " | ".join(BAND_LABELS) + " | median price | median distance from 0.5 | Yes share |")
    w("|---|" + "---|" * (len(BAND_LABELS) + 3))
    for b in sorted(by_b) + ["all"]:
        s = rows if b == "all" else by_b[b]
        bc = Counter(r["band"] for r in s)
        w("| %s | %s | %.2f | %.2f | %.0f%% |" % (b, " | ".join("%.0f%%" % (100 * bc[lab] / len(s)) for lab in BAND_LABELS),
                                                  st.median(r["q"] for r in s), st.median(r["dist"] for r in s),
                                                  100 * st.fmean(r["outcome_yes"] for r in s)))
    w("")


def depth_price_table(sets: Dict[str, List[dict]], w) -> Dict[str, dict]:
    """Correlation between log depth and |price - 0.5|, overall and per bucket, for several row sets."""
    names = list(sets)
    res = {}
    for name, rows in sets.items():
        rows = [r for r in rows if r["depth_usd"] > 0]
        xs = [math.log(r["depth_usd"]) for r in rows]
        ys = [r["dist"] for r in rows]
        d = {"all": (_pearson(xs, ys), _spearman(xs, ys), len(rows))}
        for b in range(1, 6):
            s = [r for r in rows if r["bucket"] == b]
            d[b] = (_pearson([math.log(r["depth_usd"]) for r in s], [r["dist"] for r in s]),
                    _spearman([math.log(r["depth_usd"]) for r in s], [r["dist"] for r in s]), len(s))
        res[name] = d
    w("| Set | " + " | ".join("%s: Pearson r (Spearman) n" % nm for nm in names) + " |")
    w("|---|" + "---|" * len(names))
    for key in ["all"] + list(range(1, 6)):
        cells = []
        for nm in names:
            pr, sr, n = res[nm][key]
            cells.append("%+.3f (%+.3f) %d" % (pr, sr, n))
        w("| %s | %s |" % ("all rows" if key == "all" else "bucket %d" % key, " | ".join(cells)))
    w("")
    return res


def main():
    con = scoring.connect()
    moments = scoring.load_moments(con)
    cuts = scoring.load_or_compute_cuts(moments)
    out = []
    w = out.append
    for win in ("A", "B"):
        for arm in ARMS:
            rows = with_buckets(primary_rows(moments, win, arm), cuts[win])
            w("### Window %s, arm %s (%d rows)" % (win, arm, len(rows)))
            w("")
            shape_tables(rows, w)
    sets = {"Window A": with_buckets(primary_rows(moments, "A"), cuts["A"]), "Window B": with_buckets(primary_rows(moments, "B"), cuts["B"])}
    w("### Depth versus price")
    w("")
    depth_price_table(sets, w)
    print("\n".join(out))
    print("cuts:", cuts)


if __name__ == "__main__":
    main()
