import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
"""thinking-and-length-more-thinking-more-disagreement: inside one model, the rows it thought longest
about are the rows where it strayed furthest from the market price, with no accuracy gain.

Every model in the roster is covered (the list comes from harness/models.json through _roster, so a
model added to the roster appears here without an edit). For each model and window, on the primary analysis
set, rows are split at the quarter points of the number of thinking tokens the model spent, and the top
quarter is compared with the bottom quarter on two things:

  departure from the market   |p - q|, how far the forecast sits from the price
  Brier edge                  (q - y)^2 - (p - y)^2, market score minus model score; negative means the
                              model was less accurate than the market on those rows

Then three checks that the pattern is not simply harder rows: whether thinking length tracks the
horizon (days from forecast to settlement), whether the same split holds inside one horizon band, and
whether thinking length tracks how far the price sits from a coin flip. The share of rows at or above
the 1,000-token thinking budget is printed per model, because a model sitting on its budget cannot
think longer and its quartile split says little.

The quartile rule, the primary set and the band edges are the same for every model.
"""
import sys, statistics as st
if not hasattr(st, "correlation"):   # Python 3.9 has no statistics.correlation (added in 3.10)
    def _pearson(x, y):
        n = len(x); mx = st.fmean(x); my = st.fmean(y)
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
        return sxy / (sxx * syy) ** 0.5
    st.correlation = _pearson
from common_ins import load_moments, load_forecasts as _load_forecasts

def load_forecasts(mode, models=None):
    # common_ins.load_forecasts takes no models= filter, so filter here
    fc = _load_forecasts(mode)
    return [f for f in fc if models is None or f["model"] in models]
import scoring as sc

THINKING_BUDGET = 1000        # the roster's thinking cap, for the share-at-the-cap line only

moments = load_moments()
fc = load_forecasts("normal", models=_roster.SCORED)

rows_by_model = {}
for f in fc:
    m = moments.get(f["row_id"])
    if m is None or not m["primary"] or f["reasoning_tokens"] is None:
        continue
    w = m["window"]
    if w not in ("A", "B", "C"):
        continue
    y = float(m["outcome_yes"])
    p = f["parsed_probability"]
    rows_by_model.setdefault(f["model"], {"A": [], "B": [], "C": []})[w].append(
        {"p": p, "q": m["q"], "y": y, "rt": f["reasoning_tokens"],
         "edge": (m["q"] - y) ** 2 - (p - y) ** 2,
         "horizon_days": m["horizon_days"], "depth_usd": m["depth_usd"], "event_id": m["event_id"]})

MODELS = [m for m in _roster.SCORED if m in rows_by_model]


def band(h):
    if h is None: return None
    if h <= 1: return "<=1d"
    if h <= 3: return "1-3d"
    if h <= 7: return "3-7d"
    if h <= 30: return "7-30d"
    return ">30d"


def split(rs):
    """(bottom quarter, top quarter) by thinking tokens, at the same quarter points for every model."""
    rts = sorted(r["rt"] for r in rs)
    n = len(rts)
    q1, q3 = rts[n // 4], rts[(3 * n) // 4]
    return [r for r in rs if r["rt"] <= q1], [r for r in rs if r["rt"] >= q3]


print("=== Share of rows at or above the %d-token thinking budget (a capped model cannot think longer) ===" % THINKING_BUDGET)
for mdl in MODELS:
    for w, rs in rows_by_model[mdl].items():
        if rs:
            at_cap = sum(1 for r in rs if r["rt"] >= THINKING_BUDGET)
            print("  %-16s window %s: n=%d median thinking tokens %d, share at or above the budget %.1f%%"
                  % (_roster.SHORT[mdl], w, len(rs), st.median(r["rt"] for r in rs), 100.0 * at_cap / len(rs)))

print()
print("=== Base result: top against bottom thinking quartile, departure from the market and Brier edge ===")
for mdl in MODELS:
    for w, rs in rows_by_model[mdl].items():
        if not rs:
            continue
        bottom, top = split(rs)
        bmean = st.fmean(abs(r["p"] - r["q"]) for r in bottom)
        tmean = st.fmean(abs(r["p"] - r["q"]) for r in top)
        bedge = st.fmean(r["edge"] for r in bottom)
        tedge = st.fmean(r["edge"] for r in top)
        print("  %-16s window %s: n=%d bottom(n=%d) |p-q|=%.3f edge=%+.4f | top(n=%d) |p-q|=%.3f edge=%+.4f | diff |p-q|=%+.3f edge=%+.4f"
              % (_roster.SHORT[mdl], w, len(rs), len(bottom), bmean, bedge, len(top), tmean, tedge, tmean - bmean, tedge - bedge))

print()
print("=== Confound: does thinking length track the horizon (days to settlement)? ===")
for mdl in MODELS:
    for w, rs in rows_by_model[mdl].items():
        paired = [(r["rt"], r["horizon_days"]) for r in rs if r["horizon_days"] is not None]
        if len(paired) > 5:
            try:
                corr = st.correlation([p[0] for p in paired], [p[1] for p in paired])
            except Exception:
                corr = float("nan")
            print("  %-16s window %s: corr(thinking tokens, horizon)=%.3f" % (_roster.SHORT[mdl], w, corr))

print()
print("=== Confound: does the same split hold inside one horizon band? ===")
for mdl in MODELS:
    for w, rs in rows_by_model[mdl].items():
        if not rs:
            continue
        hb = {}
        for r in rs:
            hb.setdefault(band(r["horizon_days"]), []).append(r)
        for b, group in sorted(hb.items(), key=lambda x: str(x[0])):
            if len(group) < 40:
                continue
            bottom, top = split(group)
            if len(bottom) < 5 or len(top) < 5:
                continue
            bmean = st.fmean(abs(r["p"] - r["q"]) for r in bottom)
            tmean = st.fmean(abs(r["p"] - r["q"]) for r in top)
            bedge = st.fmean(r["edge"] for r in bottom)
            tedge = st.fmean(r["edge"] for r in top)
            print("  %-16s window %s horizon %-6s n=%d bottom |p-q|=%.3f edge=%+.4f top |p-q|=%.3f edge=%+.4f diff |p-q|=%+.3f edge=%+.4f"
                  % (_roster.SHORT[mdl], w, b, len(group), bmean, bedge, tmean, tedge, tmean - bmean, tedge - bedge))

print()
print("=== Confound: does thinking length track how far the price sits from a coin flip? ===")
for mdl in MODELS:
    for w, rs in rows_by_model[mdl].items():
        if len(rs) > 5:
            corr = st.correlation([r["rt"] for r in rs], [abs(r["q"] - 0.5) for r in rs])
            print("  %-16s window %s: corr(thinking tokens, |q - 0.5|)=%.3f" % (_roster.SHORT[mdl], w, corr))
