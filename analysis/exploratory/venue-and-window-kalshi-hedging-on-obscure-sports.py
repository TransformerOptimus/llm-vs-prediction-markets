"""Hypothesis: on Kalshi (Window C) the models sit closer to 0.5 than the market, and this is
concentrated on obscure sports sub-markets (low-tier tennis, esports maps) that Polymarket
windows do not carry; on sports both venues carry (golf, UFC/MMA, motorsport) and on Kalshi rows
with news and a horizon of a day or more, the gap to the market's confidence is about zero,
as on Polymarket B. On Window A the models are MORE extreme than the market, mostly on wide books.
Compression = |p - 0.5| - |q - 0.5| (negative: model less confident than the market).
Read-only; prints every number; saves the figure."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import os, sys, re, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vw_base

M, cuts, rows = vw_base.load()
for r in rows:
    r["comp"] = abs(r["p"] - 0.5) - abs(r["q"] - 0.5); r["absq"] = abs(r["q"] - 0.5); r["absp"] = abs(r["p"] - 0.5)
def title(r):
    m = M[r["row_id"]]; return ((m["event_title"] or "") + " || " + (m["question"] or "")).lower()
def kw(r, ks): t = title(r); return any(k in t for k in ks)
strata = [
    ("sports, all", lambda r: r["sports"]),
    ("sports, news and >= 1 day", lambda r: r["sports"] and r["news_available"] and r["horizon_band"] != "<1d"),
    ("sports, no news or same-day", lambda r: r["sports"] and not (r["news_available"] and r["horizon_band"] != "<1d")),
    ("tennis (title match)", lambda r: kw(r, ["tennis", "atp", "wta", " round of ", " set 1 ", "set 2"])),
    ("esports (title match)", lambda r: kw(r, ["esports", "cs2", "dota", "valorant", " lol ", "league of legends", "map "])),
    ("golf (title match)", lambda r: kw(r, ["pga", "golf", "open: top"])),
    ("ufc/mma (title match)", lambda r: kw(r, ["ufc", "mma"])),
    ("motorsport (title match)", lambda r: kw(r, ["nascar", "f1", "grand prix", "indianapolis"])),
    ("A/B: 'will X win on <date>'", lambda r: r["sports"] and re.search(r"will .* win on 20", title(r)) is not None),
    ("A/B: both teams to score", lambda r: "both teams to score" in title(r)),
    ("A/B: end in a draw", lambda r: "end in a draw" in title(r)),
    ("A only: spread > 0.20", lambda r: r["window"] == "A" and r["sports"] and r["spread_yes"] is not None and r["spread_yes"] > 0.20),
    ("A only: live book (spread <= 0.10)", lambda r: r["window"] == "A" and r["sports"] and r["live_book"] == 1),
]
res = {}
print("compression |p-.5|-|q-.5|, pooled over models (one obs per model-row), event bootstrap %d draws seed %d; also mean |q-.5| and |p-.5|" % (vw_base.N_BOOT, vw_base.SEED))
for lab, f in strata:
    line = "%-34s" % lab
    for w in "ABC":
        s = [r for r in rows if r["window"] == w and f(r)]
        if len(s) >= 200:
            a = vw_base.bm(s, "comp"); res[(lab, w)] = (a, len(s))
            line += " %s n=%5d %+.4f [%+.4f,%+.4f] |q-.5|=%.3f |p-.5|=%.3f |" % (w, len(s), *a, st.fmean(r["absq"] for r in s), st.fmean(r["absp"] for r in s))
        else:
            line += " %s n=%5d (under 200, not reported) |" % (w, len(s))
    print(line)
print("== per model, sports rows, per window")
for mdl in vw_base.MODELS:
    line = "%-16s" % vw_base.short(mdl)
    for w in "ABC":
        s = [r for r in rows if r["window"] == w and r["sports"] and r["model"] == mdl]
        if s: line += " %s n=%4d %+.4f [%+.4f,%+.4f] |" % (w, len(s), *vw_base.bm(s, "comp"))
    print(line)
print("== per model, Kalshi sports rows with news and >= 1 day")
for mdl in vw_base.MODELS:
    s = [r for r in rows if r["window"] == "C" and r["sports"] and r["news_available"] and r["horizon_band"] != "<1d" and r["model"] == mdl]
    print("%-16s n=%4d %+.4f [%+.4f,%+.4f]" % (vw_base.short(mdl), len(s), *vw_base.bm(s, "comp")))

if _paths.want_figure():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # figure
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labs = [l for l, _ in strata]; cols = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    for wi, w in enumerate("ABC"):
        ys, xs, lo, hi = [], [], [], []
        for i, lab in enumerate(labs):
            if (lab, w) in res:
                (a, n) = res[(lab, w)]; ys.append(i + (wi - 1) * 0.27); xs.append(a[0]); lo.append(a[0] - a[1]); hi.append(a[2] - a[0])
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", color=cols[w], ecolor=cols[w], elinewidth=2, markersize=6, label="Window %s (%s)" % (w, {"A": "Polymarket, Feb 2026", "B": "Polymarket, Jul-Aug 2026", "C": "Kalshi, May-Jun 2026"}[w]))
    ax.axvline(0, color="#999", lw=1); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("|p - 0.5| - |q - 0.5|, pooled over models (negative = model less sure than the market)", fontsize=9, color="#52514e")
    ax.grid(axis="x", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Kalshi pull toward 0.5 sits on tennis and esports sub-markets;\nshared sports and news rows match Polymarket B", fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(_paths.FIG_DIR, "venue-and-window-kalshi-hedging-on-obscure-sports.png")
    fig.savefig(out, dpi=150); print("figure", out)
