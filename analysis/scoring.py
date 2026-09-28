"""Scoring core for the benchmark (analysis side, read-only).

For one forecaster and one market-moment (a "row"):

  p             the forecaster's probability of Yes
  q             the market's probability of Yes = price_yes at the forecast moment
  brier_model   (p - outcome)^2, brier_market (q - outcome)^2, lower is better
  brier_edge    brier_market - brier_model; positive = forecaster more accurate
  trade         trading rule: p above best Yes ask -> buy $1 of Yes at the ask;
                p below best Yes bid -> buy $1 of No at (1 - bid); else no trade.
                Return = payout - cost per $1 staked. Fees: the taker fee per share is
                fee_rate x p x (1 - p) from the per-row schedule on both venues; on Kalshi
                the order's fee is rounded up to the next cent; charged on top of the spread.
  direction     with the crowd / against the crowd / no lean, relative to
                the side the market favors (q above or below 0.5)
  band          q in 0.05-0.25, 0.25-0.5, 0.5-0.75, 0.75-0.95
  depth_usd     dollars on the Yes book at the five best prices on each side; the
                ONLY depth measure (depth_usd_full is never used for bucketing).
                Buckets are five quintiles with cut values fixed per window on that
                window's primary set.

Market probability q: on Window A the order-book midpoint mid_yes
(PolyBench's price_yes is a stale copy of one early sweep), falling back to price_yes on
the one-sided rows where no mid exists; on Windows B and C price_yes, which already equals
the mid. The primary-set price filter and the price bands use the same q.

Primary analysis set: is_ladder = 0, news_dateline_after_capture = 0,
end_date_revised = 0, and q in [0.05, 0.95]. Ladder rows are a labeled side set, never in
a headline. Windows are identified by venue and book_source together, never by row_id
ranges. Window A is also reported restricted to live-book rows (two-sided, spread at most
0.10, the B and C sampling rule), with its own cuts under the key "A_live".
Two arms: "news" = rows with news_available = 1, "all" = every row.
Any helper that groups by depth bucket refuses rows from more than one window (one_window).
One-sided and degenerate books (crossed, or spread above 0.50) stay in accuracy numbers
and leave returns numbers.
Intervals: 95% bootstrap resampling EVENTS (the exchange's grouping of markets
about one thing, stored as source_event_id), 2,000 resamples, seed 20260902. Every
bootstrap function takes `cluster`; the default is "event". "market" (resampling single markets) is kept only so
the widening from clustering by event can be reported.

This module never writes to the benchmark. It opens the database read-only.
"""
import json
import math
import os
import random
import sqlite3
import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DB_PATH = os.path.join(REPO, "benchmark", "benchmark.db")
OUT_DIR = os.path.join(HERE, "out")

SEED = 20260902          # event-cluster bootstrap
N_BOOT = 2000            # event-cluster bootstrap
CLUSTER = "event"        # resample events; "market" only for the widening comparison
CLUSTER_KEY = {"event": "event_id", "market": "row_id"}
PRICE_LO, PRICE_HI = 0.05, 0.95   # primary-set price filter
BANDS = [(0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95)]   # price bands
BOTH_SIDES = ("Yes", "No")

# Windows are reported separately. A row's window follows from venue AND book source together.
WINDOWS = {("polymarket", "polybench_snapshot"): "A",
           ("polymarket", "pmxt_archive"): "B",
           ("kalshi", "pmxt_archive"): "C"}
LIVE_BOOK_MAX_SPREAD = 0.10       # the B and C sampling rule; a reporting split on A

# Sports tags (for the sports stratification). Polymarket's event tags name the sport without always
# saying "Sports"; Kalshi and PolyBench use the one word. Case-insensitive, whole-tag match.
SPORTS_TAGS = {
    "sports", "games", "soccer", "football", "nfl", "nba", "wnba", "mlb", "baseball", "nhl", "hockey",
    "ncaa", "ncaab", "ncaaf", "college football", "college basketball", "tennis", "golf", "pga", "pga tour",
    "ufc", "mma", "boxing", "esports", "chess", "cricket", "rugby", "formula 1", "f1", "grand prix",
    "motorsport", "nascar", "cycling", "tour de france", "professional cycling", "olympics", "mls",
    "ucl", "uel", "fifa world cup", "2026 fifa world cup", "world cup", "wc game props", "atp", "wta",
    "athletics", "swimming", "volleyball", "handball", "darts", "snooker", "table tennis", "badminton",
    "horse racing", "lacrosse", "cfl", "xfl", "afl", "nrl", "ipl", "kbo", "npb", "liga mx", "serie a",
    "bundesliga", "la liga", "ligue 1", "premier league", "epl", "champions league", "europa league",
}


def is_sports(moment: dict) -> int:
    """1 if any topic tag names a sport (SPORTS_TAGS, case-insensitive); empty or missing tags are non-sports."""
    tags = moment.get("topic_tags") or []
    return int(any(str(t).strip().lower() in SPORTS_TAGS for t in tags))


def market_prob(moment: dict) -> float:
    """q, the crowd's probability of Yes: the book mid on Window A (price_yes on its one-sided
    rows, where no mid exists); price_yes on B and C, where it already equals the mid."""
    if moment["window"] == "A" and moment.get("mid_yes") is not None:
        return moment["mid_yes"]
    return moment["price_yes"]


def is_live_book(moment: dict) -> int:
    """Two-sided book with spread at most 0.10."""
    return int(not moment["book_one_sided"] and moment.get("spread_yes") is not None
               and moment["spread_yes"] <= LIVE_BOOK_MAX_SPREAD + 1e-9)

# Fee formulas: Polymarket charges takers, per share traded at price p,
# fee_rate x (p x (1 - p)) ^ fee_exponent, from the per-row schedule in moment_outcomes
# (fees_enabled, fee_rate, fee_exponent, fee_taker_only). fee_rate_bps is NOT used.
# Kalshi (Window C): fee_rate already holds 0.07 x the series multiplier; the order's fee is
# rounded up to the next cent (order_fee), and the taker rate is charged on every fill.


def fee_per_share(moment: dict, price: float) -> float:
    """Taker fee in dollars for one share bought at `price`, before any per-order rounding.

    Polymarket and Kalshi share the form fee_rate x (p x (1 - p)) ^ fee_exponent per share
    (Kalshi's fee_rate already holds 0.07 x multiplier). The simulated trader is always a
    taker, so the taker rate is charged even on Kalshi series where makers also pay
    (fee_taker_only = 0).
    """
    if not moment.get("fees_enabled"):
        return 0.0
    rate = moment.get("fee_rate") or 0.0
    expo = moment.get("fee_exponent")
    expo = 1.0 if expo is None else expo
    return rate * (price * (1.0 - price)) ** expo


def round_up_cent(x: float) -> float:
    """Kalshi rounds the fee of each order up to the next cent."""
    return math.ceil(round(x * 100.0, 9)) / 100.0


def order_fee(moment: dict, price: float, shares: float) -> float:
    """Total fee for one order of `shares` at `price`: per-share fee times shares, rounded up to
    the cent on Kalshi, exact on Polymarket."""
    raw = fee_per_share(moment, price) * shares
    return round_up_cent(raw) if moment.get("venue") == "kalshi" else raw


def buy_return(moment: dict, price: float, won: float) -> dict:
    """Buy about one dollar of contracts at `price`; pay the taker fee on top of the spread.

    Shares = 1 / (price + per-share fee), so on Polymarket the outlay is exactly $1. On Kalshi
    the order's fee is then rounded up to the next cent, so the outlay can exceed $1 by under a
    cent; the return is per dollar actually laid out. Payout is $1 per share if `won`.
    `ret_nofee` is the same trade with no fee, kept so the size of the fee correction can be shown.
    """
    per_share = fee_per_share(moment, price)
    shares = 1.0 / (price + per_share)
    fee = order_fee(moment, price, shares)
    outlay = shares * price + fee
    return {"cost": price, "shares": shares, "fee_per_share": per_share, "fee_order": fee, "outlay": outlay,
            "won": won, "ret": (shares * won - outlay) / outlay, "ret_nofee": won / price - 1.0}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

HORIZON_BANDS = [("<1d", 0.0, 1.0), ("1-3d", 1.0, 3.0), ("3-7d", 3.0, 7.0), (">7d", 7.0, float("inf"))]


def _hours_between(a: str, b: str) -> float:
    from datetime import datetime
    fa = datetime.fromisoformat(a.replace("Z", "+00:00"))
    fb = datetime.fromisoformat(b.replace("Z", "+00:00").replace(" ", "T")[:19] + "+00:00") if "+" not in b.replace("Z", "+00:00")[19:] else datetime.fromisoformat(b.replace("Z", "+00:00"))
    return (fb - fa).total_seconds() / 3600.0


def horizon_band(days: Optional[float]) -> Optional[str]:
    """Test 5 (within-horizon) bands on days from forecast to settlement: under 1, 1 to 3, 3 to 7, over 7."""
    if days is None:
        return None
    for name, lo, hi in HORIZON_BANDS:
        if lo <= days < hi:
            return name
    return None


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open the benchmark strictly read-only."""
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    con.row_factory = sqlite3.Row
    return con


def load_moments(con: sqlite3.Connection) -> Dict[int, dict]:
    """Every market-moment joined with its outcome, keyed by row_id."""
    sql = """
      SELECT m.row_id, m.moment_key, m.venue, m.venue_market_id, m.book_source, m.event_title, m.question,
             m.forecast_ts, m.price_yes, m.best_bid_yes, m.best_ask_yes, m.mid_yes, m.spread_yes,
             m.order_book_json, m.depth_usd, m.book_empty, m.book_one_sided, m.news_available,
             m.news_dateline_after_capture, m.is_ladder, m.depth_usd_full, m.end_date_revised,
             m.question_detail, o.scheduled_start_ts, o.in_play_start_before_forecast,
             o.outcome_yes, o.winning_outcome, o.lifetime_volume_usd, o.venue_liquidity,
             o.topic_tags, o.split, o.source_event_id,
             o.fees_enabled, o.fee_rate, o.fee_exponent, o.fee_taker_only, o.fee_source,
             o.kalshi_close_time, o.resolved_at
      FROM market_moments m JOIN moment_outcomes o USING (row_id)
    """
    out = {}
    for r in con.execute(sql):
        d = dict(r)
        d["book"] = json.loads(d.pop("order_book_json") or "{}")
        d["topic_tags"] = json.loads(d["topic_tags"] or "[]")
        d["window"] = WINDOWS.get((d["venue"], d["book_source"]), "?")
        d["q"] = market_prob(d)
        d["in_price_range"] = int(PRICE_LO <= d["q"] <= PRICE_HI)
        d["primary"] = int(d["in_price_range"] and not d["news_dateline_after_capture"] and not d["is_ladder"]
                           and not (d.get("end_date_revised") or 0))
        d["sports"] = is_sports(d)
        d["live_book"] = is_live_book(d)
        d["horizon_days"] = _hours_between(d["forecast_ts"], d["resolved_at"]) / 24.0 if d.get("resolved_at") else None
        d["horizon_band"] = horizon_band(d["horizon_days"])
        # Cluster for the bootstrap. A row with no event id is its own cluster.
        d["event_id"] = d["source_event_id"] if d["source_event_id"] is not None else "row:%d" % d["row_id"]
        out[d["row_id"]] = d
    return out


def load_quarantine(con: sqlite3.Connection, moments: Optional[Dict[int, dict]] = None) -> Dict[Tuple[str, str], int]:
    """(moment_key, model_name) -> quarantined flag (memory-probe quarantine). Empty until the probe runs.

    Keyed on moment_key so the flags stay valid if row ids change. A flag row without a moment_key is
    resolved through row_id when `moments` is given; otherwise it is skipped and counted
    in the returned dict under the key ("_unresolved", "")."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(quarantine_flags)")}
    has_key = "moment_key" in cols
    sql = "SELECT row_id, model_name, quarantined%s FROM quarantine_flags" % (", moment_key" if has_key else "")
    out: Dict[Tuple[str, str], int] = {}
    unresolved = 0
    for r in con.execute(sql):
        key = r["moment_key"] if has_key else None
        if not key and moments and r["row_id"] in moments:
            key = moments[r["row_id"]]["moment_key"]
        if not key:
            unresolved += 1
            continue
        out[(key, r["model_name"])] = r["quarantined"]
    if unresolved:
        out[("_unresolved", "")] = unresolved
    return out


# ----------------------------------------------------------------------------
# Depth buckets
# ----------------------------------------------------------------------------

def one_window(rows: List[dict], what: str) -> Optional[str]:
    """Raise if `rows` span more than one window (bucket numbers mean different dollars per window).
    Returns the window, or None for an empty list."""
    wins = {r.get("window") for r in rows}
    if len(wins) > 1:
        raise ValueError("%s: rows span windows %s; bucket comparisons must stay inside one window" % (what, sorted(wins)))
    return next(iter(wins)) if wins else None


def quintile_cuts(values: Sequence[float], k: int = 5) -> List[float]:
    """Cut values splitting sorted values into k equal groups: the k-1 upper edges of groups 1..k-1."""
    vs = sorted(values)
    return [vs[int(len(vs) * i / k) - 1] for i in range(1, k)]


def assign_bucket(v: float, cuts: Sequence[float]) -> int:
    """Bucket 1 (thinnest) .. len(cuts)+1 (deepest); a value equal to a cut goes below it."""
    for i, c in enumerate(cuts):
        if v <= c:
            return i + 1
    return len(cuts) + 1


ARMS = ("news", "all")


def in_arm(moment_or_row: dict, arm: str) -> bool:
    """Two-arm reporting: 'news' keeps rows with news_available = 1; 'all' keeps every row."""
    if arm == "all":
        return True
    if arm == "news":
        return bool(moment_or_row["news_available"])
    raise ValueError(arm)


def window_cuts(moments: Dict[int, dict]) -> Dict[str, List[float]]:
    """Quintile cuts of depth_usd per window, on that window's primary set, one value per row.
    "A_live" is Window A restricted to live-book rows, with its own cuts."""
    per = defaultdict(list)
    for m in moments.values():
        if m["primary"]:
            per[m["window"]].append(m["depth_usd"])
            if m["window"] == "A" and m["live_book"]:
                per["A_live"].append(m["depth_usd"])
    return {w: quintile_cuts(v) for w, v in sorted(per.items())}


def load_or_compute_cuts(moments: Dict[int, dict], path: str = os.path.join(OUT_DIR, "depth_cuts.json")) -> Dict[str, List[float]]:
    """The cuts are computed once. If a saved file exists it wins; otherwise compute and save."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    cuts = window_cuts(moments)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cuts, f, indent=1)
    return cuts


def bucket_of(moment: dict, cuts: Dict[str, List[float]], cut_key: Optional[str] = None) -> Optional[int]:
    """Bucket under the window's cuts, or under `cut_key` (for example "A_live")."""
    c = cuts.get(cut_key or moment["window"])
    return assign_bucket(moment["depth_usd"], c) if c else None


def band_of(q: float) -> Optional[str]:
    """Price bands on the market probability; the shared edges go to the lower band."""
    for lo, hi in BANDS:
        if lo <= q <= hi:
            return "%.2f-%.2f" % (lo, hi)
    return None


# ----------------------------------------------------------------------------
# Scoring one forecast
# ----------------------------------------------------------------------------

def direction(p: float, q: float) -> str:
    if q == 0.5 or p == q:
        return "no lean"
    if q > 0.5:
        return "with" if p > q else "against"
    return "with" if p < q else "against"


def book_status(moment: dict) -> str:
    """'ok', 'one_sided_book', or 'degenerate_book' (crossed, or spread above 0.50)."""
    ask, bid = moment["best_ask_yes"], moment["best_bid_yes"]
    if moment["book_one_sided"] or ask is None or bid is None:
        return "one_sided_book"
    if ask <= bid or ask - bid > 0.5:
        return "degenerate_book"
    return "ok"


def trade(p: float, moment: dict) -> dict:
    """The trading rule. Returns side, cost per contract, return per $1 staked; side None = no trade.

    One-sided and degenerate books are never traded. The taker fee is charged
    per share on top of the spread.
    """
    status = book_status(moment)
    if status != "ok":
        return {"side": None, "reason": status}
    ask, bid = moment["best_ask_yes"], moment["best_bid_yes"]
    if p > ask:
        side, cost = "Yes", ask
    elif p < bid:
        side, cost = "No", 1.0 - bid
    else:
        return {"side": None, "reason": "inside_spread"}
    won = 1.0 if moment["winning_outcome"] == side else 0.0
    out = buy_return(moment, cost, won)
    out.update({"side": side, "reason": "traded"})
    return out


def score_forecast(moment: dict, p: float, forecaster: str, cuts: Dict[str, List[float]],
                   extra: Optional[dict] = None, cut_key: Optional[str] = None) -> dict:
    """One scored row under the contract. q is the market probability from market_prob()."""
    q = moment["q"]
    y = float(moment["outcome_yes"])
    t = trade(p, moment)
    row = {
        "row_id": moment["row_id"], "moment_key": moment["moment_key"], "event_id": moment["event_id"],
        "venue": moment["venue"], "window": moment["window"],
        "venue_market_id": moment["venue_market_id"], "forecaster": forecaster,
        "p": p, "q": q, "outcome_yes": y, "error": p - y,
        "brier_model": (p - y) ** 2, "brier_market": (q - y) ** 2,
        "brier_edge": (q - y) ** 2 - (p - y) ** 2,
        "q_stored": moment["price_yes"], "brier_market_stored": (moment["price_yes"] - y) ** 2,
        "trade_side": t["side"], "trade_reason": t["reason"],
        "ret": t.get("ret"), "ret_nofee": t.get("ret_nofee"), "fee_per_share": t.get("fee_per_share"),
        "fee_order": t.get("fee_order"),
        "fees_enabled": moment.get("fees_enabled") or 0, "trade_cost": t.get("cost"), "trade_won": t.get("won"),
        "direction": direction(p, q), "band": band_of(q),
        "depth_usd": moment["depth_usd"],
        "log_depth": math.log(moment["depth_usd"]) if moment["depth_usd"] > 0 else None,
        "bucket": bucket_of(moment, cuts, cut_key),
        "live_book": moment["live_book"], "horizon_days": moment["horizon_days"], "horizon_band": moment["horizon_band"],
        "end_date_revised": moment.get("end_date_revised") or 0, "in_play": moment.get("in_play_start_before_forecast") or 0,
        "primary": moment["primary"], "is_ladder": moment["is_ladder"], "news_available": moment["news_available"],
        "in_price_range": moment["in_price_range"], "dateline_flag": moment["news_dateline_after_capture"],
        "book_one_sided": moment["book_one_sided"],
        "sports": moment["sports"], "split": moment["split"],
    }
    if extra:
        row.update(extra)
    return row


# ----------------------------------------------------------------------------
# Bootstrap over clusters (events by default)
# ----------------------------------------------------------------------------

def _groups(rows: List[dict], key: str, cluster: str = CLUSTER):
    ck = CLUSTER_KEY[cluster]
    g = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            g[r[ck]].append(r[key])
    return g


def boot_mean(rows: List[dict], key: str, n: int = N_BOOT, seed: int = SEED, cluster: str = CLUSTER) -> Tuple[float, float, float]:
    """(mean, lo, hi): the mean of `key` over rows with a 95% interval resampling clusters (events by default)."""
    g = _groups(rows, key, cluster)
    ids = list(g)
    if not ids:
        return (float("nan"),) * 3
    vals = [v for m in ids for v in g[m]]
    mean = st.fmean(vals)
    if len(ids) < 2 or n <= 0:
        return (mean, float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        tot = cnt = 0.0
        for m in rng.choices(ids, k=len(ids)):
            tot += sum(g[m])
            cnt += len(g[m])
        means.append(tot / cnt)
    means.sort()
    return mean, means[int(0.025 * n)], means[int(0.975 * n)]


def boot_diff(rows_a: List[dict], rows_b: List[dict], key: str, n: int = N_BOOT, seed: int = SEED, cluster: str = CLUSTER):
    """Mean(a) - mean(b) with a cluster-bootstrap interval; the two sets are resampled independently."""
    ga, gb = _groups(rows_a, key, cluster), _groups(rows_b, key, cluster)
    ia, ib = list(ga), list(gb)
    if not ia or not ib:
        return (float("nan"),) * 3
    point = st.fmean(v for m in ia for v in ga[m]) - st.fmean(v for m in ib for v in gb[m])
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sa = [v for m in rng.choices(ia, k=len(ia)) for v in ga[m]]
        sb = [v for m in rng.choices(ib, k=len(ib)) for v in gb[m]]
        diffs.append(st.fmean(sa) - st.fmean(sb))
    diffs.sort()
    return point, diffs[int(0.025 * n)], diffs[int(0.975 * n)]


# ----------------------------------------------------------------------------
# Test 4 (within-band check): tests 2 and 3 inside each price band, averaged with equal weights
# ----------------------------------------------------------------------------

def _band_groups(rows: List[dict], cluster: str):
    """cluster -> list of rows, for a joint resample of everything at once."""
    ck = CLUSTER_KEY[cluster]
    g = defaultdict(list)
    for r in rows:
        g[r[ck]].append(r)
    return g


BAND_NAMES = ["%.2f-%.2f" % b for b in BANDS]


HORIZON_BAND_NAMES = [name for name, _, _ in HORIZON_BANDS]
BAND_KEYS = {"band": BAND_NAMES, "horizon_band": HORIZON_BAND_NAMES}   # test 4 (price) and test 5 (horizon)


def band_bucket_counts(rows: List[dict], key: str, lo_bucket: int = 1, hi_bucket: int = 5,
                       band_key: str = "band") -> Dict[str, Tuple[int, int]]:
    """Per band: rows in bucket 1 and in bucket 5 that carry `key` (the counts the 30-row minimum looks at).
    band_key is "band" (price bands, test 4) or "horizon_band" (test 5, the same rule)."""
    one_window(rows, "band_bucket_counts")
    out = {}
    for band in BAND_KEYS[band_key]:
        a = sum(1 for r in rows if r[band_key] == band and r["bucket"] == lo_bucket and r.get(key) is not None)
        c = sum(1 for r in rows if r[band_key] == band and r["bucket"] == hi_bucket and r.get(key) is not None)
        out[band] = (a, c)
    return out


def voting_bands(rows: List[dict], key: str, min_rows: int, band_key: str = "band") -> List[str]:
    """Test 4: a band votes only if the smaller of its bucket-1 and bucket-5 counts is at least min_rows."""
    return [b for b, (a, c) in band_bucket_counts(rows, key, band_key=band_key).items() if min(a, c) >= min_rows]


def _within_band_diff_stat(rows: List[dict], key: str, bands: List[str], lo_bucket: int = 1, hi_bucket: int = 5,
                           band_key: str = "band"):
    """Equal-weight average over the voting bands of mean(bucket lo) - mean(bucket hi). All bands are reported."""
    per_band = {}
    for band in BAND_KEYS[band_key]:
        a = [r[key] for r in rows if r[band_key] == band and r["bucket"] == lo_bucket and r.get(key) is not None]
        c = [r[key] for r in rows if r[band_key] == band and r["bucket"] == hi_bucket and r.get(key) is not None]
        per_band[band] = (st.fmean(a) - st.fmean(c), len(a), len(c)) if a and c else None
    vals = [per_band[b][0] for b in bands if per_band[b] is not None]
    return (st.fmean(vals) if vals else float("nan")), per_band


def _within_band_slope_stat(rows: List[dict], ykey: str, bands: List[str], xkey: str = "log_depth", band_key: str = "band"):
    per_band = {}
    for band in BAND_KEYS[band_key]:
        s = [r for r in rows if r[band_key] == band and r.get(ykey) is not None and r.get(xkey) is not None]
        per_band[band] = (_ols_slope([r[xkey] for r in s], [r[ykey] for r in s]), len(s)) if len(s) >= 3 else None
    vals = [per_band[b][0] for b in bands if per_band[b] is not None and per_band[b][0] == per_band[b][0]]
    return (st.fmean(vals) if vals else float("nan")), per_band


def boot_within_band(rows: List[dict], key: str, kind: str, n: int = N_BOOT, seed: int = SEED,
                     cluster: str = CLUSTER, min_rows: int = 30, band_key: str = "band") -> dict:
    """Test 4 (band_key "band", price bands) and test 5 (band_key "horizon_band", the same rule).
    kind = 'diff' (test 2 inside bands) or 'slope' (test 3 inside bands).

    The point estimate is the equal-weight average, over the voting bands, of the band's own
    statistic. A band votes only if the smaller of its bucket-1 and bucket-5 counts on the full
    sample is at least `min_rows` (30 in the reported tables; 0 means every band votes). The set of
    voting bands is fixed on the full sample and reused in every bootstrap draw. The interval
    resamples clusters (events) jointly, recomputing every voting band from the same draw; a
    band that is empty in a draw is left out of that draw's average.
    """
    one_window(rows, "boot_within_band")
    bands = voting_bands(rows, key, min_rows, band_key)
    stat = ((lambda rs: _within_band_diff_stat(rs, key, bands, band_key=band_key)) if kind == "diff"
            else (lambda rs: _within_band_slope_stat(rs, key, bands, band_key=band_key)))
    point, per_band = stat(rows)
    g = _band_groups(rows, cluster)
    ids = list(g)
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        draw = [r for c in rng.choices(ids, k=len(ids)) for r in g[c]]
        v, _ = stat(draw)
        if v == v:
            boots.append(v)
    boots.sort()
    return {"value": point, "lo": boots[int(0.025 * len(boots))] if boots else float("nan"),
            "hi": boots[int(0.975 * len(boots))] if boots else float("nan"), "per_band": per_band,
            "voting_bands": bands, "min_rows": min_rows, "n_clusters": len(ids), "n_draws_used": len(boots)}


def _ols_slope(xs: List[float], ys: List[float]) -> float:
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def boot_slope(rows: List[dict], ykey: str, xkey: str = "log_depth", n: int = N_BOOT, seed: int = SEED,
               cluster: str = CLUSTER) -> dict:
    """Test 3: least-squares slope of `ykey` on `xkey` across rows, interval resampling clusters.

    Rows without a usable x (depth 0, so log undefined) are dropped and counted.
    """
    if xkey == "log_depth":
        one_window(rows, "boot_slope on log_depth")
    ck = CLUSTER_KEY[cluster]
    g = defaultdict(list)
    dropped = 0
    for r in rows:
        if r.get(ykey) is None:
            continue
        if r.get(xkey) is None:
            dropped += 1
            continue
        g[r[ck]].append((r[xkey], r[ykey]))
    ids = list(g)
    pts = [p for m in ids for p in g[m]]
    if len(ids) < 3:
        return {"slope": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(pts), "n_markets": 0,
                "n_clusters": len(ids), "dropped_zero_depth": dropped}
    slope = _ols_slope([x for x, _ in pts], [y for _, y in pts])
    rng = random.Random(seed)
    ss = []
    for _ in range(n):
        s = [p for m in rng.choices(ids, k=len(ids)) for p in g[m]]
        ss.append(_ols_slope([x for x, _ in s], [y for _, y in s]))
    ss.sort()
    return {"slope": slope, "lo": ss[int(0.025 * n)], "hi": ss[int(0.975 * n)], "n": len(pts),
            "n_markets": len({r["row_id"] for r in rows if r.get(ykey) is not None and r.get(xkey) is not None}),
            "n_clusters": len(ids), "dropped_zero_depth": dropped}


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------

def summarize(rows: List[dict], n_boot: int = N_BOOT, cluster: str = CLUSTER) -> dict:
    """Accuracy over all rows; returns over traded rows only. One group of rows in, one dict out."""
    if not rows:
        return {}
    traded = [r for r in rows if r["ret"] is not None]
    acc = [r for r in rows if r["brier_model"] is not None]      # no-skill rows carry no accuracy
    be, be_lo, be_hi = boot_mean(acc, "brier_edge", n_boot, cluster=cluster) if acc else (float("nan"),) * 3
    rt, rt_lo, rt_hi = boot_mean(traded, "ret", n_boot, cluster=cluster) if traded else (float("nan"),) * 3
    return {
        "n": len(rows), "n_markets": len({r["row_id"] for r in rows}),
        "n_events": len({r["event_id"] for r in rows}),
        "n_traded": len(traded), "n_no_trade": len(rows) - len(traded),
        "n_one_sided": sum(1 for r in rows if r["trade_reason"] == "one_sided_book"),
        "n_degenerate": sum(1 for r in rows if r["trade_reason"] == "degenerate_book"),
        "median_depth_usd": st.median(r["depth_usd"] for r in rows),
        "brier_model": st.fmean(r["brier_model"] for r in acc) if acc else float("nan"),
        "brier_market": st.fmean(r["brier_market"] for r in rows),
        "brier_edge": be, "be_lo": be_lo, "be_hi": be_hi,
        "ret": rt, "ret_lo": rt_lo, "ret_hi": rt_hi,
        "ret_nofee": st.fmean(r["ret_nofee"] for r in traded) if traded else float("nan"),
        "n_fee_charged": sum(1 for r in traded if r.get("fee_per_share")),
        "mean_fee_per_share": st.fmean(r["fee_per_share"] for r in traded) if traded else float("nan"),
        "win_rate": st.fmean(r["trade_won"] for r in traded) if traded else float("nan"),
        "mean_p": st.fmean(r["p"] for r in rows), "mean_q": st.fmean(r["q"] for r in rows),
    }


def summarize_by(rows: List[dict], key: str, n_boot: int = N_BOOT, order: Optional[list] = None,
                 cluster: str = CLUSTER) -> Dict:
    if key == "bucket":
        one_window(rows, "summarize_by bucket")
    groups = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            groups[r[key]].append(r)
    keys = order if order is not None else sorted(groups)
    return {k: summarize(groups[k], n_boot, cluster) for k in keys if k in groups}


def pct(x: float) -> str:
    return "nan" if x != x else "%+.1f%%" % (100 * x)


def write_csv(path: str, rows: List[dict], cols: Optional[List[str]] = None) -> None:
    import csv
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
