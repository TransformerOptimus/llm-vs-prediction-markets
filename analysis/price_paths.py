"""Price paths after the forecast moment (analysis-only data, used by the Experiment B convergence test).

Two tables, never mixed in one series:
  price_paths          the traded/candle price of Yes after the forecast. Polymarket: hourly
                       from the CLOB history (Windows A and B). Kalshi (Window C): hourly
                       candlestick mid to settlement, sparser; points may be missing and one
                       row can have no point after its forecast at all.
  price_paths_archive  Kalshi rows only: the pmxt archive's hourly book (best bid, best ask,
                       top-5 and full depth) from an hour before the forecast to the archive's
                       end. The only source of depth over time. price_yes is NULL on
                       one-sided hours.

Every function returns only points at or after forecast_ts, keyed by row_id, and
tolerates an empty path.
"""
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_price_path(con: sqlite3.Connection, moment: dict) -> List[dict]:
    """Points of `price_paths` at or after the forecast moment, oldest first. Empty list if none."""
    rows = con.execute("SELECT ts, price_yes FROM price_paths WHERE row_id = ? AND ts >= ? ORDER BY ts",
                       (moment["row_id"], moment["forecast_ts"])).fetchall()
    return [{"ts": r["ts"], "price_yes": r["price_yes"]} for r in rows if r["price_yes"] is not None]


def load_depth_path(con: sqlite3.Connection, moment: dict) -> List[dict]:
    """Kalshi only: hourly book from `price_paths_archive` at or after the forecast. Empty on Polymarket rows."""
    if moment["venue"] != "kalshi":
        return []
    rows = con.execute("""SELECT ts, price_yes, best_bid_yes, best_ask_yes, depth_usd, depth_usd_full
                          FROM price_paths_archive WHERE row_id = ? AND ts >= ? ORDER BY ts""",
                       (moment["row_id"], moment["forecast_ts"])).fetchall()
    return [dict(r) for r in rows]


def close_time(moment: dict) -> Optional[str]:
    """When the market stopped trading, for the convergence window. Kalshi: the actual close, kalshi_close_time, analysis only; Kalshi closes most
    markets early and a settlement print of 0.995 or 0.005 after the close would otherwise be
    pulled in. Polymarket: the settlement time, resolved_at, because Polymarket stores no close
    time. Rows with no price-path point inside the window are dropped from the convergence rate
    and counted per window (window_drop_counts)."""
    if moment["venue"] == "kalshi" and moment.get("kalshi_close_time"):
        return moment["kalshi_close_time"]
    return moment.get("resolved_at")


def window_end(moment: dict, hours: float = 72.0) -> datetime:
    """Convergence window end: the earlier of `hours` after the forecast and the market's close."""
    end = _iso(moment["forecast_ts"]) + timedelta(hours=hours)
    c = close_time(moment)
    if c:
        end = min(end, _iso(c))
    return end


def last_price_in_window(path: List[dict], moment: dict, hours: float = 72.0) -> Optional[float]:
    """The last price at or before the convergence window end; None if the path has no such point."""
    end = window_end(moment, hours)
    pts = [p for p in path if _iso(p["ts"]) <= end]
    return pts[-1]["price_yes"] if pts else None


def window_drop_counts(con: sqlite3.Connection, moments: Dict[int, dict], hours: float = 72.0) -> Dict[str, dict]:
    """Per window: rows with no price-path point inside the convergence window (dropped from the convergence rate)."""
    out = {}
    for m in moments.values():
        d = out.setdefault(m["window"], {"rows": 0, "dropped_no_point_in_window": 0})
        d["rows"] += 1
        if last_price_in_window(load_price_path(con, m), m, hours) is None:
            d["dropped_no_point_in_window"] += 1
    return out


def path_coverage(con: sqlite3.Connection, moments: Dict[int, dict]) -> Dict[str, dict]:
    """Per window: rows with any post-forecast point, rows with a point inside the 72-hour window,
    mean points per row. A plumbing check, computed without loading every path into memory."""
    out = {}
    by_win = {}
    for m in moments.values():
        by_win.setdefault(m["window"], []).append(m)
    for w, ms in sorted(by_win.items()):
        n_any = n_in = 0
        pts = 0
        for m in ms:
            path = load_price_path(con, m)
            pts += len(path)
            if path:
                n_any += 1
            if last_price_in_window(path, m) is not None:
                n_in += 1
        out[w] = {"rows": len(ms), "rows_with_post_forecast_point": n_any, "rows_with_point_in_72h_window": n_in,
                  "mean_points_per_row": pts / len(ms)}
    return out
