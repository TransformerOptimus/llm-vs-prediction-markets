"""Read-only access to the benchmark, restricted to the one model-visible table.

Why this file exists: the harness must never see outcomes, price paths, reference
forecasts, or quarantine flags. Instead of trusting every query to be careful, this
opens the database read-only and installs a SQLite "authorizer", a hook SQLite calls
for every table and column a statement wants to touch. Anything except a read of
`market_moments` is refused, so a wrong query fails loudly instead of leaking.

Every table SQLite asked about is recorded in `tables_touched`, so the leak check can
prove afterwards that only `market_moments` was read.
"""
import os
import sqlite3
from typing import Dict, Iterable, Iterator, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(os.path.dirname(HERE), "benchmark", "benchmark.db")

ALLOWED_TABLES = {"market_moments"}

# Every column of market_moments, in schema order.
VISIBLE_COLUMNS = (
    "row_id", "venue", "venue_market_id", "event_title", "question", "description",
    "market_end_date", "forecast_ts", "news_captured_at", "price_yes", "best_bid_yes",
    "best_ask_yes", "mid_yes", "spread_yes", "order_book_json", "depth_usd", "book_empty",
    "book_one_sided", "news_available", "news_article_count", "news_text",
    # the benchmark's own dateline exclusion flag
    "news_dateline_after_capture", "news_latest_dateline",
    # where the order book came from
    # (polybench_snapshot or pmxt_archive); logged in every record, never shown to the model
    "book_source",
    # row metadata, logged, never shown to the model
    "depth_usd_full", "is_ladder", "moment_key",
    # Kalshi contract subtitle, shown as the "Contract:" line
    "question_detail",
)

# A window is identified by venue AND book_source together, never by one alone.
# Kalshi row ids start at 1,000,000.
WINDOWS = {
    "A": ("polymarket", "polybench_snapshot"),
    "B": ("polymarket", "pmxt_archive"),
    "C": ("kalshi", "pmxt_archive"),
}


def window_of(row: Dict) -> Optional[str]:
    for w, (venue, src) in WINDOWS.items():
        if row.get("venue") == venue and row.get("book_source") == src:
            return w
    return None


def _denied(e: Exception) -> bool:
    m = str(e)
    return "not authorized" in m or "prohibited" in m


class NotAllowed(Exception):
    """Raised when a statement tries to touch anything other than market_moments."""


class BenchmarkReader:
    def __init__(self, db_path: str = DEFAULT_DB):
        if not os.path.exists(db_path):
            raise FileNotFoundError(db_path)
        self.db_path = db_path
        self.tables_touched = set()
        self.denied_attempts: List[str] = []
        # mode=ro: the file is opened read-only at the SQLite level as well.
        self.con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
        self.con.set_authorizer(self._authorize)

    # SQLite calls this for each action a statement wants to perform, before it runs.
    def _authorize(self, action, arg1, arg2, db_name, trigger):
        if action == sqlite3.SQLITE_SELECT or action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = arg1
            self.tables_touched.add(table)
            if table in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK
            self.denied_attempts.append("read %s.%s" % (table, arg2))
            return sqlite3.SQLITE_DENY
        # Anything else (insert, update, delete, create, drop, attach, pragma, ...)
        self.denied_attempts.append("action %s %s" % (action, arg1))
        return sqlite3.SQLITE_DENY

    def _select(self, where: str = "", params: Iterable = ()) -> sqlite3.Cursor:
        sql = "SELECT %s FROM market_moments %s" % (", ".join(VISIBLE_COLUMNS), where)
        try:
            return self.con.execute(sql, tuple(params))
        except sqlite3.DatabaseError as e:
            if _denied(e):
                raise NotAllowed(str(e))
            raise

    def count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM market_moments").fetchone()[0]

    def fetch_row(self, row_id: int) -> Dict:
        r = self._select("WHERE row_id = ?", (row_id,)).fetchone()
        if r is None:
            raise KeyError("no market_moments row with row_id %s" % row_id)
        return dict(zip(VISIBLE_COLUMNS, r))

    def fetch_by_key(self, moment_key: str) -> Dict:
        """Look a row up by moment_key (venue|venue_market_id|forecast_ts), which
        stays the same if the rows are renumbered; row_id does not."""
        r = self._select("WHERE moment_key = ?", (moment_key,)).fetchone()
        if r is None:
            raise KeyError("no market_moments row with moment_key %s" % moment_key)
        return dict(zip(VISIBLE_COLUMNS, r))

    def iter_rows(self, row_ids: Optional[List] = None, limit: Optional[int] = None,
                  window: Optional[str] = None, news_only: bool = False) -> Iterator[Dict]:
        """Rows in row_id order. `row_ids` may mix integer row ids and moment keys
        (missing ones raise). `window` is A, B or C and selects on venue AND
        book_source together. `news_only` keeps rows with news_available = 1 (the
        memory probe runs on those only; on no-news rows the two prompts are the same)."""
        if row_ids is not None:
            for rid in row_ids:
                row = self.fetch_by_key(rid) if isinstance(rid, str) and "|" in rid else self.fetch_row(int(rid))
                if window is not None and window_of(row) != window:
                    raise ValueError("row %s (%s) is not in window %s" % (row["row_id"], row["moment_key"], window))
                if news_only and not row["news_available"]:
                    continue
                yield row
            return
        conds, params = [], []
        if window is not None:
            venue, src = WINDOWS[window]
            conds, params = ["venue = ?", "book_source = ?"], [venue, src]
        if news_only:
            conds.append("news_available = 1")
        where = ("WHERE " + " AND ".join(conds) + " " if conds else "") + "ORDER BY row_id"
        if limit is not None:
            where += " LIMIT %d" % int(limit)
        for r in self._select(where, params):
            yield dict(zip(VISIBLE_COLUMNS, r))

    def guard_self_test(self) -> str:
        """Prove the guard works: a read of an analysis-only table must be refused."""
        touched_before, denied_before = set(self.tables_touched), list(self.denied_attempts)
        for table in ("moment_outcomes", "price_paths", "reference_forecasts", "quarantine_flags",
                      "news_articles", "price_paths_archive", "sample_log"):
            try:
                self.con.execute("SELECT * FROM %s LIMIT 1" % table).fetchone()
            except sqlite3.DatabaseError as e:
                if not _denied(e):
                    raise
                continue
            raise NotAllowed("guard failed: was able to read %s" % table)
        for sql in ("SELECT * FROM market_moments m JOIN moment_outcomes o USING (row_id) LIMIT 1",
                    "SELECT kalshi_close_time FROM moment_outcomes LIMIT 1",
                    "SELECT m.row_id FROM market_moments m WHERE EXISTS "
                    "(SELECT 1 FROM price_paths_archive a WHERE a.row_id = m.row_id) LIMIT 1"):
            try:
                self.con.execute(sql).fetchone()
            except sqlite3.DatabaseError as e:
                if not _denied(e):
                    raise
            else:
                raise NotAllowed("guard failed: allowed: %s" % sql)
        # The self-test deliberately touched forbidden tables; restore the record so
        # the after-run check only reflects real reads.
        self.tables_touched, self.denied_attempts = touched_before, denied_before
        return "guard ok: analysis-only tables are refused; database opened read-only"

    def close(self):
        self.con.close()
