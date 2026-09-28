"""Leak checks, run before every run and on every row.

Before the run:
  0. Rows the benchmark itself flags (news_dateline_after_capture = 1) are excluded
     up front and logged as excluded_by_benchmark_flag; no model call. The checks
     below are the harness's own second line of defense.
  1. The reader's guard refuses every analysis-only table (proved by trying).
  2. Every row that will be run has news_captured_at <= forecast_ts.

Per row, on the exact prompt text that will be sent:
  3. The "today" line equals the row's forecast_ts.
  4. Future timestamps (a date with a clock time later than forecast_ts) are logged
     as future_timestamp_mentions, not failed: they occur in fixture lists inside
     news ("Metz vs Lille (2026-02-06T19:45:00.000Z)" in an article written before
     the match) and in exchange text written at market creation. Only a dateline
     is evidence of a late fetch, which is rule 5.
  5. No dateline in the news block (news_text only, never the resolution rules) is two
     or more calendar days after forecast_ts. Mirrors the benchmark's own dateline
     scanner: (a) a line with a dateline word ("published", "updated",
     "posted", "modified", "last updated", "last modified") and a date; (b) when that
     line has no date, a BARE DATE LINE among the next two lines (a line of at most 60
     characters that is nothing but one date, optionally with a weekday, a clock time
     and a time-zone word); (c) any bare date line within an article's first 25 lines,
     even without a dateline word. A hit blocks the row, except for rows on the
     benchmark's labelled-forward list (--forward-rows-file in run.py: fixture tables,
     schedule pages, wrong-year parses), which are logged as
     scanner_hit_labelled_forward and sent. The benchmark flag remains the only
     exclusion. A dateline exactly one day ahead is
     allowed because the capture time is UTC and the site may print a local date
     (16:02 UTC on Feb 6 is already Feb 7 in Tokyo); it is recorded as a soft
     finding (dateline_next_day). Rows that fail are never sent to the model; they
     are written to the log with status leak_check_failed and the run continues,
     unless --abort-on-leak is given.
  6. Plain calendar dates later than forecast_ts (for example "the Super Bowl on
     Feb 8, 2026" in an article written on Feb 6) are forward references, not leaks.
     They are counted and written to the log as future_date_mentions, not failed.

After the run:
  7. The set of tables SQLite touched is exactly {"market_moments"}.
"""
import json
import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Set

from benchmark_reader import BenchmarkReader
from prompt import build_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FORWARD_ROWS_FILE = os.path.join(os.path.dirname(HERE), "benchmark", "dateline_forward_rows.json")
_forward_rows: Optional[Set[int]] = None
_forward_rows_source: str = "not loaded"


def load_forward_rows(path: Optional[str] = None) -> Set[int]:
    """Row ids labelled "forward" when the benchmark was built, after hand-checking the date
    scanner's hits. Accepts a JSON list of row ids or an object with a "rows" list.
    A missing file means an empty list; the manifest records which."""
    global _forward_rows, _forward_rows_source
    path = path or DEFAULT_FORWARD_ROWS_FILE
    if not os.path.exists(path):
        _forward_rows, _forward_rows_source = set(), "missing: %s" % path
        return _forward_rows
    with open(path) as f:
        data = json.load(f)
    rows = data.get("rows", []) if isinstance(data, dict) else data
    _forward_rows = {int(r) for r in rows}
    _forward_rows_source = "%s (%d rows)" % (path, len(_forward_rows))
    return _forward_rows


def forward_rows_source() -> str:
    return _forward_rows_source

_MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_ISO_DT = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)[T ](\d\d):(\d\d)(?::(\d\d))?Z?\b")
_ISO_D = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
_MDY = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)\b")
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+(20\d\d)\b")
_DATELINE = re.compile(r"\b(published|updated|posted|modified|last updated|last modified)\b", re.IGNORECASE)
_HEADER = re.compile(r"^--- News Article \d+: (.*) ---$")
DATELINE_LOOKAHEAD = 2
BARE_DATE_LINES = 25


def bare_date(line: str):
    """A short line that is nothing but one date (optionally a weekday, a time or a time
    zone around it), such as "Feb. 8, 2026" or "February 14, 2026 · 10:00 AM ET".
    Same definition as the benchmark's own dateline scanner."""
    s = line.strip()
    if not s or len(s) > 60:
        return None
    ds = _dates(s)
    if len(ds) != 1:
        return None
    rest = _ISO_D.sub(" ", _MDY.sub(" ", _DMY.sub(" ", s)))
    rest = re.sub(r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b|\b\d{1,2}(:\d\d)?\s*(am|pm)?\b"
                  r"|\b(et|est|edt|pt|pst|pdt|ct|cst|cdt|mt|gmt|utc|bst|cet)\b|\b(at|on|by)\b",
                  " ", rest, flags=re.IGNORECASE)
    return ds[0] if not re.search(r"[A-Za-z0-9]", rest) else None


def scan_datelines(news_lines: List[str]):
    """Every dateline date found in a news block, as (date, line text), by the rules in
    the module docstring (item 5)."""
    found = []
    art_line = 0
    for i, line in enumerate(news_lines):
        if _HEADER.match(line.strip()):
            art_line = 0
            continue
        art_line += 1
        if art_line <= BARE_DATE_LINES:
            bd = bare_date(line)
            if bd:
                found.append((bd, line.strip()[:120]))
        if not _DATELINE.search(line):
            continue
        ds = _dates(line)
        shown = line.strip()[:120]
        if not ds:
            for j in range(i + 1, min(i + 1 + DATELINE_LOOKAHEAD, len(news_lines))):
                bd = bare_date(news_lines[j])
                if bd:
                    ds = [bd]
                    shown = line.strip()[:60] + " / " + news_lines[j].strip()[:60]
                    break
        if ds:
            found.append((max(ds), shown))
    return found


class LeakError(AssertionError):
    pass


def _ts(s: str) -> datetime:
    return datetime.strptime(s.replace("Z", "")[:19], "%Y-%m-%dT%H:%M:%S")


def _datetimes(text: str) -> List[datetime]:
    out = []
    for y, mo, d, h, mi, s in _ISO_DT.findall(text):
        try:
            out.append(datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0)))
        except ValueError:
            pass
    return out


def _dates(text: str) -> List[date]:
    out = []
    for y, mo, d in _ISO_D.findall(text):
        try:
            out.append(date(int(y), int(mo), int(d)))
        except ValueError:
            pass
    for mo, d, y in _MDY.findall(text):
        try:
            out.append(date(int(y), _MONTHS[mo[:3].lower()], int(d)))
        except (ValueError, KeyError):
            pass
    for d, mo, y in _DMY.findall(text):
        try:
            out.append(date(int(y), _MONTHS[mo[:3].lower()], int(d)))
        except (ValueError, KeyError):
            pass
    return out


def check_row(row: Dict) -> None:
    """Checks on the row itself (before any prompt is built)."""
    if not row.get("forecast_ts"):
        raise LeakError("row %s has no forecast_ts" % row["row_id"])
    if row.get("news_captured_at") and row["news_captured_at"] > row["forecast_ts"]:
        raise LeakError("row %s: news captured at %s, after forecast %s"
                        % (row["row_id"], row["news_captured_at"], row["forecast_ts"]))


def check_prompt(row: Dict, prompt: str) -> Dict:
    """Checks on the exact prompt text. Raises LeakError on a hard failure; returns
    a small dict of soft findings to store in the log record."""
    fts = _ts(row["forecast_ts"])
    today_lines = [l for l in prompt.splitlines() if l.startswith("Today's date and time (UTC): ")]
    if len(today_lines) != 1 or today_lines[0].split(": ", 1)[1].strip() != row["forecast_ts"]:
        raise LeakError("row %s: today line does not equal forecast_ts" % row["row_id"])
    future_dates = set()
    future_timestamps = set()
    next_day_datelines = []
    forward_hits = []
    forward = _forward_rows if _forward_rows is not None else load_forward_rows()
    lines = prompt.splitlines()
    news_start = lines.index("News:") if "News:" in lines else len(lines)
    news_end = next((i for i in range(news_start, len(lines))
                     if lines[i].startswith("Give your own probability")), len(lines))
    for i, line in enumerate(lines):
        in_news = news_start < i < news_end
        if in_news:
            future_timestamps.update(dt.isoformat() for dt in _datetimes(line) if dt > fts)
        future_dates.update(d for d in _dates(line) if d > fts.date())
    for d, shown in scan_datelines(lines[news_start + 1:news_end]):
        gap = (d - fts.date()).days
        if gap >= 2:
            if row["row_id"] in forward:
                forward_hits.append("%s (+%d days)" % (shown, gap))
                continue
            raise LeakError("row %s: dateline %d days after forecast %s: %s"
                            % (row["row_id"], gap, fts.date(), shown))
        if gap == 1:
            next_day_datelines.append(shown)
    return {"future_date_mentions": sorted(d.isoformat() for d in future_dates),
            "future_timestamp_mentions": sorted(future_timestamps),
            "dateline_next_day": next_day_datelines,
            "scanner_hit_labelled_forward": forward_hits}


def check_reader_after(reader: BenchmarkReader) -> None:
    extra = reader.tables_touched - {"market_moments"}
    if extra:
        raise LeakError("tables other than market_moments were touched: %s" % sorted(extra))
    if reader.denied_attempts:
        raise LeakError("statements were refused during the run: %s" % reader.denied_attempts[:5])


def pre_run(reader: BenchmarkReader, rows: List[Dict], mode: str, template: str = "default",
            forward_rows_file: Optional[str] = None):
    """Returns (report lines, excluded, blocked): two {row_id: reason} dicts of rows
    that must not be sent. `excluded` = the benchmark's own flag; `blocked` = the
    harness's own prompt check (rows not already excluded)."""
    report = [reader.guard_self_test()]
    load_forward_rows(forward_rows_file)
    report.append("labelled-forward list: %s" % _forward_rows_source)
    excluded: Dict[int, str] = {}
    blocked: Dict[int, str] = {}
    for row in rows:
        check_row(row)
        if row.get("news_dateline_after_capture"):
            excluded[row["row_id"]] = ("benchmark flag news_dateline_after_capture = 1 (latest dateline %s)"
                                       % row.get("news_latest_dateline"))
            continue
        for strict in (False, True):
            try:
                check_prompt(row, build_prompt(row, mode, strict=strict, template=template))
            except LeakError as e:
                blocked[row["row_id"]] = str(e)
                break
    report.append("news_captured_at <= forecast_ts on all %d rows" % len(rows))
    report.append("benchmark flag news_dateline_after_capture: %d of %d rows excluded%s"
                  % (len(excluded), len(rows), (": " + ", ".join(str(k) for k in excluded)) if excluded else ""))
    report.append("harness prompt check (mode %s): %d further rows blocked%s"
                  % (mode, len(blocked), (": " + ", ".join(str(k) for k in blocked)) if blocked else ""))
    return report, excluded, blocked
