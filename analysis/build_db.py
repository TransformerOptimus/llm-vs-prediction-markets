#!/usr/bin/env python3
"""Rebuild the local SQLite databases from the released Parquet files.

The analysis code (scoring.py, results.py, gates.py, the exploratory scripts)
reads two SQLite files in a fixed schema. This script recreates them, in that
exact schema, from the released dataset, so the code runs unchanged:

  benchmark/benchmark.db   market rows, outcomes, price paths, news
  harness/runs/runs.db     every model call and run
  analysis/out/            the fixed inputs results.py reads (depth cuts, gate,
                           memory-probe flag files)

Windows:
  A (PolyBench week, Polymarket) and B (Polymarket) are rebuilt in full.
  C (Kalshi) is rebuilt only if you supply Kalshi's market data yourself
  (--kalshi, see below). The release carries only identifiers and our own labels
  for Window C, never Kalshi's prices, books, outcomes or rules. Without --kalshi,
  Window C rows are left out of benchmark.db and Window C runs and calls are left
  out of runs.db (the analysis loader refuses records it cannot join).

News text:
  The main dataset stores a SHA-256 checksum of every news text. With --news
  pointing at the news dataset, the full text is restored; without it, news_text
  is left empty and news_articles.text is NULL (no result table needs the text).
  A few texts and one reply had a personal email address replaced by
  "[email removed]" (flag columns *_redacted); their checksums are those of the
  original text, so those texts will not re-hash to the stored checksum.

Supplying Window C (--kalshi DIR). Three Parquet files, joined on
(venue_market_id, forecast_ts), where venue_market_id is the Kalshi market
ticker and forecast_ts the forecast moment as given in the release:
  kalshi_moments.parquet             one row per Window C market-moment, with every
                                     column of market_moments and moment_outcomes
                                     that the release leaves out (list printed by
                                     `build_db.py --kalshi-columns`)
  kalshi_price_paths.parquet         venue_market_id, forecast_ts, ts, price_yes
  kalshi_price_paths_archive.parquet venue_market_id, forecast_ts, ts, price_yes,
                                     best_bid_yes, best_ask_yes, depth_usd, depth_usd_full
Column meanings are in the main dataset's README.md (its data card).
This script never downloads anything.

Usage:
  python3 analysis/build_db.py --main PATH_TO_MAIN_DATASET [--news PATH] [--kalshi DIR]
                               [--root REPO_ROOT] [--force]
Needs pyarrow.
"""
import argparse
import glob
import json
import os
import shutil
import sqlite3
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

WINDOW_FOLDERS = ("window_a", "window_b", "window_c")
PRICE_PATH_COLS = ["ts", "price_yes"]
ARCHIVE_COLS = ["ts", "price_yes", "best_bid_yes", "best_ask_yes", "depth_usd", "depth_usd_full"]
ANALYSIS_OUT_FILES = ("depth_cuts.json", "calibration_gate.json", "calibration_gate.md",
                      "calibration_moment_keys.txt", "validation_moment_keys.txt")


def table_columns(con, table):
    return [r[1] for r in con.execute("PRAGMA table_info(%s)" % table)]


def rows_of(path, columns=None):
    t = pq.read_table(path, columns=columns)
    names = t.column_names
    cols = [t.column(i).to_pylist() for i in range(len(names))]
    return names, list(zip(*cols)) if cols else []


def insert(con, table, names, rows):
    if not rows:
        return 0
    sql = "INSERT INTO %s (%s) VALUES (%s)" % (table, ",".join('"%s"' % n for n in names), ",".join("?" * len(names)))
    con.executemany(sql, rows)
    return len(rows)


def load_texts(news_dir):
    texts = {}
    if not news_dir:
        return texts
    for f in glob.glob(os.path.join(news_dir, "texts.parquet")) + glob.glob(os.path.join(news_dir, "window_a", "texts.parquet")):
        names, rows = rows_of(f, ["sha256", "text"])
        texts.update(dict(rows))
    return texts


def build_benchmark(main, news_texts, kalshi, path, report):
    con = sqlite3.connect(path)
    con.executescript(open(os.path.join(main, "schema_benchmark.sql")).read())
    mm_cols = table_columns(con, "market_moments")
    mo_cols = table_columns(con, "moment_outcomes")
    na_cols = table_columns(con, "news_articles")
    missing_text = 0
    for folder in WINDOW_FOLDERS:
        wdir = os.path.join(main, "benchmark", folder)
        names, rows = rows_of(os.path.join(wdir, "market_moments.parquet"))
        oname, orows = rows_of(os.path.join(wdir, "moment_outcomes.parquet"))
        if folder == "window_c":
            if not kalshi:
                report["window_c"] = "skipped (no --kalshi); %d Window C rows need Kalshi data" % len(rows)
                continue
            names, rows, oname, orows = merge_kalshi(names, rows, oname, orows, kalshi, mm_cols, mo_cols)
        # restore news_text from its checksum
        i_sha = names.index("news_text_sha256")
        keep = [j for j, c in enumerate(names) if c in mm_cols]
        out_names = [names[j] for j in keep] + ["news_text"]
        out_rows = []
        for r in rows:
            text = news_texts.get(r[i_sha])
            if text is None:
                missing_text += 1
                text = ""
            out_rows.append([r[j] for j in keep] + [text])
        report["market_moments_" + folder] = insert(con, "market_moments", out_names, out_rows)
        report["moment_outcomes_" + folder] = insert(con, "moment_outcomes", oname, orows)
        pp = os.path.join(wdir, "price_paths.parquet")
        if os.path.exists(pp):
            n, r = rows_of(pp)
            report["price_paths_" + folder] = insert(con, "price_paths", n, r)
        na = os.path.join(wdir, "news_articles.parquet")
        if os.path.exists(na):
            n, r = rows_of(na)
            i_sha = n.index("text_sha256")
            keep = [j for j, c in enumerate(n) if c in na_cols]
            out = []
            for row in r:
                text = news_texts.get(row[i_sha]) if row[i_sha] else None
                if row[i_sha] and text is None:
                    missing_text += 1
                out.append([row[j] for j in keep] + [text])
            report["news_articles_" + folder] = insert(con, "news_articles", [n[j] for j in keep] + ["text"], out)
    if kalshi:
        report["price_paths_kalshi"], report["price_paths_archive_kalshi"] = insert_kalshi_paths(con, kalshi)
    n, r = rows_of(os.path.join(main, "benchmark", "quarantine_flags.parquet"))
    insert(con, "quarantine_flags", n, r)
    report["news_texts_missing"] = missing_text
    con.commit()
    con.close()


def merge_kalshi(names, rows, oname, orows, kalshi, mm_cols, mo_cols):
    kn, kr = rows_of(os.path.join(kalshi, "kalshi_moments.parquet"))
    key = {(r[kn.index("venue_market_id")], r[kn.index("forecast_ts")]): r for r in kr}
    need_mm = [c for c in mm_cols if c not in names and c != "news_text"]
    need_mo = [c for c in mo_cols if c not in oname]
    lacking = [c for c in need_mm + need_mo if c not in kn]
    if lacking:
        sys.exit("kalshi_moments.parquet lacks columns: %s" % ", ".join(lacking))
    by_row = {}
    new_rows = []
    iv, it, ir = names.index("venue_market_id"), names.index("forecast_ts"), names.index("row_id")
    for r in rows:
        k = key.get((r[iv], r[it]))
        if k is None:
            sys.exit("no Kalshi data for %s at %s" % (r[iv], r[it]))
        new_rows.append(list(r) + [k[kn.index(c)] for c in need_mm])
        by_row[r[ir]] = k
    oi = oname.index("row_id")
    new_orows = [list(r) + [by_row[r[oi]][kn.index(c)] for c in need_mo] for r in orows]
    return names + need_mm, new_rows, oname + need_mo, new_orows


def insert_kalshi_paths(con, kalshi):
    rowid = {(v, t): r for r, v, t in con.execute("SELECT row_id, venue_market_id, forecast_ts FROM market_moments WHERE venue = 'kalshi'")}
    counts = []
    for fname, table, cols in (("kalshi_price_paths.parquet", "price_paths", PRICE_PATH_COLS),
                               ("kalshi_price_paths_archive.parquet", "price_paths_archive", ARCHIVE_COLS)):
        n, r = rows_of(os.path.join(kalshi, fname))
        iv, it = n.index("venue_market_id"), n.index("forecast_ts")
        idx = [n.index(c) for c in cols]
        out = [[rowid[(x[iv], x[it])]] + [x[j] for j in idx] for x in r]
        counts.append(insert(con, table, ["row_id"] + cols, out))
    return counts


def build_runs(main, path, with_window_c, report):
    """Window C runs and calls go in only when Window C market rows exist (--kalshi):
    the analysis loader refuses to score records that do not join to benchmark.db."""
    con = sqlite3.connect(path)
    con.executescript(open(os.path.join(main, "schema_runs.sql")).read())
    for table in ("schema_version", "runs"):
        n, r = rows_of(os.path.join(main, "runs", table + ".parquet"))
        if table == "runs" and not with_window_c:
            iw = n.index("window")
            r = [x for x in r if x[iw] != "C"]
        report[table] = insert(con, table, n, r)
    call_cols = table_columns(con, "calls")   # release-only columns such as raw_reply_redacted are not loaded
    pf = pq.ParquetFile(os.path.join(main, "runs", "calls.parquet"))
    total = skipped = 0
    for batch in pf.iter_batches(batch_size=5000):
        names = batch.schema.names
        cols = [batch.column(i).to_pylist() for i in range(len(names))]
        rows = list(zip(*cols))
        if not with_window_c:
            iw = names.index("window")
            kept = [x for x in rows if x[iw] != "C"]
            skipped += len(rows) - len(kept)
            rows = kept
        keep = [j for j, c in enumerate(names) if c in call_cols]
        total += insert(con, "calls", [names[j] for j in keep], [[x[j] for j in keep] for x in rows])
    report["calls"] = total
    if skipped:
        report["calls_window_c_left_out"] = skipped
    con.commit()
    con.close()


def restore_analysis_out(main, root, report):
    src = os.path.join(main, "analysis_out")
    dst = os.path.join(root, "analysis", "out")
    os.makedirs(os.path.join(dst, "results"), exist_ok=True)
    n = 0
    for f in ANALYSIS_OUT_FILES:
        shutil.copy2(os.path.join(src, f), dst)
        n += 1
    for f in glob.glob(os.path.join(src, "quarantine_*.json*")):
        shutil.copy2(f, dst)
        n += 1
    report["analysis_out_files"] = n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--main", help="the main dataset folder (holds benchmark/, runs/, analysis_out/)")
    ap.add_argument("--news", help="the news dataset folder (holds texts.parquet)")
    ap.add_argument("--kalshi", help="folder with your own Kalshi files for Window C")
    ap.add_argument("--root", default=REPO, help="repository root to write into (default: this repo)")
    ap.add_argument("--force", action="store_true", help="replace existing database files")
    ap.add_argument("--kalshi-columns", action="store_true", help="print the columns kalshi_moments.parquet must hold")
    a = ap.parse_args()
    if a.kalshi_columns:
        if not a.main:
            sys.exit("--kalshi-columns needs --main")
        con = sqlite3.connect(":memory:")
        con.executescript(open(os.path.join(a.main, "schema_benchmark.sql")).read())
        have = set(rows_of(os.path.join(a.main, "benchmark", "window_c", "market_moments.parquet"))[0])
        have |= set(rows_of(os.path.join(a.main, "benchmark", "window_c", "moment_outcomes.parquet"))[0])
        need = [c for t in ("market_moments", "moment_outcomes") for c in table_columns(con, t) if c not in have and c != "news_text"]
        print("\n".join(["venue_market_id", "forecast_ts"] + need))
        return
    if not a.main:
        sys.exit("--main is required")
    bench = os.path.join(a.root, "benchmark", "benchmark.db")
    runs = os.path.join(a.root, "harness", "runs", "runs.db")
    for p in (bench, runs):
        if os.path.exists(p):
            if not a.force:
                sys.exit("%s exists; use --force to replace it" % p)
            os.remove(p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
    report = {}
    build_benchmark(a.main, load_texts(a.news), a.kalshi, bench, report)
    build_runs(a.main, runs, bool(a.kalshi), report)
    restore_analysis_out(a.main, a.root, report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
