"""Repo-relative paths for the exploratory scripts. Every script starts with

    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths

which puts analysis/ on sys.path (for scoring, harness_loader, price_paths) and exposes the
two database paths. Both databases are opened read-only; nothing here writes to them.
"""
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
RUNS_DB = os.path.join(REPO, "harness", "runs", "runs.db")
BENCH_DB = os.path.join(REPO, "benchmark", "benchmark.db")
FIG_DIR = os.path.join(HERE, "figures")
EXCLUDE_MODEL = "openrouter/z-ai/glm-5.3-flash"   # the probe-validation model, never scored

if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def runs_con() -> sqlite3.Connection:
    con = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    con.row_factory = sqlite3.Row
    return con


def want_figure() -> bool:
    """Figures are optional: only drawn when the script is run with --figure (matplotlib needed then)."""
    return "--figure" in sys.argv


def test_split_only() -> bool:
    """True when the script was run with --test-split.

    These scripts score the primary set as a whole by default, both the calibration split and the
    test split, because they were written to look at shape rather than to report a headline. The
    result tables in analysis/out/results use the test split alone. Passing --test-split makes an
    exploratory script use the same rows as those tables, so a number quoted from a script and a
    number quoted from a table mean the same thing.

    The flag only ever drops rows. The depth cuts are read from out/depth_cuts.json, which was fixed
    on the full primary set, so buckets keep the same dollar boundaries either way. Default is off.
    """
    return "--test-split" in sys.argv
