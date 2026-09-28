# Harness tests

These tests run the harness end to end with the fake model (see
`../adapters/fake.py`), so no real model is called and nothing is spent. Each
works in a temporary folder and cleans up after itself.

| File | What it checks |
|---|---|
| `test_output_limit.py` | When a host refuses a reply for running past the length limit, only that row is affected: the stricter re-ask gets a chance, and the run carries on instead of stopping. |
| `test_resume_counts.py` | When a run is stopped and resumed, the totals in its `run.json` cover the whole run, not just the last sitting. |

Run each from the repository root, for example:

```sh
python3 harness/tests/test_output_limit.py
```

They need a benchmark database at `benchmark/benchmark.db` (built by
`analysis/build_db.py`).
