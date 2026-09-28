# runs.db: the index of harness runs

`harness/runs/runs.db` is a SQLite index built from the run folders under `harness/runs/`. The folders (`run.json` and `records.jsonl`) are the raw record and stay exactly as `run.py` writes them; the database is derived from them and can be rebuilt at any time with `python3 harness/load_runs.py --rebuild`. Git-ignored, like the folders. Nothing in it reads `benchmark.db`.

**Rule: changes to this schema are additive only** (new columns, tables or views; never a rename, removal, or change of meaning). `schema_version` records the version; it is 1.

## How rows get in

- `run.py` calls the loader for its own run when the run finishes, also after `--resume`.
- `python3 harness/load_runs.py --load <run_id>` loads or reloads one folder; `--rebuild` deletes the database and loads every folder.
- Loading a run id already present replaces that run's rows, so a resumed run reloads cleanly.
- `python3 harness/load_runs.py --void <run_id> --reason "..."` writes `void: true`, `void_at` and `void_reason` into the run's `run.json`, then reloads it. Void runs stay in the tables and leave the views.
- `python3 harness/load_runs.py --check` runs the guard and rebuilds the views.
- A `runs.db` file older than the schema (columns missing) is refused with a message to `--rebuild`.
- `run.py --dry-run` and `run.py --smoke` write their manifests with `void: true` themselves.

## Table `runs`

One row per manifest.

| Column | Meaning |
|---|---|
| `run_id` | The folder name; primary key. |
| `mode` | `normal` or `memory_probe`. |
| `model` | The adapter name (the LiteLLM model id, or `fake:0.5`). |
| `model_spec` | The `--model` argument as typed. |
| `window` | `A`, `B`, `C`, or null when the run was not restricted to a window. |
| `template` | `default` (price-blind) or `with_price`. |
| `template_id` | `harness-prompt-<hash>`. |
| `prompt_template_hash` | The template's hash. |
| `db_sha256` | Hash of the benchmark file the run read. |
| `git_commit` | Commit at run time, `-dirty` appended when the tree had changes. |
| `git_tree` | `clean`, `dirty`, or `dirty, override used`. |
| `litellm_version` | LiteLLM version in the environment that ran it. |
| `started_at`, `finished_at` | UTC. |
| `rows_planned`, `rows_ok`, `rows_failed`, `rows_blocked`, `rows_excluded`, `rows_completed`, `rows_not_started` | Counts from the manifest (ineligible rows are in `manifest_json`). |
| `cost_usd` | Accumulated cost from our price table. |
| `litellm_cost_usd` | Accumulated LiteLLM estimate. |
| `cost_gap_percent` | LiteLLM minus ours, as a percentage of ours. |
| `stopped_at_budget_cap` | 0/1. |
| `stopped_on_permanent_error` | The error message, or null. |
| `void` | 0/1, from the `void` flag in `run.json`. Void runs are kept but feed no view. |
| `manifest_json` | The whole `run.json` as text. |
| `pinned_host`, `host_override` | The host slug the calls were pinned to, and the `--host-override` value when one was used (else null). |
| `expected_served_model` | The `served_model` string from `models.json` that every reply had to match. |
| `dry_run`, `smoke` | 0/1. Dry runs and smoke runs are written void, so they never feed the views. |
| `tokens_per_s_median` | Smoke runs: the median of `tokens_per_s` over the calls. |
| `state` | `finished`, `stopped_at_budget_cap`, `stopped_on_permanent_error`, `paused_on_network`, `paused_by_owner`, `stopped_by_error`; while a run is going, `running` or `host_saturated` (the manifest is rewritten mid-run when it changes). |
| `rate_limit_429s` | Number of 429 replies in the run. |
| `network_outages_count`, `rows_requeued_on_outage` | Outages seen (details in `manifest_json`) and rows put back on the queue because of them. |
| `seconds_paused_total`, `rows_per_min_outside_pauses` | Seconds the run spent paused on 429s (overlapping pauses counted once) and the pace over the time not paused. |
| `resume_commit` | The code commit at the last `--resume`, beside the original `git_commit`; the full list is `resumes` in `manifest_json`. |

## Table `calls`

One row per line of `records.jsonl`: every attempt and every transport try, nothing dropped. The prompt text is not stored, only its hash; the folder has the text.

| Column | Meaning |
|---|---|
| `call_id` | Row id in this database. |
| `run_id`, `line_no` | Which folder and which line of its `records.jsonl`. |
| `mode`, `model`, `template` | As on the run. |
| `row_id`, `moment_key`, `window`, `venue`, `venue_market_id`, `forecast_ts`, `book_source` | The benchmark row. |
| `attempt` | 1 = normal prompt, 2 = strict prompt. |
| `transport_try` | 1 upward within an attempt. |
| `strict`, `final` | 0/1. The final record of a row is its outcome. |
| `status` | `ok`, `ok_after_retry`, `retry_needed`, `failed`, `excluded_by_benchmark_flag`, `leak_check_failed`, `ineligible_release_date`. |
| `error_class` | `transport`, `permanent`, `no_number`, `bare_integer`, `output_limit` (the host refused a reply for running past the reply limit instead of truncating it: the strict retry gets a go, and only that row fails if it fails too), `network_outage` (a non-final record: the row went back to the queue), or null. |
| `error` | Traceback or reason text. |
| `prompt_sha256` | Hash of the exact prompt sent. |
| `raw_reply` | The exact reply text. |
| `parsed_probability` | The parsed number, or null. |
| `parse_warning` | JSON list (`percent_form`, `extreme`, `bare_integer`, `rejected_number_form`). |
| `timestamp` | When the record was written, UTC. |
| `input_tokens`, `output_tokens` | Provider counts (local estimates for the fake model). |
| `reasoning_tokens` | Thinking tokens used, from the provider's usage details, when reported. |
| `cost_usd`, `litellm_cost_usd` | Our table's estimate and LiteLLM's, per call. |
| `finish_reason` | As the provider reported it. |
| `provider` | The host that served the call. |
| `served_model` | The model string the provider returned. |
| `latency_s` | Wall time of the call. |
| `tokens_per_s` | Output tokens over wall time, when both are known. |
| `rate_limit` | JSON: for a 429, the Retry-After value, the wait applied, and the error body's metadata (error_type, provider_code, provider_name, raw). |
| `rate_limit_headers` | JSON: the x-ratelimit-* headers on a successful call (direct OpenAI). |
| `price_shown_from` | `mid_yes` or `price_yes` (only meaningful for the `with_price` template). |
| `description_paragraphs_stripped` | Count of clarification paragraphs removed. |
| `request_params` | JSON: exactly what was sent, minus the prompt. |
| `adapter_raw` | JSON: response id, model, provider, finish reason, reasoning tokens, pinned host, LiteLLM cost. |
| `leak_check` | JSON: soft findings for the prompt. |
| `description_stripped_text` | JSON list of the removed paragraphs. |

Indexes: `(run_id, row_id)`, `(model, mode, moment_key)`, `moment_key`.

## Table `schema_version`

One row: `version`, `created_at`.

## Views

- **`forecasts`**: for each (model, mode, row_id), the latest final record from non-void runs, latest by the record's `timestamp`, then the run's `started_at`. All `calls` columns plus `run_started_at`.
- **`probe_forecasts`**: for each (model, row_id), the `forecasts` row with mode `memory_probe` when one exists, else the normal-mode row, with `source` = `memory_probe` or `normal_stand_in` (on rows without news the normal record is the probe record).

Both views hold one row per row id that the run finished, and that includes the rows
that never produced a number: a row excluded by a benchmark flag, blocked by the leak
check, skipped by the release-date rule, or failed after the strict retry is in the
view with `parsed_probability` null. That is on purpose, so the views show what happened
to every row. **Anything that scores, averages or counts forecasts must filter on
`parsed_probability IS NOT NULL`** (and, if it wants successes only, `status IN ('ok',
'ok_after_retry')`). As an example, `kimi-k2.5_probe_A` contributes 162 such null rows:
159 excluded by the benchmark flag and 3 failed.

## The guard

Before the views are built, the loader groups the non-void runs by (model, mode, window) and checks that every run in a group agrees on `prompt_template_hash`, `db_sha256`, and the request settings (thinking cap or effort, temperature, token limits, host or provider order, read from `request_params` on the calls, or from the manifest's settings when no call carried them). If any group disagrees the views are not built and the loader names the runs; `run.py` prints the same warning at the end of the run. Fix by voiding the run that does not belong, then `--check`.

`--check` also recounts every run against its own call log: `rows_ok + rows_failed +
rows_blocked + rows_excluded + rows_ineligible` in the manifest must equal the number
of distinct row ids that have a final call. A mismatch means the manifest counters and
the record disagree, and the line names the run and both numbers. This does not stop the
views from being built; it is a metadata check, not a data one. Dry runs are skipped
because they make no calls.
