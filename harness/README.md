# Harness: asking the models

The harness is the code that prompts each model and records its answer. For one
market-moment of the benchmark it builds a prompt, sends it to one model, reads
back a probability that the question resolves Yes, and logs everything: the
exact prompt, the exact reply, tokens, cost, host and settings.

The model is never shown the market price, the best bid or ask, or the order
book ("price-blind"). It sees the question, the exchange's resolution rules,
the date, the scheduled close and, where one was captured, a block of news.

**You do not need to run the harness to reproduce the paper.** Every call it
made is in the main dataset. Running it again calls paid model APIs with your
own keys.

## What each file does

| File | What it does |
|---|---|
| `run.py` | Runs one model over a set of rows in one mode and writes the log. The main entry point. |
| `prompt.py` | The prompt template. `default` is the study's only template; `with_price` (adds the price) is kept in the code but was not used in the study. |
| `forecast.py` | One row in, one probability out: sends the prompt, retries on network errors, re-asks once with a stricter instruction if the reply has no usable number, and reads the `PROBABILITY: x` line. |
| `leak_check.py` | Checks, before every run and every call, that nothing in the prompt comes from after the forecast time. See "Leak check" below. |
| `benchmark_reader.py` | Opens the benchmark database read-only and lets the harness read only the `market_moments` table, so outcomes and other analysis-only data are out of reach. |
| `gate.py` | Controls the pace of calls: the calls-per-minute limit, a shared pause when a host says "too many requests", and detection of network outages. |
| `run_plan.py`, `plan.example.json` | Runs a list of runs for you: you write a plan file (model, mode, windows, budget per run) and `run_plan.py` works through it, one lane per host, and can be stopped and restarted safely. `plan.example.json` is a free template that uses the fake model. |
| `models.json` | **Which models, and how to call them**: each model's id, release date, host, precision, thinking setting and the exact model string the host must report. The analysis and the paper's roster table read it too. |
| `prices.json` | **What each call costs**: dollars per million tokens for each model and host, with the source and the day it was read. Only the harness reads it, for the cost log and the budget stop. |
| `resolve_models.py` | Checks every roster entry without spending anything: the key is present and the request LiteLLM would send is right. |
| `load_runs.py` | Builds `runs/runs.db`, a SQLite index of every run folder. |
| `summarize.py` | A summary of runs: rows done, failures by kind, tokens, cost, parse warnings, host check. |
| `RUNS_SCHEMA.md` | Every table, view and column of `runs.db`. |
| [`adapters/`](adapters/) | The connection to model providers: one real adapter (through LiteLLM) and one fake model for tests. See its README. |
| [`tests/`](tests/) | Tests that run the harness end to end with the fake model, at no cost. See its README. |
| `runs/` | Not in the repository. Each run writes `runs/<run_id>/run.json` (settings and totals) and `runs/<run_id>/records.jsonl` (one line per call). |

## Trying it without spending money

The fake model answers locally and costs nothing. It needs a benchmark database
at `benchmark/benchmark.db`, built by `analysis/build_db.py` (see
[`../analysis/README.md`](../analysis/README.md)).

```sh
python3 harness/run.py --mode normal       --model fake:0.5 --window A --limit 10
python3 harness/run.py --mode memory_probe --model fake:0.5 --window B --news-only --workers 4
python3 harness/summarize.py                      # every run; or give a run id
```

Real models need `--i-approve-paid-calls`, a price in `prices.json`, an entry in
`models.json`, and API keys in a `.env` file at the repository root
(`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and so on). Keys are never printed or
logged.

## The two modes

- `normal`: the prompt carries the news captured for that row, or the line "No
  news is available." when there is none.
- `memory_probe`: the same prompt with the news always replaced by "No news is
  available." Comparing the two shows what the news was worth, and a model that
  is confidently right without news may already know the outcome (the
  quarantine rule in `analysis/gates.py`). On rows without news the two prompts
  are identical, so the probe runs only on rows with news (`--news-only`).

One run id covers one model, one mode and one window.

## Reference

The rest of this file describes the harness in detail, for anyone who wants to
run or change it.

**Windows.** `--window A|B|C` selects by exchange and data source together (A:
Polymarket via PolyBench, B: Polymarket via the pmxt archive, C: Kalshi via the
pmxt archive; Kalshi row ids start at 1,000,000). Row-id files (`--row-ids-file`,
or its alias `--row-list`) may hold integer row ids or moment keys
(`venue|venue_market_id|forecast_ts`); moment keys stay the same if rows are
renumbered. A run refuses to start if any selected row lacks a `moment_key`,
and a real model refuses rows outside the windows listed for it in
`models.json` (roster label `bridge`: A, B, C; `2026`: B, C; `validation`: A).

**Dry run.** `--dry-run` does everything up to the first model call (row
selection, leak check, eligibility, manifest) and stops: no call, no records,
and the manifest is written with `dry_run: true` and `void: true` so it is
indexed but never used. It does not need `--i-approve-paid-calls` or a clean
git tree.

**Smoke test.** `--smoke` is a short paid test of a host on 20 fixed rows listed
in `harness/smoke_rows.txt`. That file is not included; supply your own (one
row id or moment key per line). Every record carries `tokens_per_s`, the
manifest gets `smoke: true`, `smoke_passed`, `smoke_failures` and a
`smoke_report` (429 count, median latency and tokens per second, a recommended
worker count), and the run is voided automatically so it never feeds the
analysis. Any reply cut off at the length limit, any reply without a usable
number, and any row without a probability fails the test (exit status 1).

**Plan runner.** To run several models, modes or windows in one go, write a
plan file and give it to `run_plan.py`. `plan.example.json` is a template: it
explains every field in its note and runs the free fake model on 10 rows, so
`python3 harness/run_plan.py --plan harness/plan.example.json` shows the whole
cycle at no cost. A plan has one entry per model and mode with its windows (one
run per window, run id `<entry id>_<window>`), workers, an optional per-run
budget cap, and optional row list, `ignore_eligibility` and `host_override`. An
entry's lane is its pinned host from `models.json` (direct OpenAI is its own
lane). `python3 harness/run_plan.py --plan my_plan.json --backup-dir <second
disk> --i-approve-paid-calls` starts at most one run per
lane (`max_per_lane`; `max_per_lane_by_lane` can raise it for one lane) and
every lane at once, each as a separate `run.py` process. State is kept in
`runs/plan_state.json` and child output in `runs/plan_logs/<run id>.log`. On
restart, runs left running are relaunched with `--resume`, finished runs are
never started again, a run paused on a network outage (exit 75) is relaunched
when the network is back, a run stopped by hand (exit 76) is relaunched on the
next start, and a failed run (any other non-zero exit) stops its lane until
`--retry-failed`; other lanes continue. Every finished run's folder is copied
to `--backup-dir/<run id>`. A status line per lane prints every minute.
`--dry-run` gives every run `--dry-run`, so the whole plan can be checked
without a call; `--only` limits the plan to named entries. Without `--plan`,
the runner uses `plan.example.json`.

**Templates.** `--template default|with_price`. `default` is price-blind and is
the study's only template. The template name goes into `run.json` and every
record; `template_id` is `harness-prompt-` plus a hash of the template text,
including the stricter re-ask wording.

**Model settings.** `--reasoning-tokens N` (the thinking cap; default from
`models.json`), `--reasoning-effort` (for `openai/` ids; default from
`models.json`), `--answer-tokens` (the reply limit sent as `max_tokens`,
default 4,000), `--temperature` (a number, or `none` to leave it out). The
thinking cap goes to each provider the way LiteLLM expects it:
`thinking={"type": "enabled", "budget_tokens": N}` for `anthropic/` and
`gemini/` ids, `reasoning_effort` for `openai/` ids, and
`extra_body.reasoning.max_tokens` for `openrouter/` ids (or
`extra_body.reasoning.effort` for a model that takes an effort level instead,
such as gpt-oss). The cap is a request, and hosts differ: in our tests
SiliconFlow and OpenAI's effort levels honoured it while DeepInfra, AtlasCloud
and Novita ignored it, and hosts differ on whether thinking counts inside
`max_tokens` (OpenAI, DeepInfra, AtlasCloud, Novita: inside; SiliconFlow: on
top). Every call logs the thinking tokens the host reports. `openrouter/` ids
get temperature 0 together with the thinking cap; direct `openai/` ids get no
temperature when an effort level is set, and `anthropic/` and `gemini/` ids get
none when a thinking budget is set, because those providers reject the pair.
The exact request parameters are logged on every record as `request_params`.

**Workers and rate limit.** `--workers 1..16` runs rows in parallel;
`--rate-limit-rpm` caps calls per minute across all workers (default 120 for
real models; off for the fake model). For `openrouter/` ids every call is
pinned to the host in `models.json` with fallbacks disabled, and the host that
answered is logged. `--host-override <slug>` swaps in the entry's recorded
backup host. `summarize.py` fails a run that shows more than one host or model
string.

**Rate limits (HTTP 429, "too many requests").** Any 429 pauses new calls for
the whole run until the wait is over, then the same prompt is sent again
without using up a retry (a row gives up only after 30 in a row). The wait is
the host's Retry-After value, held between 2 and 120 seconds (15 seconds if the
host sends none). Above 10 429s in the last minute the pause doubles, above 20
it quadruples, still capped at 120 seconds. When 429s outnumber successful
calls in three whole minutes in a row, the manifest's `state` becomes
`host_saturated` and the run keeps going, so the operator can decide whether to
move to the backup host; it clears after three minutes with 429s at or below
the successes. Each 429 is logged on its record as `rate_limit`, and the
manifest counts them.

**Stopping a run cleanly.** Send the `run.py` process SIGTERM or SIGINT, or
create a file named `STOP` in its run folder. No new rows start, rows in flight
finish and are logged, the manifest gets `state: paused_by_owner`, and the
process exits with code 76. `--resume` continues it; the manifest keeps the
original `git_commit` and records each resume.

**Network outages.** When every call in flight fails with a connection error
within one minute and none succeeded, the run treats it as an outage, not as
row failures: the rows go back in the queue and new calls stop. The run checks
the host every 30 seconds and carries on by itself when it answers. After 30
minutes down (`--network-give-up-s`) it saves its state as `paused_on_network`
and exits with code 75.

**Host kinds.** `direct` (`openai/`, `anthropic/`, `gemini/` ids, the vendor's
own API), OpenRouter (`openrouter/` ids, pinned to one host), and
`direct_compatible`: any OpenAI-compatible endpoint named in `models.json` with
`api_base`, `key_env` (the `.env` variable holding its key), `upstream_model`
and, if needed, `thinking_extra_body` and `thinking_budget_field`. Two
smoke-test-only flags: `--no-reasoning-budget` sends no thinking cap, to
measure a host's default thinking; `--host-quantization Q` asks OpenRouter for
an endpoint at precision Q.

**Served-model check.** Every reply's model string must equal `served_model`
in `models.json` (for direct `openai/` ids, the id without its prefix). A
mismatch logs the reply, marks the row failed as `permanent`, and stops the
run.

**Budget.** `--budget-usd X` adds up `cost_usd` across rows and starts no new
rows once X is reached; rows in flight finish. `--resume` picks up where it
stopped.

**Eligibility.** A row is skipped as `ineligible_release_date` when its
outcome-date bound is before the model's release date in `models.json`, so no
model forecasts an event that resolved before it was published. The bound is
the scheduled end (`market_end_date`), else the forecast time; the true outcome
date is analysis-only and the harness cannot read it. `--ignore-eligibility`
(recorded in the manifest) exists only for the probe-validation model.

**Refusals for real (non-fake) runs.** No `--i-approve-paid-calls`; a git tree
with uncommitted changes unless `--allow-dirty` is given (recorded); a model
missing from `prices.json` or `models.json`; an `openrouter/` id without a
pinned host; rows outside the model's windows; a `--host-override` that is not
the entry's backup host. `--dry-run` skips only the first two.

### How every row ends

Every row ends in exactly one final status, and every call is logged:

- `ok`: the first reply had a usable probability.
- `ok_after_retry`: a network retry or the stricter re-ask produced one.
- `failed`: no probability. `error_class` says why: `transport` (a network or
  server error on all six tries, with waits of 15, 30, 60, 120 and 300 seconds
  between them), `permanent` (an error that must not be retried, or a
  served-model mismatch), `no_number` (two replies, neither with a number), or
  `bare_integer` (a bare 0 or 1 both times).
- `excluded_by_benchmark_flag`, `leak_check_failed`, `ineligible_release_date`:
  no model call was made.

`--resume` skips finished rows and redoes rows whose final status is `failed`.

**Error classes.** Retried: LiteLLM's RateLimitError, Timeout,
APIConnectionError, ServiceUnavailableError and InternalServerError, HTTP 408,
429, 500, 502, 503 and 504. Never retried: LiteLLM's AuthenticationError,
NotFoundError, BadRequestError, PermissionDeniedError and
UnprocessableEntityError, HTTP 400, 401, 402, 403, 404, 413 and 422, and any
message saying the model is unavailable or not found. A permanent error stops
new rows (the same call would fail on every row) and `run.json` records
`stopped_on_permanent_error` with the message. Unknown errors count as
retryable.

### Reading the reply

The last `PROBABILITY: x` line wins (on the stricter re-ask, a reply that is
only a number is also accepted). A number followed by `,`, `/`, `e` or another
digit is rejected ("0,65", "1/3", "1e-3"). A bare `0` or `1` triggers the
stricter re-ask, which asks for a decimal; if it comes back bare again it is
accepted with a warning. `parse_warning` lists `percent_form`, `extreme` (below
0.01 or above 0.99), `bare_integer` or `rejected_number_form`.

### Log record fields

`run_id`, `mode`, `model`, `template`, `row_id`, `moment_key`, `window`,
`venue`, `venue_market_id`, `forecast_ts`, `book_source`, `attempt` (1 normal
prompt, 2 stricter re-ask), `transport_try`, `strict`, `final`, `status`,
`error_class`, `prompt` (exact text sent), `prompt_sha256`, `raw_reply` (exact
text received), `parsed_probability`, `parse_warning`, `timestamp`,
`input_tokens`, `output_tokens`, `cost_usd` (from our price table),
`litellm_cost_usd` (LiteLLM's own estimate), `request_params`,
`finish_reason`, `provider` (the host that served the call), `served_model`,
`latency_s`, `tokens_per_s`, `rate_limit`, `rate_limit_headers`, `error`,
`adapter_raw`, `leak_check`, `description_paragraphs_stripped`,
`description_stripped_text`, `price_shown_from`.

The fake model's token counts are local estimates (tiktoken, o200k encoding);
real models report the provider's own counts.

### Leak check

Before the run:

1. Prove the read guard works by trying to read each analysis-only table and
   the Kalshi close-time column; all must be refused.
2. Every row: the news capture time must not be later than the forecast time.
3. Every row the benchmark flags as having a news dateline after capture is
   excluded and logged as `excluded_by_benchmark_flag`, never sent to a model.
4. Every remaining row: build the prompt and run the checks below. A row that
   fails is logged as `leak_check_failed` and never sent (`--abort-on-leak`
   stops the run instead).

For every prompt (run again right before each call):

5. The "today" line equals the row's forecast time.
6. No dateline in the news two or more calendar days after the forecast time,
   using the same rule as the benchmark's own dateline scanner. One day ahead
   is allowed, because capture times are UTC and sites print local dates; it is
   logged as `dateline_next_day`. A hit blocks the row unless the row was
   hand-checked and labelled a forward reference when the benchmark was built
   (`--forward-rows-file`; without that file, every hit blocks).
7. Plain future dates in the text (an article on 6 February saying "the Super
   Bowl on 8 February") are forward references, not leaks; they are logged, not
   failed.

After the run:

8. The only table SQLite touched must be `market_moments`, with no refused
   reads.

### runs.db

Every run folder is indexed into `harness/runs/runs.db` when the run finishes;
the folders stay the raw record. `python3 harness/load_runs.py --rebuild`
recreates the database from every folder; `--load <run_id>` reloads one;
`--void <run_id> --reason "..."` marks a run void (it stays in the tables but
leaves the views); `--check` runs the consistency guard and rebuilds the views.
The `forecasts` view gives the latest final record per model, mode and row;
`probe_forecasts` gives the memory-probe record per model and row, falling back
to the normal record on rows the probe did not run. Full documentation:
`RUNS_SCHEMA.md`.

### What the harness never does

No web search, no retrieval, no live data, no tools given to the model, and no
system prompt beyond the logged prompt text. The only network call is to the
model provider. The harness never writes to the benchmark.
