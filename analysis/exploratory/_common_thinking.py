"""Shared loader for the thinking-and-length scripts. Read-only on both databases."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import os, sys, sqlite3, math, random, statistics as st
from collections import defaultdict
REPO = _paths.REPO
import scoring

RUNS_DB = os.path.join(REPO, "harness", "runs", "runs.db")
EXCLUDE = set(_roster.NOT_SCORED)  # the probe-validation model and the fixture
SHORT = _roster.SHORT              # display names from harness/models.json
MODELS = _roster.SCORED            # every scored model in the roster
SEED, N_BOOT = 20260902, 1000


def load_scored(mode="normal", primary_only=True):
    """One scored row per (model, market-moment) with the run's token/latency fields as extras."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    by_key = {m["moment_key"]: m for m in moments.values()}
    cuts = scoring.load_or_compute_cuts(moments)
    rcon = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    rcon.row_factory = sqlite3.Row
    rows = []
    for r in rcon.execute("""SELECT model, mode, window, moment_key, parsed_probability, input_tokens, output_tokens,
                                    reasoning_tokens, finish_reason, latency_s, status, strict, attempt, parse_warning,
                                    length(raw_reply) AS reply_chars
                             FROM forecasts WHERE mode = ? AND status IN ('ok','ok_after_retry')
                               AND parsed_probability IS NOT NULL""", (mode,)):
        if r["model"] in EXCLUDE or r["moment_key"] not in by_key:
            continue
        m = by_key[r["moment_key"]]
        if primary_only and not m["primary"]:
            continue
        rt, ot = r["reasoning_tokens"], r["output_tokens"]
        extra = {"model": r["model"], "short": SHORT[r["model"]], "reasoning_tokens": rt, "output_tokens": ot,
                 "answer_tokens": (ot - rt) if (ot is not None and rt is not None) else None,
                 "input_tokens": r["input_tokens"], "latency_s": r["latency_s"], "strict": r["strict"] or 0,
                 "attempt": r["attempt"], "finish_reason": r["finish_reason"], "reply_chars": r["reply_chars"],
                 "parse_warning": r["parse_warning"], "mode": mode}
        row = scoring.score_forecast(m, r["parsed_probability"], r["model"], cuts, extra=extra)
        row["abs_dev"] = abs(row["p"] - row["q"])          # disagreement with the market
        row["log_rt"] = math.log(rt) if rt else None
        rows.append(row)
    return rows


def boot_mean(rows, key, n=N_BOOT, seed=SEED):
    return scoring.boot_mean(rows, key, n=n, seed=seed)


def boot_diff(a, b, key, n=N_BOOT, seed=SEED):
    return scoring.boot_diff(a, b, key, n=n, seed=seed)


def boot_slope(rows, ykey, xkey, n=N_BOOT, seed=SEED):
    return scoring.boot_slope(rows, ykey, xkey=xkey, n=n, seed=seed)


def boot_corr(rows, xkey, ykey, n=N_BOOT, seed=SEED):
    """Pearson correlation with an event-cluster bootstrap interval."""
    g = defaultdict(list)
    for r in rows:
        if r.get(xkey) is not None and r.get(ykey) is not None:
            g[r["event_id"]].append((r[xkey], r[ykey]))
    ids = list(g)
    def corr(pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        mx, my = st.fmean(xs), st.fmean(ys)
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
        return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
    pts = [p for i in ids for p in g[i]]
    point = corr(pts)
    rng = random.Random(seed)
    cs = sorted(corr([p for i in rng.choices(ids, k=len(ids)) for p in g[i]]) for _ in range(n))
    return point, cs[int(0.025 * n)], cs[int(0.975 * n)], len(pts)


def fmt(t):
    return "%+.4f [%+.4f, %+.4f]" % tuple(t[:3])
