"""Shared loader for the reply-text scripts. Read-only on both databases.

Classes are assigned by regular expressions on the reply body (everything before the
final PROBABILITY line). Reply text is data; nothing in it is executed or followed.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths, _roster  # noqa: E702
import os, re, sqlite3, sys, json
REPO = _paths.REPO
import scoring

RUNS_DB = os.path.join(REPO, "harness", "runs", "runs.db")
EXCLUDE = set(_roster.NOT_SCORED)  # the probe-validation model and the fixture
N_BOOT = 1000
SEED = scoring.SEED

SHORT = _roster.SHORT              # display names from harness/models.json
PROB_LINE = re.compile(r"PROBABILITY\s*:\s*[0-9.]+\s*%?\s*$", re.I | re.M)

# --- text classes -----------------------------------------------------------------
RX = {
    # says the news given was absent / not useful (the prompt line is "No news is available.")
    "no_news": re.compile(r"\b(no|without|absent|lack(?:ing|s)? of|not?\s+(?:any\s+)?(?:relevant|specific|available))\s+(?:\w+\s){0,3}?(news|articles?|reports?|information|data)\b", re.I),
    # refers to the news block it was given
    "cites_news": re.compile(r"\b(the (?:provided|given|supplied|available|attached|included|latest|recent) (?:news|articles?|reports?|headlines?)|according to (?:the )?(?:news|articles?|reports?)|(?:the )?(?:news|articles?|reports?) (?:say|says|state|states|note|notes|indicate|indicates|mention|mentions|suggest|suggests|confirm|confirms|show|shows|report|reports|describe|describes)|as (?:the )?(?:news|article|report) (?:notes|states|says|indicates)|(?:in|from|per) the (?:news|articles?|reports?)\b|news (?:snippets?|items?|context|excerpts?|coverage|headlines?))\b", re.I),
    # hedging language
    "hedge": re.compile(r"\b(uncertain(?:ty)?|unclear|hard to (?:say|know|judge|assess|estimate)|difficult to (?:say|know|judge|assess|estimate|predict)|limited (?:information|data)|can(?:'|no)t (?:be sure|know|verify|confirm)|without (?:more|further|additional) (?:information|data|context)|insufficient (?:information|data)|guess(?:work)?|highly speculative|speculative)\b", re.I),
    # refusal or 'unknowable'
    "refuse": re.compile(r"\b(unknowable|impossible to (?:know|predict|determine|say)|cannot (?:be )?(?:determined|predicted|known|forecast)|can(?:'|no)t (?:provide|give|make) (?:a |an )?(?:probability|forecast|estimate|prediction)|I (?:do not|don't) (?:have|know)\b|no way to (?:know|tell|determine)|not possible to (?:know|determine|predict))\b", re.I),
    # base-rate language
    "base_rate": re.compile(r"\b(base[ -]rate|historical(?:ly)?|on average|typical(?:ly)?|prior|in general|long[ -]run|statistically|empirical(?:ly)?)\b", re.I),
    # market or odds language although no price was shown
    "market_odds": re.compile(r"\b(odds|bookmakers?|sportsbooks?|betting (?:line|market)s?|moneyline|money line|implied probabilit(?:y|ies)|market(?:s)? (?:price|pricing|prices|is pricing|implies|imply|implied|suggests|has priced|would price|probability|consensus|expectation|sentiment)|priced (?:at|in|around)|point spread|the line\b|[-+]\d{3}\b)", re.I),
    # sports-style favourite / underdog talk (kept apart from explicit odds language)
    "favourite": re.compile(r"\b(favou?rites?|favou?red|underdogs?)\b", re.I),
    # cites a specific number in the body (a percent, a dollar figure, a year, a rank, a score)
    "cites_number": re.compile(r"(\d+(?:\.\d+)?\s?%|\$\s?\d|\b(?:19|20)\d{2}\b|\b(?:rank(?:ed|ing)?|no\.|#)\s?\d+|\b\d+\s?(?:-|–|to)\s?\d+\b|\b\d+(?:\.\d+)?\s?(?:goals?|runs?|points?|wins?|games?|matches?|seasons?|days?|hours?|minutes?|weeks?|months?|years?|times?)\b)", re.I),
    # claims of the event being under way or already decided
    "in_progress": re.compile(r"\b(already (?:started|begun|under\s?way|in progress|played|been played|concluded|over|finished|resolved|happened|occurred|taken place)|in progress|under\s?way|currently (?:being played|playing|ongoing)|has (?:likely |probably )?(?:started|begun|concluded|ended|finished))\b", re.I),
    # claims of remembered knowledge ("I recall", "as of my knowledge")
    "recall": re.compile(r"\b(I recall|as I recall|from memory|my (?:training|knowledge)(?: data| cutoff)?|(?:knowledge|training) cutoff|as of my (?:last|latest) (?:update|knowledge))\b", re.I),
}


def body_of(reply: str) -> str:
    """Everything before the final PROBABILITY line."""
    if not reply:
        return ""
    m = list(PROB_LINE.finditer(reply))
    return reply[: m[-1].start()] if m else reply


def classify(reply: str) -> dict:
    b = body_of(reply)
    out = {k: int(bool(rx.search(b))) for k, rx in RX.items()}
    out["body_chars"] = len(b.strip())
    out["body_words"] = len(b.split())
    out["bare"] = int(len(b.strip()) < 20)
    return out


def load_forecasts(mode: str = "normal", statuses=("ok", "ok_after_retry")):
    con = sqlite3.connect("file:%s?mode=ro" % RUNS_DB, uri=True)
    con.row_factory = sqlite3.Row
    q = ("SELECT model, mode, window, row_id, moment_key, parsed_probability, raw_reply, output_tokens, "
         "reasoning_tokens, finish_reason, status, attempt, strict, parse_warning FROM forecasts WHERE mode=? "
         "AND status IN (%s)" % ",".join("?" * len(statuses)))
    rows = [dict(r) for r in con.execute(q, (mode,) + tuple(statuses))]
    return [r for r in rows if r["model"] not in EXCLUDE and r["parsed_probability"] is not None]


def load_scored(mode: str = "normal", primary_only: bool = True):
    """Scored rows: one per (model, moment), on the primary set, with text classes attached."""
    con = scoring.connect()
    moments = scoring.load_moments(con)
    by_key = {m["moment_key"]: m for m in moments.values()}
    cuts = scoring.load_or_compute_cuts(moments)
    out = []
    for f in load_forecasts(mode):
        m = by_key.get(f["moment_key"])
        if m is None or (primary_only and not m["primary"]):
            continue
        extra = classify(f["raw_reply"])
        extra.update({"model": SHORT.get(f["model"], f["model"]), "mode": mode, "attempt": f["attempt"],
                      "strict": f["strict"], "status": f["status"], "output_tokens": f["output_tokens"],
                      "reasoning_tokens": f["reasoning_tokens"], "finish_reason": f["finish_reason"]})
        r = scoring.score_forecast(m, f["parsed_probability"], SHORT.get(f["model"], f["model"]), cuts, extra)
        r["abs_dist"] = abs(r["p"] - r["q"])
        r["signed_dist"] = r["p"] - r["q"]
        out.append(r)
    return out


def boot_diff(a, b, key):
    return scoring.boot_diff(a, b, key, n=N_BOOT, seed=SEED)


def boot_mean(a, key):
    return scoring.boot_mean(a, key, n=N_BOOT, seed=SEED)


def fmt(t, d=4):
    return "%+.*f [%+.*f, %+.*f]" % (d, t[0], d, t[1], d, t[2])


def boot_within_model(rows, cls, key, n=N_BOOT, seed=SEED, min_per=30):
    """Mean(key | cls=1) - mean(key | cls=0) computed inside each model, then averaged with equal
    weight over models that have at least `min_per` rows on both sides. Interval: resample events
    jointly, recompute. Returns (point, lo, hi, per_model dict, n1, n0)."""
    import random, statistics as st
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            g[r["event_id"]].append(r)

    def stat(rs):
        per = defaultdict(lambda: ([], []))
        for r in rs:
            per[r["model"]][0 if r[cls] else 1].append(r[key])
        vals = {m: (st.fmean(a) - st.fmean(b), len(a), len(b)) for m, (a, b) in per.items()
                if len(a) >= min_per and len(b) >= min_per}
        return (st.fmean(v[0] for v in vals.values()) if vals else float("nan")), vals

    point, per_model = stat(rows)
    ids = list(g)
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        draw = [r for c in rng.choices(ids, k=len(ids)) for r in g[c]]
        v, _ = stat(draw)
        if v == v:
            boots.append(v)
    boots.sort()
    n1 = sum(1 for r in rows if r[cls] and r.get(key) is not None)
    n0 = sum(1 for r in rows if not r[cls] and r.get(key) is not None)
    return point, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))], per_model, n1, n0
