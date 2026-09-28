"""The core: one market_moments row in, one probability out, with the exact prompt
and the exact reply.

Forced answers, two kinds of trouble kept apart:

- Transient transport error (timeout, connection error, HTTP 408/5xx): the SAME prompt
  is sent again after a wait that doubles each time (`backoff`, default six tries over
  about ten minutes). If every try fails the row is `failed` with error_class
  'transport'. The strict prompt is never used here.
- HTTP 429 (rate limit): the same prompt is sent again with no backoff sleep here and
  no transport try used up; the run gate (gate.py) holds every worker for the host's
  Retry-After instead. A row gives up only after RATE_LIMIT_MAX_HITS_PER_ROW hits.
- Permanent error (HTTP 400/401/402/403/404/413/422, or a message saying the model is
  unavailable, not found, gated, or needs a different client): never retried; the row
  is `failed` with error_class 'permanent' and the exact message, and the caller stops
  the run, because the same call would fail on every row.
- Served-checkpoint mismatch: the reply's model string differs from the adapter's
  `expected_served_model` (models.json served_model). The reply is logged in full, the
  row is `failed` with error_class 'permanent', and the run stops, the same path as a
  permanent API error.
- Reply received but no usable number: one retry with the stricter instruction
  appended (error_class 'no_number'; 'bare_integer' when the reply was a bare 0 or 1,
  which needs a decimal). If that also fails the row is `failed`.

Every call is returned as one attempt record so the caller logs all of them.
"""
import re
import time
import traceback
from typing import Callable, Dict, List, Optional

from adapters.base import Adapter, classify_error, is_output_limit_error
from gate import NetworkOutage
from prompt import build_prompt_info

DEFAULT_BACKOFF = [15, 30, 60, 120, 300]   # waits between the 6 transport tries (525 s)
RATE_LIMIT_MAX_HITS_PER_ROW = 30           # 429s on one row before it is given up as a transport failure

_PROB_LINE = re.compile(r"PROBABILITY\s*[:=]\s*\**\s*([0-9]*\.?[0-9]+)(\s*%)?", re.IGNORECASE)
_BARE = re.compile(r"^\s*\**\s*([0-9]*\.?[0-9]+)(\s*%)?\s*\**\s*$")
_BAD_FOLLOW = set(",/eE0123456789")


def parse_reply(text: str, lenient: bool = False) -> Dict:
    """Read the probability out of a reply.

    Returns {"probability": float|None, "parse_warning": [..], "bare_integer": bool}.
    Strict (first attempt): needs a "PROBABILITY: x" line; the last one wins. Lenient
    (the retry): also accepts a reply that is nothing but a number or a percent.
    A number followed by ',', '/', 'e' or another digit is rejected ("0,65", "1/3",
    "1e-3" are not probabilities). A bare 0 or 1 (no decimal point, no percent) is
    flagged: on the first attempt it triggers the strict retry asking for a decimal;
    on the retry it is accepted with a warning.
    """
    out: Dict = {"probability": None, "parse_warning": [], "bare_integer": False}
    if not text:
        return out
    matches = list(_PROB_LINE.finditer(text))
    if not matches and lenient:
        m = _BARE.match(text)
        matches = [m] if m else []
    if not matches:
        return out
    m = matches[-1]
    num, pct = m.group(1), m.group(2)
    follow = text[m.end(1):m.end(1) + 1] if pct is None else ""
    if follow and follow in _BAD_FOLLOW:
        out["parse_warning"].append("rejected_number_form")
        return out
    try:
        p = float(num)
    except ValueError:
        return out
    if pct:
        p = p / 100.0
        out["parse_warning"].append("percent_form")
    elif p > 1.0:
        p = p / 100.0
        out["parse_warning"].append("percent_form")
    if p < 0.0 or p > 1.0:
        return out
    if "." not in num and pct is None and num in ("0", "1"):
        out["bare_integer"] = True
        if not lenient:
            return out
        out["parse_warning"].append("bare_integer")
    if p < 0.01 or p > 0.99:
        out["parse_warning"].append("extreme")
    out["probability"] = p
    return out


def parse_probability(text: str, lenient: bool = False) -> Optional[float]:
    return parse_reply(text, lenient)["probability"]


def forecast(row: Dict, mode: str, adapter: Adapter, template: str = "default",
             prompt_check: Optional[Callable[[Dict, str], Dict]] = None,
             backoff: Optional[List[float]] = None, sleep=time.sleep) -> Dict:
    """Run one row. Returns {"probability", "status", "error_class", "parse_warning",
    "attempts": [...]}. Each attempt dict holds: attempt (1 normal, 2 strict),
    transport_try, strict flag, prompt, raw reply, parsed probability, parse_warning,
    error_class, token counts, cost, finish reason, provider, latency, and error text.

    prompt_check, if given, is called with (row, prompt) before every model call and
    may raise to stop the call; its return value is stored as "leak_check".
    """
    waits = DEFAULT_BACKOFF if backoff is None else list(backoff)
    attempts: List[Dict] = []
    probability, error_class, warning = None, None, []
    permanent_message = None
    strict = False
    transport_try = 0
    rate_limit_hits = 0
    while True:
        attempt = 2 if strict else 1
        prompt, prompt_info = build_prompt_info(row, mode, strict=strict, template=template)
        transport_try += 1
        t0 = time.time()
        rec = {"attempt": attempt, "transport_try": transport_try, "strict": strict, "prompt": prompt,
               "raw_reply": None, "parsed_probability": None, "parse_warning": [], "error_class": None,
               "input_tokens": None, "output_tokens": None, "cost_usd": None, "litellm_cost_usd": None,
               "request_params": None, "finish_reason": None,
               "provider": None, "served_model": None, "latency_s": None, "error": None,
               "adapter_raw": None, "leak_check": None, "prompt_info": prompt_info, "error_message": None,
               "rate_limit": None, "rate_limit_headers": None}
        if prompt_check is not None:
            rec["leak_check"] = prompt_check(row, prompt)  # raises before any model call
        try:
            reply = adapter.complete(prompt, meta={"row_id": row["row_id"], "attempt": attempt,
                                                   "transport_try": transport_try, "mode": mode})
        except NetworkOutage as outage:      # not a row failure: the caller requeues the row
            rec["error_class"] = "network_outage"
            rec["error_message"] = str(outage)
            rec["latency_s"] = round(time.time() - t0, 3)
            outage.attempts = attempts + [rec]
            raise
        except Exception as exc:
            rec["error"] = traceback.format_exc()
            rec["rate_limit"] = getattr(exc, "rate_limit", None)
            try:
                rec["error_message"] = str(exc) or exc.__class__.__name__
            except Exception:
                rec["error_message"] = exc.__class__.__name__
            rec["latency_s"] = round(time.time() - t0, 3)
            if is_output_limit_error(exc):
                # the host refused the reply for running past the reply limit instead of
                # sending the truncated text back: this row's answer was too long, not a
                # broken request. Same path as a truncated reply, so the strict retry (a
                # bare number, a short reply) gets a go and only this row fails if that
                # fails too. The run keeps going.
                rec["error_class"] = "output_limit"
                attempts.append(rec)
                if strict:
                    error_class = "output_limit"
                    break
                strict = True
                transport_try = 0
                continue
            if classify_error(exc) == "permanent":
                rec["error_class"] = "permanent"
                attempts.append(rec)
                error_class = "permanent"
                permanent_message = rec["error_message"]
                break
            rec["error_class"] = "transport"
            attempts.append(rec)
            if rec["rate_limit"] is not None:
                # a 429: the gate already holds every worker for the host's Retry-After, so no
                # backoff sleep here and no transport try used up; a row gives up only after
                # RATE_LIMIT_MAX_HITS_PER_ROW hits in a row (a saturated host never fails rows by itself)
                transport_try -= 1
                rate_limit_hits += 1
                if rate_limit_hits >= RATE_LIMIT_MAX_HITS_PER_ROW:
                    error_class = "transport"
                    break
                continue
            if transport_try > len(waits):
                error_class = "transport"
                break
            sleep(waits[transport_try - 1])
            continue
        rec["rate_limit_headers"] = (reply.raw or {}).get("rate_limit_headers")
        rec.update(raw_reply=reply.text, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                   cost_usd=adapter.cost_estimate(reply), litellm_cost_usd=adapter.provider_cost(reply),
                   request_params=(reply.raw or {}).get("request_params"), finish_reason=reply.finish_reason,
                   provider=reply.provider, served_model=reply.served_model, adapter_raw=reply.raw)
        parsed = parse_reply(reply.text, lenient=strict)
        rec["parsed_probability"] = parsed["probability"]
        rec["parse_warning"] = parsed["parse_warning"]
        # latency of the call itself; the gate's rate-limit or 429 wait is logged separately in adapter_raw
        rec["latency_s"] = (reply.raw or {}).get("latency_s_call") or round(time.time() - t0, 3)
        expected = getattr(adapter, "expected_served_model", None)
        if expected and reply.served_model != expected:
            rec["error_class"] = "permanent"
            rec["error_message"] = ("served model mismatch: the provider reported %r (host %r) but models.json "
                                    "expects %r; the pinned checkpoint did not serve this call"
                                    % (reply.served_model, reply.provider, expected))
            rec["error"] = rec["error_message"]
            rec["parsed_probability"] = None           # a reply from the wrong checkpoint is not a forecast
            attempts.append(rec)
            error_class, permanent_message = "permanent", rec["error_message"]
            break
        if parsed["probability"] is not None:
            attempts.append(rec)
            probability, warning = parsed["probability"], parsed["parse_warning"]
            break
        rec["error_class"] = "bare_integer" if parsed["bare_integer"] else "no_number"
        attempts.append(rec)
        if strict:
            error_class = rec["error_class"]
            break
        strict = True
        transport_try = 0
    if probability is not None:
        status = "ok" if len(attempts) == 1 else "ok_after_retry"
    else:
        status = "failed"
    return {"probability": probability, "status": status, "error_class": error_class,
            "permanent_error": permanent_message, "parse_warning": warning, "attempts": attempts}
