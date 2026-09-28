"""Dry-run model. Costs nothing, calls nothing, returns a fixed number.

Spec forms (everything after "fake:" is optional; row lists join ids with "+"):
  fake                      -> always answers 0.5
  fake:0.37                 -> always answers 0.37
  fake:0.5:refuse_rows=7    -> on row 7 refuses every time (a no_number failure record)
  fake:0.5:refuse_first=9   -> on row 9 refuses the first attempt only (strict retry succeeds)
  fake:0.5:raise_rows=3     -> on row 3 raises an exception (a transport error) every time
  fake:0.5:raise_first=4    -> on row 4 raises on the first two calls, then answers
  fake:0.5:bare_rows=5      -> on row 5 answers "PROBABILITY: 1" (a bare integer) on the first attempt
  fake:0.5:outlimit_rows=9   -> on row 9 the normal attempt raises the host's "output limit was reached"
                              400 error every time; the strict retry answers (the real OpenAI behaviour
                              seen with gpt-5.5)
  fake:0.5:outlimit_both=9   -> as above, but the strict retry raises it too, so the row fails
  fake:0.5:permanent_rows=6 -> on row 6 raises an HTTP 403-style error ("model only available on agentic
                               harnesses"), which must never be retried and stops the run
  fake:0.5:served_rows=8    -> on row 8 reports a different served model string ("fake:0.5-other"), which
                               forecast.py treats as a permanent error (served-checkpoint check)
  fake:0.5:length_rows=2    -> on row 2 replies with finish_reason "length" (for the smoke-test check)
  fake:0.5:outage_file=PATH -> every call raises a connection error while the file at PATH exists, and
                               probe() reports the network down; delete the file to end the outage
  fake:0.5:rate429_rows=4   -> on row 4 the first call raises an HTTP 429 with Retry-After: 3 and an
                               OpenRouter-style metadata block, then answers
  fake:0.5:rate429_rows=all -> every row's first call raises the 429 (for the escalation and saturation tests)
  fake:0.5:rate429_retry_after=1    -> the Retry-After value sent (default 3); "none" sends no header
  fake:0.5:rate429_until_call=500   -> the 429s stop once the adapter has made this many calls
                                       (the escalation and saturation clear afterwards)
  fake:0.5:sleep_s=1        -> every call takes this long (a stand-in for latency in timing tests)
"""
import os
import time
from typing import Dict, Optional

from .base import Adapter, PermanentError, Reply, count_tokens_estimate

PROVIDER = "fake"


class FakeAdapter(Adapter):
    is_paid = False
    provider = "fake"

    def __init__(self, value: float = 0.5, refuse_rows=(), refuse_first=(), raise_rows=(), raise_first=(), bare_rows=(),
                 permanent_rows=(), served_rows=(), length_rows=(), outage_file=None, rate429_rows=(),
                 outlimit_rows=(), outlimit_both=(),
                 rate429_retry_after="3", rate429_until_call=None, sleep_s=0.0):
        self.sleep_s = float(sleep_s)
        self.permanent_rows = set(permanent_rows)
        self.outlimit_rows, self.outlimit_both = set(outlimit_rows), set(outlimit_both)
        self.served_rows, self.length_rows = set(served_rows), set(length_rows)
        self.outage_file = outage_file
        self.rate429_all = rate429_rows == "all"
        self.rate429_rows = set() if self.rate429_all else set(rate429_rows)
        self.rate429_retry_after = None if str(rate429_retry_after).lower() == "none" else str(rate429_retry_after)
        self.rate429_until_call = rate429_until_call
        self.value = value
        self.refuse_rows, self.refuse_first = set(refuse_rows), set(refuse_first)
        self.raise_rows, self.raise_first = set(raise_rows), set(raise_first)
        self.bare_rows = set(bare_rows)
        self.name = "fake:%s" % ("%.4f" % value).rstrip("0").rstrip(".")
        self.expected_served_model = self.name
        self.host = self.host_name = "fake"
        self.calls = 0
        self.calls_per_row: Dict = {}

    def complete(self, prompt: str, meta: Optional[Dict] = None) -> Reply:
        self.calls += 1
        meta = meta or {}
        rid, attempt = meta.get("row_id"), meta.get("attempt", 1)
        n = self.calls_per_row[rid] = self.calls_per_row.get(rid, 0) + 1
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.outage_file and os.path.exists(self.outage_file):
            raise ConnectionError("fake network outage: [Errno 8] nodename nor servname provided (connection error)")
        if (self.rate429_all or rid in self.rate429_rows) and n == 1 and \
                (self.rate429_until_call is None or self.calls <= self.rate429_until_call):
            raise FakeRateLimit(self.rate429_retry_after)
        if rid in self.outlimit_both or (rid in self.outlimit_rows and attempt == 1):
            raise FakeBadRequest("litellm.BadRequestError: OpenAIException - Could not finish the message "
                                 "because max_tokens or model output limit was reached. Please try again "
                                 "with higher max_tokens.")
        if rid in self.permanent_rows:
            raise PermanentError("Error code: 403 - {'error': {'message': 'This model is only available on agentic "
                                 "harnesses', 'code': 403}}", status_code=403)
        if rid in self.raise_rows or (rid in self.raise_first and n <= 2):
            raise ConnectionError("fake transport error on row %s (call %d)" % (rid, n))
        if rid in self.refuse_rows or (rid in self.refuse_first and attempt == 1):
            text = "I'm sorry, but I can't provide a forecast for this question."
        elif rid in self.bare_rows and attempt == 1:
            text = "PROBABILITY: 1"
        else:
            text = ("Dry run: this is the fake model, which returns a fixed number without reading the question.\n"
                    "PROBABILITY: %s" % self.value)
        return Reply(text=text, input_tokens=count_tokens_estimate(prompt),
                     output_tokens=count_tokens_estimate(text),
                     finish_reason="length" if rid in self.length_rows else "stop",
                     provider="fake", served_model=self.name + ("-other" if rid in self.served_rows else ""),
                     raw={"fake": True})


    def probe(self) -> bool:
        return not (self.outage_file and os.path.exists(self.outage_file))


class _FakeResponse:
    status_code = 429

    def __init__(self, retry_after):
        self.headers = {"content-type": "application/json"}
        if retry_after is not None:
            self.headers["retry-after"] = retry_after

    @staticmethod
    def json():
        return {"error": {"code": 429, "message": "Provider returned error",
                          "metadata": {"provider_name": "FakeHost", "raw": "fake host rate limit"}}}


class FakeBadRequest(Exception):
    """Looks like LiteLLM's BadRequestError to the classifier: HTTP 400, which
    classify_error calls permanent. The harness must recognise the output-limit message
    inside it and fail only the row, not the run."""
    status_code = 400


class FakeRateLimit(Exception):
    """Looks like LiteLLM's RateLimitError to the classifier: status 429, a response with a
    Retry-After header (or none), and an OpenRouter-style body."""
    status_code = 429

    def __init__(self, retry_after="3"):
        super().__init__("litellm.RateLimitError: fake - Provider returned error (429)")
        self.response = _FakeResponse(retry_after)


def _ids(s: str):
    return [int(x) for x in s.split("+") if x]


def build(model_arg: str) -> Adapter:
    parts = [p for p in (model_arg or "").split(":") if p != ""]
    value = float(parts[0]) if parts else 0.5
    kw = {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        if k == "rate429_rows" and v == "all":
            kw[k] = "all"
        elif k in ("refuse_rows", "refuse_first", "raise_rows", "raise_first", "bare_rows", "permanent_rows",
                   "served_rows", "length_rows", "rate429_rows", "outlimit_rows", "outlimit_both"):
            kw[k] = _ids(v)
        elif k in ("outage_file", "rate429_retry_after", "sleep_s"):
            kw[k] = v
        elif k == "rate429_until_call":
            kw[k] = int(v)
        else:
            raise ValueError("unknown fake option %r" % p)
    return FakeAdapter(value, **kw)
