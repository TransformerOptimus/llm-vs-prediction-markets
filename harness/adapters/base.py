"""What every model adapter must provide.

An adapter turns one prompt string into one reply string, and reports token counts,
the finish reason, and which provider and model string actually served the call. It
must not do anything else: no web access, no retrieval, no tools. The only network
call allowed is the one to the model provider.

Two settings: `reasoning_tokens` is the thinking cap N
(1,000), sent as a request that some hosts honour and some do not; `answer_tokens`
(4,000) is the reply limit sent as max_tokens. Hosts differ on whether thinking counts
inside that limit (OpenAI, DeepInfra, AtlasCloud, Novita: inside; SiliconFlow: on top).

To add a model: copy fake.py or litellm_adapter.py to a new file in this folder,
set PROVIDER to a new short name, and implement build(model_arg). Nothing else
changes; adapters/__init__.py finds every file here automatically.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.dirname(HERE)
PRICES_FILE = os.path.join(HARNESS_DIR, "prices.json")
MODELS_FILE = os.path.join(HARNESS_DIR, "models.json")
PROVIDERS_FILE = os.path.join(HARNESS_DIR, "providers.json")


@dataclass
class Reply:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None      # "stop", "length", ... as the provider reports it
    provider: Optional[str] = None           # the host that served the call (OpenRouter's provider field)
    served_model: Optional[str] = None       # the model string in the response
    raw: Dict = field(default_factory=dict)  # provider response details, minus anything secret


class Adapter:
    name: str = "unnamed"          # "provider:model"; the key into prices.json, models.json, providers.json
    provider: str = "unnamed"      # short provider name, for the per-provider rate limit
    is_paid: bool = True           # run.py refuses paid adapters without explicit approval
    reasoning_tokens: Optional[int] = None   # thinking cap N; None = provider default
    reasoning_effort: Optional[str] = None   # openai path only ("low", "medium", "high")
    answer_tokens: int = 4000                # the reply limit sent as max_tokens (4,000; hosts differ on whether thinking counts inside it)
    temperature: Optional[float] = 0.0       # None = do not send the parameter

    def complete(self, prompt: str, meta: Optional[Dict] = None) -> Reply:
        raise NotImplementedError

    def settings(self) -> Dict:
        return {"reasoning_tokens": self.reasoning_tokens, "reasoning_effort": self.reasoning_effort,
                "answer_tokens": self.answer_tokens, "temperature": self.temperature}

    # Cost: dollars per million tokens, looked up in harness/prices.json by "provider:model".
    def price_per_million(self) -> Optional[Dict]:
        p = load_table(PRICES_FILE).get(self.name)
        if not p or p.get("input") is None or p.get("output") is None:
            return None
        return p

    @staticmethod
    def provider_cost(reply: Reply) -> Optional[float]:
        """LiteLLM's own estimate from the provider's usage counts, when the adapter recorded one."""
        v = (reply.raw or {}).get("litellm_cost_usd")
        return float(v) if v is not None else None

    def cost_estimate(self, reply: Reply) -> Optional[float]:
        p = self.price_per_million()
        if not p or reply.input_tokens is None or reply.output_tokens is None:
            return None
        return (reply.input_tokens * p["input"] + reply.output_tokens * p["output"]) / 1e6


TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
PERMANENT_STATUS = {400, 401, 402, 403, 404, 413, 422}
_TRANSIENT_WORDS = re.compile(r"temporarily rate-?limited|rate ?limit|overloaded|timed? ?out|connection (reset|error|refused)|"
                              r"service unavailable|bad gateway|gateway time", re.I)
_PERMANENT_WORDS = re.compile(r"not available|unavailable|not found|does not exist|no such model|gated|"
                              r"only available on|requires? (a |an )?(different|another) client|not supported|"
                              r"agentic harness|invalid api key|insufficient credits|unsupported", re.I)


_TRANSIENT_TYPES = {"RateLimitError", "Timeout", "APITimeoutError", "APIConnectionError", "ServiceUnavailableError",
                    "InternalServerError", "BadGatewayError", "ConnectionError", "TimeoutError"}
_PERMANENT_TYPES = {"AuthenticationError", "NotFoundError", "BadRequestError", "PermissionDeniedError",
                    "UnprocessableEntityError", "ContentPolicyViolationError", "ContextWindowExceededError",
                    "BudgetExceededError", "UnsupportedParamsError", "InvalidRequestError", "PermanentError"}


# A reply the host refused because it ran past the reply limit, instead of sending the
# truncated text back. Seen on direct OpenAI with gpt-5.5: the same call on
# the same row hit the limit four times and came back three times as an ordinary truncated
# reply (finish_reason "length") and once as this error. It is about this row's answer
# being long, not about the request, so it must not be treated as permanent.
_OUTPUT_LIMIT_WORDS = re.compile(
    r"could not finish the message because max_tokens|max_tokens or model output limit|"
    r"output limit was reached|try again with (a )?higher max_tokens", re.I)


def is_output_limit_error(exc: Exception) -> bool:
    """True when the host refused a reply for running past the reply limit.

    classify_error would call this permanent, because it arrives as a BadRequestError,
    and the run would stop. It is a row-level problem: the caller hands it to the strict
    retry (which asks for a bare number, so the reply is short) and fails only this row
    if that fails too.
    """
    try:
        msg = str(exc) or exc.__class__.__name__
    except Exception:
        msg = exc.__class__.__name__
    return bool(_OUTPUT_LIMIT_WORDS.search(msg))


def classify_error(exc: Exception) -> str:
    """'transient' (retry with backoff) or 'permanent' (never retry; the same call would
    fail on every row). Order: OpenRouter's "temporarily rate-limited upstream" is always
    transient; then the exception type (LiteLLM's RateLimitError, Timeout, APIConnectionError,
    ServiceUnavailableError, InternalServerError are transient; AuthenticationError,
    NotFoundError, BadRequestError, PermissionDeniedError, UnprocessableEntityError and the
    like are permanent); then the HTTP status; then the message; unknown errors count as
    transient."""
    try:
        msg = str(exc) or exc.__class__.__name__
    except Exception:
        msg = exc.__class__.__name__
    if re.search(r"temporarily rate-?limited", msg, re.I):
        return "transient"
    name = exc.__class__.__name__
    if name in _TRANSIENT_TYPES:
        return "transient"
    if name in _PERMANENT_TYPES:
        return "permanent"
    status = getattr(exc, "status_code", None)
    if status in TRANSIENT_STATUS:
        return "transient"
    if status in PERMANENT_STATUS:
        return "permanent"
    if _PERMANENT_WORDS.search(msg):
        return "permanent"
    return "transient"


_CONNECTION_TYPES = {"APIConnectionError", "Timeout", "APITimeoutError", "ConnectionError", "TimeoutError",
                     "ConnectError", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "ReadError",
                     "WriteError", "RemoteProtocolError", "NetworkError", "gaierror", "ConnectionRefusedError",
                     "ConnectionResetError", "BrokenPipeError"}
_CONNECTION_WORDS = re.compile(r"connection (error|reset|refused|aborted)|name or service not known|nodename nor servname|"
                               r"temporary failure in name resolution|network is unreachable|no route to host|"
                               r"timed? ?out|read timeout|connect timeout|remote end closed|server disconnected|"
                               r"getaddrinfo failed|failed to establish", re.I)


def is_connection_error(exc: Exception) -> bool:
    """A connection-level failure (DNS, connect, read timeout, reset): the request never
    got an HTTP answer. These are the errors that, seen on every in-flight worker at
    once, mean the network is down rather than the row is bad. LiteLLM wraps them in
    an APIError with a made-up status 500, so the chain (__cause__ / __context__) is
    checked first: httpx.ConnectError, socket.gaierror and the like sit two or three
    levels down. A real 4xx answer is never a connection error."""
    status = getattr(exc, "status_code", None)
    if status is not None and 400 <= int(status) < 500:
        return False
    try:
        msg = str(exc)
    except Exception:
        msg = ""
    cur, depth = exc, 0
    while cur is not None and depth < 6:
        if cur.__class__.__name__ in _CONNECTION_TYPES:
            return True
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return bool(_CONNECTION_WORDS.search(msg))


def is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 429 or exc.__class__.__name__ == "RateLimitError"


def _responses_in_chain(exc: Exception):
    """LiteLLM's mapped exception carries an empty response; the SDK or httpx error two
    levels down (__cause__ / __context__) still has the real headers and body."""
    cur, depth = exc, 0
    while cur is not None and depth < 8:
        r = getattr(cur, "response", None)
        if r is not None and hasattr(r, "headers"):
            yield cur, r
        cur = cur.__cause__ or cur.__context__
        depth += 1


def retry_after_from(exc: Exception) -> Optional[float]:
    """Seconds from a Retry-After header anywhere in the exception chain, else None
    (an HTTP-date value is ignored; the header is almost always seconds)."""
    for _, r in _responses_in_chain(exc):
        v = None
        try:
            v = r.headers.get("retry-after") or r.headers.get("Retry-After")
        except Exception:
            continue
        if v:
            try:
                return float(v)
            except ValueError:
                return None
    return None


def rate_limit_info(exc: Exception) -> Dict:
    """What a 429 said: the Retry-After value and the error body's metadata block
    (OpenRouter puts provider_name and the host's raw message there, which tells an
    OpenRouter-side limit from a host-side one)."""
    info: Dict = {"status": getattr(exc, "status_code", None), "retry_after": retry_after_from(exc),
                  "error_type": None, "provider_code": None, "provider_name": None, "raw": None}
    body = None
    for holder, r in _responses_in_chain(exc):
        b = getattr(holder, "body", None)
        if isinstance(b, dict):
            body = b
            break
        try:
            body = r.json()
            break
        except Exception:
            continue
    if body is None:
        try:
            m = re.search(r"\{.*\}", str(exc), re.S)
            body = json.loads(m.group(0)) if m else None
        except Exception:
            body = None
    err = (body or {}).get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict) and isinstance(body, dict):
        err = body
    if isinstance(err, dict):
        meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
        info["error_type"] = err.get("type") or meta.get("error_type")
        info["provider_code"] = meta.get("provider_code") or err.get("code")
        info["provider_name"] = meta.get("provider_name")
        info["raw"] = (meta.get("raw") or err.get("message") or "")[:300] or None
    return info


def ratelimit_headers(headers: Optional[Dict]) -> Dict:
    """The x-ratelimit-* headers out of LiteLLM's additional_headers (it repeats them
    with an llm_provider- prefix; the prefixed copy carries the reset values)."""
    out: Dict = {}
    for k, v in (headers or {}).items():
        key = k.lower()
        if key.startswith("llm_provider-"):
            key = key[len("llm_provider-"):]
        if "ratelimit" in key:
            out[key] = v
    return out


class PermanentError(Exception):
    """Raised by adapters (or tests) for an error that must not be retried."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def load_table(path: str) -> Dict:
    """A JSON object file (prices, models, providers). Keys starting with "_" are notes."""
    try:
        with open(path) as f:
            table = json.load(f)
    except FileNotFoundError:
        return {}
    return {k: v for k, v in table.items() if not k.startswith("_")}


def load_env(path: Optional[str] = None) -> Dict[str, str]:
    """Read KEY=value lines from the repo-root .env. Values are never logged or printed."""
    path = path or os.path.join(os.path.dirname(HARNESS_DIR), ".env")
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def count_tokens_estimate(text: str) -> int:
    """Local token estimate for adapters that do not report usage (the fake model)."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)
