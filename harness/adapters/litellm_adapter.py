"""The one real adapter: every provider through LiteLLM.

Spec: litellm:<LiteLLM model id>, for example
  litellm:openai/gpt-4.1-mini        key OPENAI_API_KEY
  litellm:anthropic/claude-opus-4-5  key ANTHROPIC_API_KEY
  litellm:gemini/gemini-2.5-pro      key GEMINI_API_KEY
  litellm:openrouter/<vendor/model>  key OPENROUTER_API_KEY
The adapter name (the key into prices.json and models.json) is the LiteLLM id itself.
Keys come from .env only; they are placed in the process environment for LiteLLM and
never printed or logged. One user message, no tools, no system prompt.

Thinking cap N, per provider as LiteLLM expects it:
  anthropic/, gemini/   thinking={"type": "enabled", "budget_tokens": N}; temperature omitted
                        (Anthropic rejects a temperature with thinking on)
  openai/               reasoning_effort=<effort named in models.json>; temperature omitted
  openrouter/           extra_body={"reasoning": {"max_tokens": N}}; temperature 0 is sent beside it
                        (a host that rejects the pair returns HTTP 400 = permanent error; some hosts do)
max_tokens = answer_tokens (4,000; max_completion_tokens on the openai/ path). The thinking cap is a separate
request; hosts differ on whether thinking counts inside max_tokens.

Host pinning: for openrouter/ ids the host slug in models.json is sent as
provider={"order": [host], "allow_fallbacks": False} and the response's provider string
is logged. --host-override in run.py may swap in the entry's backup host (recorded).

Served-checkpoint check: `expected_served_model` is the served_model string from
models.json (for direct openai/ ids the id without its prefix). forecast.py compares
every reply's model string with it; a mismatch is a permanent error that stops the run.

Every reply carries the exact request parameters sent and the model string returned.
LiteLLM's own cost estimate per call is recorded next to ours (prices.json); the budget
stop uses ours.
"""
import os
from typing import Dict, Optional

from .base import MODELS_FILE, Adapter, Reply, load_env, load_table, ratelimit_headers

PROVIDER = "litellm"
ENV_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")
KEY_FOR_PREFIX = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                  "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def litellm_version() -> Optional[str]:
    try:
        import importlib.metadata as im
        return im.version("litellm")
    except Exception:
        return None


class LiteLLMAdapter(Adapter):
    is_paid = True

    def __init__(self, model_id: str):
        import litellm  # imported here so the fake model needs no SDK
        self.litellm = litellm
        litellm.drop_params = False          # a rejected parameter must fail loudly, not vanish
        litellm.suppress_debug_info = True
        self.model_id = model_id
        self.name = model_id
        self.provider = model_id.split("/", 1)[0]
        env = load_env()
        for k in ENV_KEYS:
            if env.get(k):
                os.environ[k] = env[k]
        entry = load_table(MODELS_FILE).get(model_id) or {}
        self.entry = entry
        # host kinds: 'direct' (openai/, anthropic/, gemini/ ids), OpenRouter (openrouter/ ids), and
        # 'direct_compatible': any OpenAI-compatible endpoint named in models.json (api_base, key_env,
        # upstream_model), called through LiteLLM's openai/ path with that base URL and key
        self.host_kind = entry.get("host_kind") or ("openrouter" if self.provider == "openrouter" else "direct")
        self.api_base = None
        self.api_key = None
        if self.host_kind == "direct_compatible":
            for f in ("api_base", "key_env", "upstream_model"):
                if not entry.get(f):
                    raise RuntimeError("%s: direct_compatible entry needs %s in models.json" % (model_id, f))
            self.provider = "direct_compatible"
            self.api_base = entry["api_base"]
            self.litellm_model = "openai/" + entry["upstream_model"]
            key = env.get(entry["key_env"])
            if not key:
                raise RuntimeError("%s is not set in .env; cannot use %s" % (entry["key_env"], model_id))
            self.api_key = key
        else:
            self.litellm_model = model_id
            need = KEY_FOR_PREFIX.get(self.provider)
            if need and not os.environ.get(need):
                raise RuntimeError("%s is not set in .env; cannot use %s" % (need, model_id))
        self.host = entry.get("host")               # 'direct' or the pinned OpenRouter host slug
        self.host_name = entry.get("host_name")     # the display name OpenRouter returns as provider
        if entry.get("reasoning_effort"):
            self.reasoning_effort = entry["reasoning_effort"]
        if entry.get("reasoning_tokens") is not None:   # the thinking cap N from models.json; run.py flags override
            self.reasoning_tokens = entry["reasoning_tokens"]
        self.expected_served_model = entry.get("served_model")
        if not self.expected_served_model and self.host_kind == "direct":
            self.expected_served_model = model_id.split("/", 1)[1]
        if not self.expected_served_model and self.host_kind == "direct_compatible":
            self.expected_served_model = entry["upstream_model"]
        self.host_override_kind = None

    def use_backup_host(self, slug: str, allow_any: bool = False) -> None:
        """--host-override: only the entry's recorded backup host may replace the pin, except in a
        smoke run (allow_any), where any OpenRouter host slug may be tried; the kind is recorded."""
        backup = self.entry.get("backup_host") or {}
        if self.provider != "openrouter":
            raise ValueError("--host-override applies to openrouter/ ids only; %s is %s" % (self.model_id, self.host_kind))
        if backup and slug == backup.get("host"):
            self.host, self.host_name = backup["host"], backup.get("host_name")
            self.host_override_kind = "backup"
            return
        if not allow_any:
            raise ValueError("--host-override %r is not the recorded backup host for %s (backup_host: %s)"
                             % (slug, self.model_id, backup.get("host")))
        self.host, self.host_name = slug, slug
        self.host_override_kind = "non-backup host, smoke test only"

    def dry_request(self, prompt: str, api_base: str) -> Dict:
        """Resolve without spending: send the real request through LiteLLM to a local
        capture server with a dummy key, so the exact body LiteLLM builds can be read
        back. The real key never leaves the process. Returns LiteLLM's response object
        as a dict (the capture server's canned reply)."""
        params = self.request_params()
        resp = self.litellm.completion(messages=[{"role": "user", "content": prompt}], num_retries=0,
                                       timeout=30.0, api_base=api_base, api_key="dry-run-no-key", **params)
        return resp.model_dump()

    def request_params(self) -> Dict:
        """Exactly what is sent, minus the messages."""
        total = self.answer_tokens          # the reply limit; the thinking cap is sent separately
        kw: Dict = {"model": self.litellm_model}
        thinking_on = False
        if self.host_kind == "direct_compatible":
            # an OpenAI-compatible open-model endpoint: max_tokens, temperature, and the host's own
            # thinking switch from models.json (thinking_extra_body), which may or may not take a budget
            kw["max_tokens"] = total
            extra = dict(self.entry.get("thinking_extra_body") or {})
            if self.entry.get("thinking_budget_field") and self.reasoning_tokens is not None:
                extra[self.entry["thinking_budget_field"]] = self.reasoning_tokens
            if self.reasoning_effort is not None and self.entry.get("reasoning_effort_supported"):
                kw["reasoning_effort"] = self.reasoning_effort
            if extra:
                kw["extra_body"] = extra
        elif self.provider in ("anthropic", "gemini"):
            kw["max_tokens"] = total
            if self.reasoning_tokens is not None:
                kw["thinking"] = {"type": "enabled", "budget_tokens": self.reasoning_tokens}
                thinking_on = True
        elif self.provider == "openai":
            kw["max_completion_tokens"] = total
            if self.reasoning_effort is not None:
                kw["reasoning_effort"] = self.reasoning_effort
                thinking_on = True
        else:  # openrouter and anything else OpenAI-shaped
            kw["max_tokens"] = total
            extra: Dict = {}
            if self.reasoning_tokens is not None:
                extra["reasoning"] = {"max_tokens": self.reasoning_tokens}
            elif self.reasoning_effort is not None:      # models that take an effort level, not a budget (gpt-oss)
                extra["reasoning"] = {"effort": self.reasoning_effort}
            if self.provider == "openrouter" and self.host and self.host != "direct":
                extra["provider"] = {"order": [self.host], "allow_fallbacks": False}
                if getattr(self, "host_quantization", None):     # smoke runs: pick the host's endpoint by precision
                    extra["provider"]["quantizations"] = [self.host_quantization]
            if extra:
                kw["extra_body"] = extra
        if self.temperature is not None and not thinking_on:
            kw["temperature"] = self.temperature
        return kw

    def complete(self, prompt: str, meta: Optional[Dict] = None) -> Reply:
        params = self.request_params()
        conn = {"api_base": self.api_base, "api_key": self.api_key} if self.host_kind == "direct_compatible" else {}
        resp = self.litellm.completion(messages=[{"role": "user", "content": prompt}],
                                       num_retries=0, timeout=180.0, **conn, **params)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        hidden = getattr(resp, "_hidden_params", {}) or {}
        provider = None
        try:
            raw = resp.model_dump()
            provider = raw.get("provider") or (raw.get("model_extra") or {}).get("provider")
        except Exception:
            pass
        if provider is None and self.provider != "openrouter":
            provider = self.host_name or self.provider
        try:
            litellm_cost = float(self.litellm.completion_cost(completion_response=resp))
        except Exception:
            litellm_cost = hidden.get("response_cost")
        rl_headers = ratelimit_headers(hidden.get("additional_headers"))
        return Reply(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=choice.finish_reason,
            provider=provider,
            served_model=getattr(resp, "model", None),
            raw={"id": getattr(resp, "id", None), "model": getattr(resp, "model", None),
                 "provider": provider, "finish_reason": choice.finish_reason,
                 "reasoning_tokens_used": getattr(details, "reasoning_tokens", None),
                 "request_params": params, "pinned_host": self.host,
                 "litellm_cost_usd": litellm_cost, "litellm_version": litellm_version(),
                 "rate_limit_headers": rl_headers or None},
        )

    @property
    def probe_url(self) -> str:
        """The base URL a HEAD request can reach to tell 'network down' from 'call failed'."""
        if self.api_base:
            return self.api_base
        return {"openai": "https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1",
                "gemini": "https://generativelanguage.googleapis.com"}.get(self.provider, "https://openrouter.ai/api/v1")

    def probe(self) -> bool:
        """True when the host answers at all (any HTTP status counts; only a transport
        failure means down). Sends no key and costs nothing."""
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.probe_url, method="HEAD")
        try:
            urllib.request.urlopen(req, timeout=10)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False


def build(model_arg: str) -> Adapter:
    if not model_arg or "/" not in model_arg:
        raise ValueError("model id required, for example litellm:openai/gpt-4.1-mini")
    return LiteLLMAdapter(model_arg)
