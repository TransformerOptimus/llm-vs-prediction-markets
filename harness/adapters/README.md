# Adapters: the connection to model providers

An adapter takes one prompt and returns one reply, together with token counts,
the reason the reply ended, and which host and model actually answered. It
does nothing else: no web access, no search, no tools.

| File | What it does |
|---|---|
| `base.py` | What every adapter must provide, plus the shared settings: the thinking cap (how many tokens a model may spend thinking before it answers) and the reply limit. Also reads `prices.json` and `models.json`. |
| `litellm_adapter.py` | The one real adapter. It reaches every provider through the LiteLLM library: OpenAI directly, and open-weight models through OpenRouter, pinned to one host each. Model ids look like `litellm:openai/gpt-5.2` or `litellm:openrouter/moonshotai/kimi-k2.5`. API keys are read from `.env` and never printed or logged. |
| `fake.py` | A fake model for tests. It answers a fixed number, costs nothing, calls nothing, and can imitate problems on chosen rows (refusals, network errors, rate limits, cut-off replies, outages) so the harness's handling of each can be tested. Example: `fake:0.37` always answers 0.37. |
| `__init__.py` | Finds every adapter file in this folder and builds the right one from a model id such as `fake:0.5` or `litellm:openai/gpt-5.2`. |

To add a provider, copy `fake.py` or `litellm_adapter.py` to a new file here,
give it a new short `PROVIDER` name and a `build()` function. `__init__.py`
picks it up automatically.
