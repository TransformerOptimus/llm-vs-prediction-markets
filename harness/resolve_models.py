"""Resolve every roster model without spending a cent.

For each non-fake entry in models.json: check the key is in .env, ask LiteLLM which
provider the id maps to, build our request parameters, and then send the real request
through LiteLLM to a local capture server (127.0.0.1) with a dummy key, so the exact
body LiteLLM builds (model name, reasoning cap, provider order, fallbacks off,
temperature, token limits) can be read back. No request leaves this machine and the real
keys never leave the process. The capture server answers with a canned reply whose model
string is the entry's served_model, which also exercises the served-checkpoint check.

  python3 harness/resolve_models.py            every roster model
  python3 harness/resolve_models.py <id> ...   only these LiteLLM ids
Exit status 1 if any model fails to resolve for a reason other than a missing key.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import adapters  # noqa: E402
from adapters.base import MODELS_FILE, PRICES_FILE, load_table  # noqa: E402

CAPTURED: List[Dict] = []
CANNED = {"served_model": "unknown", "provider": None}


class _Capture(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"_raw": body[:2000].decode("utf-8", "replace")}
        CAPTURED.append({"path": self.path, "header_names": sorted(k.lower() for k in self.headers.keys()),
                         "body": parsed})   # header values (the dummy key) are never stored
        reply = {"id": "dry-run", "object": "chat.completion", "created": 0, "model": CANNED["served_model"],
                 "provider": CANNED["provider"],
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "PROBABILITY: 0.5"},
                              "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17,
                           "completion_tokens_details": {"reasoning_tokens": 0}}}
        out = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # keep the console clean
        pass


def start_server() -> str:
    srv = HTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return "http://127.0.0.1:%d" % srv.server_address[1]


def resolve_one(model_id: str, entry: Dict, prices: Dict, api_base: str) -> Dict:
    out: Dict = {"model": model_id, "set": entry.get("set"), "host": entry.get("host"),
                 "resolved": False, "reason": None, "key": None, "litellm_provider": None,
                 "request_params": None, "sent_body": None, "checks": {}, "in_litellm_cost_map": None}
    out["priced"] = model_id in prices and prices[model_id].get("input") is not None
    try:
        adapter = adapters.get_adapter("litellm:" + model_id)
    except RuntimeError as e:      # "... is not set in .env"
        out["reason"] = "key missing: %s" % e
        out["key"] = "missing"
        return out
    out["key"] = "present"
    import litellm
    try:
        _, provider, _, _ = litellm.get_llm_provider(model_id)
        out["litellm_provider"] = provider
    except Exception as e:
        out["reason"] = "LiteLLM does not recognise the id: %s" % e
        return out
    out["in_litellm_cost_map"] = model_id in litellm.model_cost or model_id.split("/", 1)[1] in litellm.model_cost
    params = adapter.request_params()
    out["request_params"] = params
    CANNED["served_model"] = adapter.expected_served_model
    CANNED["provider"] = adapter.host_name
    n_before = len(CAPTURED)
    try:
        resp = adapter.dry_request("Dry run. Reply with PROBABILITY: 0.5", api_base)
    except Exception as e:
        out["reason"] = "LiteLLM raised before or while sending: %s: %s" % (e.__class__.__name__, str(e)[:300])
        out["sent_body"] = CAPTURED[n_before]["body"] if len(CAPTURED) > n_before else None
        return out
    if len(CAPTURED) <= n_before:
        out["reason"] = "no request reached the capture server"
        return out
    cap = CAPTURED[-1]
    body = cap["body"]
    out["sent_body"] = body
    out["sent_path"] = cap["path"]
    upstream = model_id.split("/", 1)[1]           # what the provider receives: gpt-5.2, z-ai/glm-4.7, ...
    thinking_on = bool(params.get("reasoning_effort") or params.get("thinking")
                       or (params.get("extra_body") or {}).get("reasoning"))
    checks = {"model_in_body": body.get("model") == upstream,
              "no_tools": "tools" not in body and "tool_choice" not in body,
              "one_user_message": [m.get("role") for m in body.get("messages", [])] == ["user"],
              # the adapter's rule: temperature is omitted when thinking is on for direct openai/,
              # anthropic/ and gemini/ ids (they reject it); openrouter/ ids get temperature 0 beside
              # the reasoning budget (the open models accept both)
              "temperature_rule": (body.get("temperature") == adapter.temperature) if adapter.provider == "openrouter"
                                  else ((not thinking_on) or ("temperature" not in body)),
              "served_model_matches": resp.get("model") == adapter.expected_served_model}
    if adapter.provider == "openrouter":
        prov = body.get("provider") or {}
        checks["reasoning_max_tokens"] = (body.get("reasoning") or {}).get("max_tokens") == adapter.reasoning_tokens
        checks["provider_order_pinned"] = prov.get("order") == [adapter.host]
        checks["fallbacks_off"] = prov.get("allow_fallbacks") is False
        checks["max_tokens"] = body.get("max_tokens") == adapter.answer_tokens
    elif adapter.provider == "openai":
        checks["reasoning_effort"] = body.get("reasoning_effort") == adapter.reasoning_effort
        checks["max_completion_tokens"] = body.get("max_completion_tokens") == adapter.answer_tokens
    out["checks"] = checks
    out["resolved"] = all(checks.values())
    if not out["resolved"]:
        out["reason"] = "checks failed: %s" % [k for k, v in checks.items() if not v]
    return out


def main():
    models = load_table(MODELS_FILE)
    prices = load_table(PRICES_FILE)
    ids = sys.argv[1:] or [m for m in models if not m.startswith("fake")]
    api_base = start_server()
    results = []
    for mid in ids:
        if mid not in models:
            results.append({"model": mid, "resolved": False, "reason": "not in models.json"})
            continue
        r = resolve_one(mid, models[mid], prices, api_base)
        results.append(r)
        print("=" * 78)
        print("%s  [%s, host %s]  -> %s" % (mid, r.get("set"), r.get("host"),
                                             "RESOLVED" if r["resolved"] else "NOT RESOLVED: %s" % r["reason"]))
        if r.get("key") == "present":
            print("  key: present   LiteLLM provider: %s   in LiteLLM cost map: %s   priced in prices.json: %s"
                  % (r["litellm_provider"], r["in_litellm_cost_map"], r["priced"]))
            print("  request_params: %s" % json.dumps(r["request_params"], sort_keys=True))
            if r.get("sent_body") is not None:
                body = dict(r["sent_body"])
                body.pop("messages", None)
                print("  body sent (minus messages), to %s: %s" % (r.get("sent_path"), json.dumps(body, sort_keys=True)))
            print("  checks: %s" % json.dumps(r["checks"], sort_keys=True))
    report = os.path.join(HERE, "runs", "resolve_report.json")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 78)
    print("report written to %s" % report)
    bad = [r["model"] for r in results if not r["resolved"] and r.get("key") != "missing"]
    if bad:
        raise SystemExit("not resolved: %s" % ", ".join(bad))


if __name__ == "__main__":
    main()
