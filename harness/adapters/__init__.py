"""Find every adapter file in this folder and build one from a spec string.

A spec is "<provider>[:<model or options>]", for example "fake:0.5" or
"litellm:openai/gpt-4.1-mini". Each adapter file declares PROVIDER (a name or a tuple of
names) and a build(model_arg[, provider]) function.
"""
import importlib
import inspect
import pkgutil
from typing import Dict

from .base import Adapter, Reply  # noqa: F401

_REGISTRY: Dict[str, object] = {}


def _load():
    if _REGISTRY:
        return
    for m in pkgutil.iter_modules(__path__):
        if m.name == "base":
            continue
        mod = importlib.import_module("%s.%s" % (__name__, m.name))
        providers = getattr(mod, "PROVIDER", None)
        if not providers:
            continue
        if isinstance(providers, str):
            providers = (providers,)
        for p in providers:
            _REGISTRY[p] = mod


def available() -> Dict[str, str]:
    _load()
    return {p: mod.__name__ for p, mod in sorted(_REGISTRY.items())}


def get_adapter(spec: str) -> Adapter:
    _load()
    provider, _, model_arg = spec.partition(":")
    if provider not in _REGISTRY:
        raise ValueError("unknown provider %r; known: %s" % (provider, ", ".join(sorted(_REGISTRY))))
    build = _REGISTRY[provider].build
    if "provider" in inspect.signature(build).parameters:
        return build(model_arg, provider=provider)
    return build(model_arg)
