"""The one place the exploratory scripts learn which models exist.

The model lists come from one source: `harness/models.json`, the study's model roster. Add a model
there and run it, and every script here picks it up.

What it exposes:

  SCORED      the models whose results are reported: the three-window set (roster label "bridge") and
              the Windows B and C set (roster label "2026"), in roster order.
              The probe-validation model (never scored) and the fake fixture used
              by the tests are not in it.
  BY_SET      {"bridge": [...], "2026": [...]}, for scripts that contrast the two sets.
  ROSTER      window -> the scored models the roster runs on that window ("A" holds the three-window
              set only, because the Windows B and C models were released after Window A's outcomes
              were known).
  SHORT       model id -> short display name. Looking up a model that is not in the roster returns the
              last part of its id instead of raising, so an unknown name can never stop a script.
  VALIDATION  the probe-validation model id, for the `model != ?` filters in the loaders.
  NOT_SCORED  every roster entry that is not scored (the validation model and the fixture).
  short(m)    the same lookup as SHORT, as a function.

Nothing here changes any measure, threshold or filter: it only says which models the scripts read.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROSTER_FILE = os.path.join(REPO, "harness", "models.json")

SCORED_SETS = ("bridge", "2026")
WINDOWS = ("A", "B", "C")


class _Names(dict):
    """Display names that never raise: an unknown model shows as the last part of its id."""

    def __missing__(self, key):
        return str(key).split("/")[-1]


def _load():
    with open(ROSTER_FILE) as f:
        raw = json.load(f)
    scored, by_set, not_scored, windows_of = [], {s: [] for s in SCORED_SETS}, [], {}
    for model, spec in raw.items():
        if not isinstance(spec, dict) or "set" not in spec:
            continue                      # the file's own notes
        if spec["set"] in SCORED_SETS:
            scored.append(model)
            by_set[spec["set"]].append(model)
            windows_of[model] = list(spec.get("windows") or [])
        else:
            not_scored.append((model, spec["set"]))
    return raw, scored, by_set, not_scored, windows_of


RAW, SCORED, BY_SET, _NOT_SCORED, WINDOWS_OF = _load()
NOT_SCORED = frozenset(m for m, _ in _NOT_SCORED)
VALIDATION = next((m for m, s in _NOT_SCORED if s == "validation"), "")
SHORT = _Names((m, m.split("/")[-1]) for m in SCORED)
ROSTER = {w: [m for m in SCORED if w in WINDOWS_OF.get(m, ())] for w in WINDOWS}
BRIDGE = BY_SET["bridge"]
SET2026 = BY_SET["2026"]


def short(model: str) -> str:
    return SHORT[model]


def set_of(model: str) -> str:
    """"bridge", "2026", or whatever the roster file says; "" for a model it does not list."""
    spec = RAW.get(model)
    return spec.get("set", "") if isinstance(spec, dict) else ""


if __name__ == "__main__":
    print("roster file:", ROSTER_FILE)
    print("scored models (%d):" % len(SCORED))
    for m in SCORED:
        print("   %-40s %-7s windows %s" % (m, set_of(m), ",".join(WINDOWS_OF[m])))
    print("not scored:", ", ".join("%s (%s)" % (m, s) for m, s in _NOT_SCORED) or "none")
    for w in WINDOWS:
        print("window %s: %s" % (w, ", ".join(SHORT[m] for m in ROSTER[w])))
