"""Hand-computed checks of the memory-probe quarantine rule and the calibration gate in analysis/gates.py.
Run: python3 analysis/test_gates.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates
from gates import ece, passes, quarantine_rule as rule, verdict


def close(a, b, tol=1e-9):
    assert abs(a - b) < tol, (a, b)


# --- quarantine rule: quarantined iff |p - y| <= 0.10 and |q - y| >= 0.30 ---------------------
assert rule(0.95, 0.55, 1.0)[0] == 1          # memorized-looking: 0.95 on the winner, market at 0.55
assert rule(0.95, 0.90, 1.0)[0] == 0          # easy favorite: market already at 0.90
assert rule(0.60, 0.50, 1.0)[0] == 0          # right but not confident
assert rule(0.95, 0.55, 0.0)[0] == 0          # confidently wrong: noise, not memory
assert rule(0.05, 0.60, 0.0)[0] == 1          # the No side mirrors
assert rule(0.90, 0.70, 1.0)[0] == 1          # boundaries inclusive
assert rule(0.89, 0.70, 1.0)[0] == 0
assert rule(0.90, 0.71, 1.0)[0] == 0

# quarantine_flags never sees a failed record (the loader drops them); flags keep the column shape.
flags = gates.quarantine_flags([{"moment_key": "k2", "row_id": 2, "p": 0.95, "q": 0.55, "outcome_yes": 1.0, "record_source": "memory_probe"},
                                {"moment_key": "k1", "row_id": 1, "p": 0.50, "q": 0.55, "outcome_yes": 1.0, "record_source": "normal_as_probe"}],
                               "fake:test", "run-x", flagged_at="2026-01-01T00:00:00Z")
assert [f["moment_key"] for f in flags] == ["k1", "k2"]                      # sorted by moment_key
assert [f["quarantined"] for f in flags] == [0, 1]
assert set(flags[0]) == {"moment_key", "row_id", "model_name", "quarantined", "reason", "flagged_at", "probe_run_id", "record_source"}

# --- calibration gate: 3 equal-count bins, ECE = weighted mean |mean p - outcome rate| --------
e, bins = ece([(0.1, 0), (0.2, 0), (0.3, 1), (0.6, 1), (0.7, 0), (0.9, 1)])
close(e, 1.0 / 3.0)
assert [b["n"] for b in bins] == [2, 2, 2]
close(bins[0]["gap"], 0.15)
close(bins[1]["gap"], 0.55)
close(bins[2]["gap"], 0.30)

e, _ = ece([(0.2, 0), (0.2, 0), (0.2, 0), (0.2, 0), (0.2, 1), (0.8, 1), (0.8, 1), (0.8, 1), (0.8, 1), (0.8, 0)][:10], k=2)
close(e, 0.0)

# A bin never splits a tied probability, whatever order the rows arrive in.
for shift in range(0, 30, 7):
    tied = [(0.70, 1)] * 21 + [(0.70, 0)] * 9
    tied = tied[shift:] + tied[:shift]
    e, bins = ece(tied)
    close(e, 0.0)
    assert [b["n"] for b in bins] == [30], bins
assert verdict(30, ece([(0.70, 1)] * 21 + [(0.70, 0)] * 9)[0]) == "pass"

e, bins = ece([(0.2, 1)] * 4 + [(0.2, 0)] * 16 + [(0.9, 1)] * 18 + [(0.9, 0)] * 2, k=3)
close(e, 0.0)
assert [b["n"] for b in bins] == [20, 20], bins

e, bins = ece([(0.9, 1)] * 10 + [(0.9, 0)] * 10)
close(e, 0.4)

assert verdict(30, 0.10) == "pass"
assert verdict(30, 0.101) == "fail"
assert verdict(29, 0.0) == "insufficient"

gate = {"m@t": {"A": {"1": {"verdict": "pass"}, "2": {"verdict": "fail"}, "3": {"verdict": "insufficient"}}}}
assert passes(gate, "m@t", "A", 1) is True
assert passes(gate, "m@t", "A", 2) is False
assert passes(gate, "m@t", "A", 3) is False
assert passes(gate, "m@t", "A", 4) is False
assert passes(gate, "m@t", "B", 1) is False
assert passes(gate, "other", "A", 1) is False

# Constants are the study's thresholds.
assert (gates.PROBE_MARGIN, gates.MARKET_MARGIN) == (0.10, 0.30)       # quarantine rule
assert (gates.N_BINS, gates.MIN_ROWS, gates.ECE_MAX) == (3, 30, 0.10)  # calibration gate

print("all gate checks pass")
