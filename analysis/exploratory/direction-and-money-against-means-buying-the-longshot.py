import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys
from common_scripts import load_benchmark as load_moments, load_forecasts
import scoring as S

def build_scored(moments, cuts, forecasts, primary_only=False):
    # same as common_scripts.build_rows, with the primary filter the script asks for
    rows = []
    for f in forecasts:
        m = moments.get(f["row_id"])
        if m is None or (primary_only and not m["primary"]):
            continue
        if _paths.test_split_only() and m["split"] != "test":
            continue
        row = S.score_forecast(m, f["parsed_probability"], f["model"], cuts)
        row["model"] = f["model"]
        rows.append(row)
    return rows

moments, cuts = load_moments()
forecasts = load_forecasts("normal")
rows = build_scored(moments, cuts, forecasts, primary_only=True)
print("total primary scored rows:", len(rows))

for win in ["A", "B", "C"]:
    wr = [r for r in rows if r["window"] == win]
    against = [r for r in wr if r["direction"] == "against"]
    withc = [r for r in wr if r["direction"] == "with"]
    fade_fav = [r for r in against if r["q"] > 0.5]
    back_long = [r for r in against if r["q"] < 0.5]
    band_hi_against = [r for r in against if r["band"] == "0.75-0.95"]
    be_fade, lo1, hi1 = S.boot_mean(fade_fav, "brier_edge", n=1000)
    be_back, lo2, hi2 = S.boot_mean(back_long, "brier_edge", n=1000)
    be_band, lo3, hi3 = S.boot_mean(band_hi_against, "brier_edge", n=1000)
    print(f"[{win}] fade_fav n={len(fade_fav)} edge={be_fade:.3f} [{lo1:.3f},{hi1:.3f}]  "
          f"back_long n={len(back_long)} edge={be_back:.3f} [{lo2:.3f},{hi2:.3f}]  "
          f"band0.75-0.95_against n={len(band_hi_against)} edge={be_band:.3f} [{lo3:.3f},{hi3:.3f}]")

# CONFOUND CHECK: compare mean p (model probability) in band 0.75-0.95 with-crowd vs against-crowd,
# and check if the "against" rows in that band are just p stuck near the 0.55-0.60 plateau (known fact)
print()
print("=== confound: is fading-a-favorite loss just the known compression-to-plateau? ===")
for win in ["A", "B", "C"]:
    wr = [r for r in rows if r["window"] == win]
    band = [r for r in wr if r["band"] == "0.75-0.95"]
    against = [r for r in band if r["direction"] == "against"]
    withc = [r for r in band if r["direction"] == "with"]
    mean_p_against = sum(r["p"] for r in against) / len(against) if against else float("nan")
    mean_p_with = sum(r["p"] for r in withc) / len(withc) if withc else float("nan")
    mean_q_against = sum(r["q"] for r in against) / len(against) if against else float("nan")
    print(f"[{win}] band 0.75-0.95: n_against={len(against)} mean_p={mean_p_against:.3f} mean_q={mean_q_against:.3f}  "
          f"n_with={len(withc)} mean_p_with={mean_p_with:.3f}")

# The money half: checks the sentence "against-the-crowd bets are 30-cent contracts that win about 28%
# of the time". The contract price and the winning side both come from the trading rule in
# scoring.trade, and the bootstrap is the same one the rest of this script uses (1,000 resamples of
# events, seed 20260902).
print()
print("=== The money half: what the trading rule actually buys ===")
print("Contract price is what one contract of the side bought costs: the best ask when the model buys Yes,")
print("one minus the best bid when it buys No. Win share is how often that side won. Traded rows only;")
print("one-sided and crossed order books are never traded, so they are left out here and counted below.")
for win in ["A", "B", "C"]:
    wr = [r for r in rows if r["window"] == win]
    for label, sel in (("against the crowd", [r for r in wr if r["direction"] == "against"]),
                       ("with the crowd", [r for r in wr if r["direction"] == "with"])):
        traded = [r for r in sel if r["trade_cost"] is not None]
        if not traded:
            continue
        c, clo, chi = S.boot_mean(traded, "trade_cost", n=1000)
        w_, wlo, whi = S.boot_mean(traded, "trade_won", n=1000)
        yes = sum(1 for r in traded if r["trade_side"] == "Yes")
        print("[%s] %-18s traded n=%d of %d (buys Yes on %d, No on %d): contract price %.3f [%.3f,%.3f], "
              "win share %.3f [%.3f,%.3f]"
              % (win, label, len(traded), len(sel), yes, len(traded) - yes, c, clo, chi, w_, wlo, whi))
    no_trade = [r for r in wr if r["direction"] == "against" and r["trade_cost"] is None]
    print("     against-the-crowd rows the rule did not trade: %d (%s)"
          % (len(no_trade), ", ".join(sorted({r["trade_reason"] for r in no_trade})) or "none"))

print()
print("=== The same split by side, because buying Yes against the crowd is the longshot buy ===")
for win in ["A", "B", "C"]:
    against = [r for r in rows if r["window"] == win and r["direction"] == "against" and r["trade_cost"] is not None]
    for side in ("Yes", "No"):
        sel = [r for r in against if r["trade_side"] == side]
        if not sel:
            continue
        c, clo, chi = S.boot_mean(sel, "trade_cost", n=1000)
        w_, wlo, whi = S.boot_mean(sel, "trade_won", n=1000)
        r_, rlo, rhi = S.boot_mean(sel, "ret", n=1000)
        print("[%s] buys %-3s against the crowd: n=%d contract price %.3f [%.3f,%.3f], win share %.3f [%.3f,%.3f], "
              "return per dollar %+.3f [%+.3f,%+.3f]"
              % (win, side, len(sel), c, clo, chi, w_, wlo, whi, r_, rlo, rhi))
