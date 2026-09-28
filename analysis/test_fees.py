"""Hand-computed checks of the fee formulas.  Run: python3 analysis/test_fees.py"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import buy_return, fee_per_share, order_fee, round_up_cent


def close(a, b, tol=1e-9):
    assert abs(a - b) < tol, (a, b)


def poly(rate, enabled=1):
    return {"venue": "polymarket", "fees_enabled": enabled, "fee_rate": rate, "fee_exponent": 1.0, "fee_taker_only": 1}


def kalshi(rate, taker_only=1):
    return {"venue": "kalshi", "fees_enabled": 1, "fee_rate": rate, "fee_exponent": 1.0, "fee_taker_only": taker_only}


# Polymarket, rate 0.05, price 0.40: per share 0.05 x 0.4 x 0.6 = 0.012.
close(fee_per_share(poly(0.05), 0.40), 0.012)
# $1 buys 1 / (0.40 + 0.012) = 2.42718... shares; outlay exactly $1; a win pays 2.42718 -> return +142.718%.
r = buy_return(poly(0.05), 0.40, 1.0)
close(r["shares"], 1 / 0.412)
close(r["outlay"], 1.0)
close(r["ret"], 1 / 0.412 - 1)
close(r["ret_nofee"], 1 / 0.40 - 1)          # 150% without the fee
# a loss returns exactly -100%
close(buy_return(poly(0.05), 0.40, 0.0)["ret"], -1.0)
# fees off: no fee at all
close(fee_per_share(poly(0.0, enabled=0), 0.40), 0.0)
close(buy_return(poly(0.0, enabled=0), 0.40, 1.0)["ret"], 1.5)

# Kalshi, standard rate 0.07, price 0.50: per share 0.07 x 0.25 = 0.0175; shares = 1 / 0.5175 = 1.93237;
# raw order fee 0.0175 x 1.93237 = 0.033816 -> rounded up to 0.04. Outlay 1.93237 x 0.5 + 0.04 = 1.006184.
m = kalshi(0.07)
close(fee_per_share(m, 0.5), 0.0175)
sh = 1 / 0.5175
close(order_fee(m, 0.5, sh), 0.04)
r = buy_return(m, 0.5, 1.0)
close(r["outlay"], sh * 0.5 + 0.04)
close(r["ret"], (sh - r["outlay"]) / r["outlay"])
# The round-up changes the result: without it the outlay would be exactly $1 and the return 93.237%; with it 92.049%
# ((1.932367 - 1.006184) / 1.006184).
close(sh - 1.0, 0.932367, 1e-6)
close(r["ret"], 0.920492, 1e-6)
assert r["ret"] < sh - 1.0
# Kalshi half-rate series (S&P/Nasdaq), rate 0.035, price 0.20: per share 0.035 x 0.16 = 0.0056; shares 1/0.2056 = 4.8638;
# raw fee 0.027237 -> 0.03.
close(fee_per_share(kalshi(0.035), 0.20), 0.0056)
close(order_fee(kalshi(0.035), 0.20, 1 / 0.2056), 0.03)
# Kalshi maker-fee series: the taker rate is charged regardless of fee_taker_only.
close(fee_per_share(kalshi(0.07, taker_only=0), 0.5), 0.0175)
# Round-up edge cases: an exact cent stays; a hair over goes up; zero stays zero.
close(round_up_cent(0.03), 0.03)
close(round_up_cent(0.030000001), 0.04)
close(round_up_cent(0.0), 0.0)
# Polymarket never rounds: the order fee is the exact per-share fee times shares.
close(order_fee(poly(0.05), 0.40, 2.5), 0.03)
close(order_fee(poly(0.05), 0.40, 2.6), 0.0312)
print("fee tests: all passed")
