import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
"""fees-stake-sensitivity: are the trading losses a property of the forecasts, or of trading one dollar
at a time?

Every return in the paper comes from staking about one dollar per trade. On Kalshi the fee for an order
is rounded UP to the next cent, and that rounding is a fixed cost that a one-dollar order cannot spread
over anything. Near a thirty-cent price the fee before rounding is a fraction of a cent per share on a
few shares, so the round-up can be a large share of the fee actually paid -- and the models trade at
about thirty cents most of the time, because they buy the side the crowd does not favour. If that is
what drives the negative returns, the result is about order size rather than about forecasting.

This script recomputes every trade at several stakes, changing nothing else: same rule, same side, same
price, same per-share fee, only the number of shares bought and therefore how the per-order rounding
lands. Polymarket charges the exact fee with no rounding, so Polymarket windows should be flat across
stakes; that is the control. Any movement on the Kalshi window is the rounding.

Reported per window and stake: mean return per dollar staked over the traded rows, the same with no fee
at all, and the fee actually paid as a share of the outlay. Intervals are 1,000 event resamples, seed
20260902. Test split and both splits are both reported.
"""
import scoring
from calibration_shape_common import MODELS, SHORT, boot_stat, load

STAKES = (1.0, 10.0, 100.0, 1000.0)


def rescore(row, moment, stake):
    """The same trade at a different stake. Returns (return per dollar, fee share of outlay)."""
    price = row["trade_cost"]
    per_share = scoring.fee_per_share(moment, price)
    shares = stake / (price + per_share)
    fee = scoring.order_fee(moment, price, shares)
    outlay = shares * price + fee
    won = row["trade_won"]
    return (shares * won - outlay) / outlay, fee / outlay


con = scoring.connect()
moments = scoring.load_moments(con)
con.close()
by_key = {m["moment_key"]: m for m in moments.values()}

rows_all, cuts = load("normal")
traded = [r for r in rows_all if r.get("trade_reason") == "traded" and r.get("trade_cost") is not None]
print("rows: traded primary rows, forecasts view, mode normal, all models pooled; the trade is "
      "unchanged at every stake -- same side, same price, same per-share fee -- only the order size "
      "and therefore the per-order rounding differ. Kalshi rounds each order's fee up to the next "
      "cent; Polymarket does not, so \\winA and \\winB are the control. intervals = 1,000 event "
      "resamples, seed 20260902")

for w in "ABC":
    for which, keep in (("test split", lambda r: r["split"] == "test"), ("both splits", lambda r: True)):
        rs = [r for r in traded if r["window"] == w and keep(r)]
        if len(rs) < 100:
            continue
        venue = rs[0]["venue"]
        print("== window %s (%s), %s: %d traded rows, mean price paid %.4f"
              % (w, venue, which, len(rs), sum(r["trade_cost"] for r in rs) / len(rs)))
        nofee = sum(r["ret_nofee"] for r in rs) / len(rs)
        for stake in STAKES:
            vals = []
            for r in rs:
                m = by_key.get(r["moment_key"])
                if m is None:
                    continue
                ret, feeshare = rescore(r, m, stake)
                vals.append((r, ret, feeshare))
            if not vals:
                continue
            for r, ret, feeshare in vals:
                r["_ret_at_stake"] = ret
            ci = boot_stat([r for r, _, _ in vals], lambda t: sum(x["_ret_at_stake"] for x in t) / len(t))
            print("   $%-7.0f mean return %+.4f [%+.4f, %+.4f]  fee as share of outlay %.4f  "
                  "(same trades with no fee at all: %+.4f)"
                  % (stake, ci[0], ci[1], ci[2], sum(f for _, _, f in vals) / len(vals), nofee))
