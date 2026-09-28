"""No-skill strategies, scored with the same trading rule and fees as the models.

Each strategy turns a row into a stand-in probability p so that the trading rule buys the
intended side and pays the taker fee: p = 1 buys Yes at the ask, p = 0 buys No at one
minus the bid. These p values are trading instructions, not forecasts, so no Brier score is
reported for them.

  1 always_yes      p = 1
  2 always_no       p = 0
  3 cheaper_side    buy the side with the lower ask (Yes ask versus 1 - Yes bid)
  4 market_favored  buy Yes if q > 0.5, No if q < 0.5, nothing at exactly 0.5 (q = the market probability)
  5 coin_flip       one fair coin per row. The coin belongs to the row, not to its position in a
                    list: draw k for row r is random.Random("%d:%s" % (SEED + k, moment_key)).random() < 0.5.
                    The reported coin_flip floor is the mean return over draws k = 0 .. 199 (200 draws),
                    so it does not depend on row order, on the subset passed, or on one lucky coin.

Rows passed to floor_by_bucket must come from one window (bucket numbers differ in dollars).
"""
import random
import statistics as st
from typing import Dict, List, Optional

from scoring import SEED, one_window, score_forecast

STRATEGIES = ["always_yes", "always_no", "cheaper_side", "market_favored", "coin_flip"]
COIN_DRAWS = 200


def coin(moment: dict, k: int) -> float:
    """Draw k of the row's own coin: p = 1 (buy Yes) or 0 (buy No)."""
    return 1.0 if random.Random("%d:%s" % (SEED + k, moment["moment_key"])).random() < 0.5 else 0.0


def stand_in_p(name: str, moment: dict, k: int = 0) -> float:
    if name == "always_yes":
        return 1.0
    if name == "always_no":
        return 0.0
    if name == "cheaper_side":
        ask, bid = moment["best_ask_yes"], moment["best_bid_yes"]
        if ask is None or bid is None:
            return 1.0
        return 1.0 if ask <= 1.0 - bid else 0.0
    if name == "market_favored":
        q = moment["q"]
        return 1.0 if q > 0.5 else (0.0 if q < 0.5 else 0.5)
    if name == "coin_flip":
        return coin(moment, k)
    raise ValueError(name)


def no_skill_rows(moments: Dict[int, dict], cuts, row_ids: Optional[List[int]] = None, window: Optional[str] = None,
                  cut_key: Optional[str] = None, coin_draws: int = COIN_DRAWS) -> List[dict]:
    """Score all five strategies on the given rows. With row_ids = None, `window` is required and
    every primary row of that window is used. coin_flip rows carry a `draw` field 0 .. coin_draws-1
    (one scored row per draw); the other strategies carry draw = 0."""
    if row_ids is None:
        if window is None:
            raise ValueError("no_skill_rows: give row_ids or a window; never all windows at once")
        row_ids = [rid for rid, m in moments.items() if m["primary"] and m["window"] == window]
    ids = sorted(row_ids, key=lambda rid: moments[rid]["moment_key"])
    out = []
    for name in STRATEGIES:
        draws = range(coin_draws) if name == "coin_flip" else [0]
        for k in draws:
            for rid in ids:
                m = moments[rid]
                row = score_forecast(m, stand_in_p(name, m, k), "noskill:" + name, cuts, extra={"draw": k}, cut_key=cut_key)
                for key in ("brier_model", "brier_edge", "error"):
                    row[key] = None                      # not forecasts; accuracy is meaningless
                out.append(row)
    return out


def floor_by_bucket(rows: List[dict]) -> Dict[int, dict]:
    """Per bucket: each strategy's mean return, and the worst of the five (the floor).
    coin_flip is the mean over its draws of the per-draw mean return; its interval is not bootstrapped."""
    from scoring import summarize_by
    one_window(rows, "no_skill.floor_by_bucket")
    out = {}
    for name in STRATEGIES:
        sub = [r for r in rows if r["forecaster"] == "noskill:" + name]
        if name == "coin_flip":
            per_draw = {}
            for r in sub:
                if r["ret"] is not None and r["bucket"] is not None:
                    per_draw.setdefault((r["draw"], r["bucket"]), []).append(r["ret"])
            by_b = {}
            for (k, b), vals in per_draw.items():
                by_b.setdefault(b, []).append(st.fmean(vals))
            for b, means in by_b.items():
                out.setdefault(b, {})[name] = {"ret": st.fmean(means), "ret_lo": min(means), "ret_hi": max(means),
                                               "n_draws": len(means), "n_traded": len(per_draw[(0, b)])}
        else:
            for b, d in summarize_by(sub, "bucket", n_boot=0).items():
                out.setdefault(b, {})[name] = d
    for b, d in out.items():
        worst = min(STRATEGIES, key=lambda n: d[n]["ret"] if n in d and d[n]["ret"] == d[n]["ret"] else 1e9)
        out[b]["floor"] = {"strategy": worst, "ret": d[worst]["ret"], "ret_lo": d[worst].get("ret_lo"), "ret_hi": d[worst].get("ret_hi")}
    return out
