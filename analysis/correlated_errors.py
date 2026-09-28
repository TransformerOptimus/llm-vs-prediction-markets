"""Correlated errors: do forecasters make the same mistakes?

  error          p - outcome, per row per forecaster
  pair r         Pearson correlation of two forecasters' errors on their common rows;
                 reported as the plain mean over pairs, with a cluster-bootstrap interval
  both wrong     share of common rows where both are on the wrong side of 0.5,
                 against the product of the two individual wrong rates
  template       each run carries a template id; pairs are labelled same-template or
                 cross-template so a drop across templates could be reported (the second,
                 with-price template was not used in the study)

Rows here are the dicts made by scoring.score_forecast (keys p, outcome_yes, row_id,
forecaster, bucket, and optionally template).
"""
import itertools
import math
import random
import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional

from scoring import CLUSTER, CLUSTER_KEY, N_BOOT, SEED, one_window


def _pearson(x, z) -> float:
    mx, mz = st.fmean(x), st.fmean(z)
    vx = sum((a - mx) ** 2 for a in x)
    vz = sum((c - mz) ** 2 for c in z)
    if vx == 0 or vz == 0:
        return float("nan")
    return sum((a - mx) * (c - mz) for a, c in zip(x, z)) / math.sqrt(vx * vz)


def _wrong(r) -> float:
    return 1.0 if (r["p"] >= 0.5) != (r["outcome_yes"] >= 0.5) else 0.0


def pairwise(rows: List[dict], min_common: int = 10, n_boot: int = N_BOOT, seed: int = SEED,
             template_key: Optional[str] = "template", cluster: str = CLUSTER) -> dict:
    """Mean pairwise error correlation and wrong-together rates for one set of rows.
    The interval resamples clusters (events by default)."""
    d: Dict[str, Dict[int, dict]] = defaultdict(dict)
    ck = CLUSTER_KEY[cluster]
    market_cluster = {}
    for r in rows:
        d[r["forecaster"]][r["row_id"]] = r          # one row per forecaster per market
        market_cluster[r["row_id"]] = r[ck]
    names = sorted(d)
    pairs = []
    for a, c in itertools.combinations(names, 2):
        common = sorted(set(d[a]) & set(d[c]))
        if len(common) < min_common:
            continue
        ea = [d[a][m]["p"] - d[a][m]["outcome_yes"] for m in common]
        ec = [d[c][m]["p"] - d[c][m]["outcome_yes"] for m in common]
        wa = [_wrong(d[a][m]) for m in common]
        wc = [_wrong(d[c][m]) for m in common]
        ta = d[a][common[0]].get(template_key) if template_key else None
        tc = d[c][common[0]].get(template_key) if template_key else None
        pairs.append({"a": a, "b": c, "n_common": len(common), "r": _pearson(ea, ec),
                      "both_wrong": st.fmean(x * z for x, z in zip(wa, wc)),
                      "both_wrong_expected": st.fmean(wa) * st.fmean(wc),
                      "same_template": (ta == tc) if (ta is not None and tc is not None) else None,
                      "_common": common, "_ea": dict(zip(common, ea)), "_ec": dict(zip(common, ec))})
    rs = [p["r"] for p in pairs if not math.isnan(p["r"])]
    res = {"n_rows": len(rows), "n_markets": len({r["row_id"] for r in rows}), "n_forecasters": len(names),
           "n_pairs": len(pairs), "mean_r": st.fmean(rs) if rs else float("nan"),
           "min_r": min(rs) if rs else float("nan"), "max_r": max(rs) if rs else float("nan"),
           "both_wrong": st.fmean(p["both_wrong"] for p in pairs) if pairs else float("nan"),
           "both_wrong_expected": st.fmean(p["both_wrong_expected"] for p in pairs) if pairs else float("nan")}
    # cluster bootstrap of the mean pairwise r
    if pairs and n_boot:
        rng = random.Random(seed)
        members = defaultdict(list)
        for m in sorted({m for p in pairs for m in p["_common"]}):
            members[market_cluster[m]].append(m)
        clusters = sorted(members)
        boots = []
        for _ in range(n_boot):
            draw = [m for c in rng.choices(clusters, k=len(clusters)) for m in members[c]]
            vals = []
            for p in pairs:
                xs = [p["_ea"][m] for m in draw if m in p["_ea"]]
                zs = [p["_ec"][m] for m in draw if m in p["_ec"]]
                if len(xs) >= min_common:
                    v = _pearson(xs, zs)
                    if not math.isnan(v):
                        vals.append(v)
            if vals:
                boots.append(st.fmean(vals))
        boots.sort()
        res["r_lo"], res["r_hi"] = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    same = [p["r"] for p in pairs if p["same_template"] is True]
    cross = [p["r"] for p in pairs if p["same_template"] is False]
    res["mean_r_same_template"] = st.fmean(same) if same else None
    res["mean_r_cross_template"] = st.fmean(cross) if cross else None
    for p in pairs:
        for k in ("_common", "_ea", "_ec"):
            p.pop(k)
    res["pairs"] = pairs
    return res


def by_bucket(rows: List[dict], n_boot: int = N_BOOT, cluster: str = CLUSTER) -> Dict[int, dict]:
    one_window(rows, "correlated_errors.by_bucket")
    groups = defaultdict(list)
    for r in rows:
        if r.get("bucket") is not None:
            groups[r["bucket"]].append(r)
    return {b: pairwise(groups[b], n_boot=n_boot, cluster=cluster) for b in sorted(groups)}
