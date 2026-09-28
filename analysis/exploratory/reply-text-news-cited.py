import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, re, statistics as st, random
from common_ins import load_moments, load_forecasts

CITE_RE = re.compile(r"the provided news|according to the article|the report says|news snippet|the news article|per the article|based on the news|the article (states|says|reports|mentions|notes)|news (report|article)s? (indicate|suggest|state)", re.I)
DISMISS_RE = re.compile(r"no relevant news|no (specific |directly )?relevant news|news (provided )?(does not|doesn't) (mention|address|contain)|no news (is )?(provided|available)|the news (does not|doesn't) (provide|contain|mention)|without (specific )?news|no direct news", re.I)

moments = load_moments()
fc = load_forecasts(mode="normal")

SEED = 20260902
N = 1000

def boot_mean(vals_by_cluster, n=N, seed=SEED):
    ids = list(vals_by_cluster)
    if len(ids) < 2:
        return float('nan'), float('nan'), float('nan')
    allvals = [v for i in ids for v in vals_by_cluster[i]]
    mean = st.fmean(allvals)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        tot, cnt = 0.0, 0
        for i in rng.choices(ids, k=len(ids)):
            tot += sum(vals_by_cluster[i]); cnt += len(vals_by_cluster[i])
        means.append(tot/cnt)
    means.sort()
    return mean, means[int(0.025*n)], means[int(0.975*n)]

def diff_within_model(rows_cite, rows_not, key):
    # within-model diff, clustered by event, paired diff of means per model then averaged?
    # Follow simple approach: pool across models but bootstrap by event cluster, computing group means directly ("within-model" bootstrap)
    from collections import defaultdict
    def by_ev(rows):
        d = defaultdict(list)
        for r in rows:
            d[r['event_id']].append(r[key])
        return d
    ga, gb = by_ev(rows_cite), by_ev(rows_not)
    ia, ib = list(ga), list(gb)
    if not ia or not ib:
        return float('nan'), float('nan'), float('nan')
    point = st.fmean(v for i in ia for v in ga[i]) - st.fmean(v for i in ib for v in gb[i])
    rng = random.Random(SEED)
    diffs = []
    for _ in range(N):
        sa = [v for i in rng.choices(ia, k=len(ia)) for v in ga[i]]
        sb = [v for i in rng.choices(ib, k=len(ib)) for v in gb[i]]
        diffs.append(st.fmean(sa) - st.fmean(sb))
    diffs.sort()
    return point, diffs[int(0.025*N)], diffs[int(0.975*N)]

for win in ["A","B","C"]:
    cite_rows = []
    not_rows = []
    dismiss_rows = []
    n_news = 0
    for f in fc:
        m = moments.get(f['row_id'])
        if not m or m['window'] != win:
            continue
        if not m['news_available']:
            continue
        if not m['primary']:
            continue
        n_news += 1
        p = f['parsed_probability']
        q = m['q']
        y = float(m['outcome_yes'])
        text = f['raw_reply'] or ''
        rec = dict(event_id=m['event_id'], p=p, q=q, y=y,
                   dist=abs(p-q), conf=abs(p-0.5), pval=p,
                   brier_edge=(q-y)**2-(p-y)**2, model=f['model'])
        if CITE_RE.search(text):
            cite_rows.append(rec)
        elif DISMISS_RE.search(text):
            dismiss_rows.append(rec)
        else:
            not_rows.append(rec)
    print(f"WINDOW {win}: news rows={n_news} cite={len(cite_rows)} ({100*len(cite_rows)/n_news:.1f}%) dismiss={len(dismiss_rows)} not={len(not_rows)}")
    for key in ["dist","conf","pval","brier_edge"]:
        pt, lo, hi = diff_within_model(cite_rows, not_rows, key)
        print(f"  cite-not {key}: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
    for key in ["dist","conf"]:
        pt, lo, hi = diff_within_model(dismiss_rows, not_rows, key)
        print(f"  dismiss-not {key}: {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
    # per-model cite share to check whether pooled effect is driven by a specific subset of models
    from collections import Counter
    cnt = Counter(r['model'] for r in cite_rows)
    tot = Counter(r['model'] for r in cite_rows+not_rows+dismiss_rows)
    print("  cite share by model:", {k: f"{100*cnt.get(k,0)/tot[k]:.0f}%" for k in tot})

print("\n--- per-model dist effect (cite - not), window B and C only, no bootstrap, simple check ---")
from collections import defaultdict
for win in ["A","B","C"]:
    per_model = defaultdict(lambda: [[],[]])
    for f in fc:
        m = moments.get(f['row_id'])
        if not m or m['window'] != win or not m['news_available'] or not m['primary']:
            continue
        text = f['raw_reply'] or ''
        p, q = f['parsed_probability'], m['q']
        if CITE_RE.search(text):
            per_model[f['model']][0].append(abs(p-q))
        elif not DISMISS_RE.search(text):
            per_model[f['model']][1].append(abs(p-q))
    print(f"window {win}:")
    for model, (c, n) in per_model.items():
        if len(c) >= 20 and len(n) >= 20:
            print(f"  {model}: cite n={len(c)} mean={st.fmean(c):.3f}  not n={len(n)} mean={st.fmean(n):.3f}  diff={st.fmean(c)-st.fmean(n):+.3f}")
