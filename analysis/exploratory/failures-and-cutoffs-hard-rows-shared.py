import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _paths  # noqa: E702
import sys, statistics as st, random
from common_ins import runs_con, EXCLUDE_MODEL
from collections import defaultdict

con = runs_con()
rows = con.execute("""
  SELECT model, mode, window, row_id, moment_key, attempt, status
  FROM calls WHERE mode='normal' AND model != ? AND attempt=1
""", (EXCLUDE_MODEL,)).fetchall()
con.close()

# first-answer unusable = attempt 1 status != 'ok'
by_model_row = defaultdict(dict)  # row_id -> model -> unusable bool
for r in rows:
    unusable = 1 if r['status'] != 'ok' else 0
    by_model_row[r['row_id']][r['model']] = unusable

KIMI25 = 'openrouter/moonshotai/kimi-k2.5'
KIMI26 = 'openrouter/moonshotai/kimi-k2.6'
V4PRO = 'openrouter/deepseek/deepseek-v4-pro'

def cond_rate(model_a, model_b, by_model_row, window_rows):
    # P(model_b unusable | model_a failed) vs P(model_b unusable | model_a ok)
    fail_b, fail_cnt = 0, 0
    ok_b, ok_cnt = 0, 0
    for rid in window_rows:
        d = by_model_row.get(rid, {})
        if model_a not in d or model_b not in d:
            continue
        if d[model_a] == 1:
            fail_cnt += 1
            fail_b += d[model_b]
        else:
            ok_cnt += 1
            ok_b += d[model_b]
    ra = fail_b/fail_cnt if fail_cnt else float('nan')
    rb = ok_b/ok_cnt if ok_cnt else float('nan')
    return ra, fail_cnt, rb, ok_cnt

con2 = runs_con()
win_of_row = {r['row_id']: r['window'] for r in con2.execute("SELECT DISTINCT row_id, window FROM calls WHERE attempt=1")}
con2.close()

for win in ['B','C']:
    window_rows = [rid for rid,w in win_of_row.items() if w==win]
    print(f"window {win}: n rows with any model = {len(window_rows)}")
    for a,b,label in [(KIMI25,V4PRO,'Kimi2.5->V4Pro'),(V4PRO,KIMI25,'V4Pro->Kimi2.5'),
                       (V4PRO,KIMI26,'V4Pro->Kimi2.6'),(KIMI25,KIMI26,'Kimi2.5->Kimi2.6')]:
        ra, fc, rb, ok = cond_rate(a,b,by_model_row, window_rows)
        print(f"  {label}: fail-cond={ra*100:.1f}% (n={fc})  ok-cond={rb*100:.1f}% (n={ok})  diff={100*(ra-rb):+.1f}pts")

print("\n--- bootstrap over rows (row is the natural cluster here; no event clustering available for this table without join) ---")
def boot_diff_rate(model_a, model_b, window_rows, n=1000, seed=20260902):
    pairs = []
    for rid in window_rows:
        d = by_model_row.get(rid, {})
        if model_a in d and model_b in d:
            pairs.append((d[model_a], d[model_b]))
    if not pairs:
        return None
    def rate(sub):
        fail = [b for a,b in sub if a==1]
        ok = [b for a,b in sub if a==0]
        if not fail or not ok:
            return None
        return sum(fail)/len(fail) - sum(ok)/len(ok)
    point = rate(pairs)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sub = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        d = rate(sub)
        if d is not None:
            diffs.append(d)
    diffs.sort()
    lo, hi = diffs[int(0.025*len(diffs))], diffs[int(0.975*len(diffs))]
    return point, lo, hi

for win in ['B','C']:
    window_rows = [rid for rid,w in win_of_row.items() if w==win]
    for a,b,label in [(KIMI25,V4PRO,'Kimi2.5->V4Pro'),(V4PRO,KIMI25,'V4Pro->Kimi2.5'),
                       (V4PRO,KIMI26,'V4Pro->Kimi2.6'),(KIMI25,KIMI26,'Kimi2.5->Kimi2.6')]:
        res = boot_diff_rate(a,b,window_rows)
        if res:
            pt,lo,hi = res
            print(f"  {win} {label}: diff={100*pt:+.1f}pts  CI=[{100*lo:+.1f},{100*hi:+.1f}]")

# confound check: news presence and length among co-failed rows (>=2 of 3 models fail) vs rows failed by none
print("\n--- confound: news length/count for multi-failed rows ---")
con3 = runs_con()
import scoring
bcon = scoring.connect()
moments = scoring.load_moments(bcon)
raw = {r['row_id']: (r['news_article_count'], r['news_text'], r['question'])
       for r in bcon.execute("SELECT row_id, news_article_count, news_text, question FROM market_moments")}
bcon.close()

for win in ['B','C']:
    window_rows = [rid for rid,w in win_of_row.items() if w==win]
    fail_count = defaultdict(int)
    models_seen = defaultdict(int)
    for rid in window_rows:
        d = by_model_row.get(rid, {})
        for mo in [KIMI25, V4PRO, KIMI26]:
            if mo in d:
                models_seen[rid]+=1
                fail_count[rid]+=d[mo]
    multi = [rid for rid in window_rows if models_seen[rid]==3 and fail_count[rid]>=2]
    none_ = [rid for rid in window_rows if models_seen[rid]==3 and fail_count[rid]==0]
    print(f"window {win}: multi-fail n={len(multi)}  none-fail n={len(none_)}")
    mac = [raw[rid][0] for rid in multi if rid in raw]
    nac = [raw[rid][0] for rid in none_ if rid in raw]
    print(f"  news_article_count: multi={st.fmean(mac):.2f} (n={len(mac)})  none={st.fmean(nac):.2f} (n={len(nac)})")
    mlen = [len(raw[rid][1] or '') for rid in multi if rid in raw]
    nlen = [len(raw[rid][1] or '') for rid in none_ if rid in raw]
    print(f"  news_text char length: multi mean={st.fmean(mlen):.0f} (n={len(mlen)})  none mean={st.fmean(nlen):.0f} (n={len(nlen)})")
    qlen = [len(raw[rid][2] or '') for rid in multi if rid in raw]
    qnlen = [len(raw[rid][2] or '') for rid in none_ if rid in raw]
    print(f"  question char length: multi mean={st.fmean(qlen):.0f}  none mean={st.fmean(qnlen):.0f}")
