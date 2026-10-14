"""v1.6-style proxy on public JevBench-231: per-type chance-corrected, equal type weights."""
import json, statistics as st, sys
sys.path.insert(0, sys.argv[1])
from jevbench.tasks import load_jsonl
tasks = {t.id: t for tier in ('easy', 'original', 'hard') for t in load_jsonl(f'{sys.argv[1]}/datasets/public/{tier}.jsonl')}
def sharp(p, T): w = [x ** T for x in p]; s = sum(w); return [x / s for x in w]
for spec in sys.argv[2:]:
    name, _, T = spec.partition('@'); T = float(T or 1)
    rs = [json.loads(l) for l in open(f'{name}-jevbench/predictions.jsonl')]
    ch, nr, nw, nn, mae, chance = [], 0, 0, 0, [], []
    for r in rs:
        v = r['variants']['direct']; t = tasks[r['id']]; p = v['probs']
        if r['kind'] == 'choice':
            ch.append((v['correct'], 1 / len(p)))
        elif r['kind'] == 'noul':
            p = sharp(p, T); nn += 1
            g = p.index(max(p)) if v['correct'] else 1 - p.index(max(p))
            nr += p[g] >= .8; nw += p[g] <= .2
        else:
            p = sharp(p, T); L = [int(x) for x in r['labels']]; g = int(t.expected)
            mae.append(abs(sum(l * q for l, q in zip(L, p)) - g)); chance.append(st.fmean(abs(l - g) for l in L))
    acc = st.fmean(c for c, _ in ch); c = st.fmean(x for _, x in ch)
    out = dict(top1=sum(r['variants']['direct']['correct'] for r in rs), choice=100 * (acc - c) / (1 - c),
               noul=100 * (nr - nw) / nn, score=100 * (1 - st.fmean(mae) / st.fmean(chance)))
    out['intel_proxy'] = st.fmean([out['choice'], out['noul'], out['score']])
    print(spec.ljust(16), {k: round(v, 1) for k, v in out.items()})
