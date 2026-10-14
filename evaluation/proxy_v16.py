"""Approximate JevBench v1.6 Intelligence on labelled local predictions (not the official scorer).

choice: chance-corrected accuracy 100*(acc-c)/(1-c), c = mean 1/options.
noul:   p(correct) >= .8 right, <= .2 wrong, otherwise abstain (no credit); chance-corrected vs 0.5.
score:  100*(1 - MAE/MAE_uniform) on the expected level.
Headline: equal-weight mean of the three types.
"""
import argparse
import json
import statistics
from pathlib import Path


def score(path, variant):
    by = {'choice': [], 'noul': [], 'score': []}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        v = r['variants'][variant]
        if 'target' not in v:
            raise SystemExit('records need targets (validation/audit splits)')
        by[r['kind']].append((v['probs'], v['target'], r['labels']))
    out = {}
    rows = by['choice']
    if rows:
        acc = statistics.fmean(p.index(max(p)) == g for p, g, _ in rows)
        c = statistics.fmean(1 / len(p) for p, _, _ in rows)
        out['choice'] = 100 * (acc - c) / (1 - c)
    rows = by['noul']
    if rows:
        right = statistics.fmean(p[g] >= .8 for p, g, _ in rows)
        wrong = statistics.fmean(p[g] <= .2 for p, g, _ in rows)
        out['noul'] = 100 * (right - wrong)
        out['noul_abstain'] = 100 * (1 - right - wrong)
    rows = by['score']
    if rows:
        mae, chance = [], []
        for p, g, labels in rows:
            levels = [int(x) for x in labels]
            mae.append(abs(sum(l * q for l, q in zip(levels, p)) - levels[g]))
            chance.append(statistics.fmean(abs(l - levels[g]) for l in levels))
        out['score'] = 100 * (1 - statistics.fmean(mae) / statistics.fmean(chance))
    out['intelligence_proxy'] = statistics.fmean(out[k] for k in ('choice', 'noul', 'score') if k in out)
    return {k: round(v, 1) for k, v in out.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('predictions', nargs='+')
    p.add_argument('--variant', default='direct')
    a = p.parse_args()
    for path in a.predictions:
        print(path, a.variant, json.dumps(score(path, a.variant)))
