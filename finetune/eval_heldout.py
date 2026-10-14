#!/usr/bin/env python3
"""Score held-out rows (training-row format) against a running server or a local checkpoint.

  --url URL          POST each row to URL/v1/systemone (the real serving path: ordering, compaction, vLLM)
  --checkpoint DIR   native single pass on one GPU, as evaluation/native_eval.py does (needs torch + CUDA)

Metrics overall, per question type and per domain: top-1 accuracy, log loss (-log p(gold)), Brier score,
expected calibration error (10 bins of top-1 confidence), and for noul the mean p(true) and ROC AUC.
--save writes per-row predictions; --baseline compares with a saved prediction file (paired wins/losses).

  python3 finetune/eval_heldout.py --url http://localhost:8770 --input data/mix1/task_eval.jsonl \\
      --save results/fft1.task.jsonl --baseline results/start.task.jsonl

For the general-skill regression check, run the same on a replay held-out file and on JevBench public
(evaluation/native_eval.py with JEVBENCH_HARNESS set).
"""
import argparse
import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import APP, gold_label, labels_of, read_jsonl, write_jsonl  # noqa: E402


def server_predict(url, model_name):
    def predict(row):
        key, q = next(iter(row['input']['questions'].items()))
        body = json.dumps(dict(model=model_name, state=row['input']['state'], questions={key: q})).encode()
        req = urllib.request.Request(url.rstrip('/') + '/v1/systemone', data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            ans = json.loads(r.read())['answers'][key]
        if q['type'] == 'noul':
            return {'false': 1 - ans['noul'], 'true': ans['noul']}
        return {str(k): v for k, v in ans['probabilities'].items()}
    return predict


def native_predict(checkpoint, app, attention):
    import torch
    sys.path.insert(0, app)
    from transformers import AutoModel, AutoTokenizer
    from scoring import candidate_codes, prepare_questions
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    codes = candidate_codes(tok)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                      attn_implementation='sdpa').cuda().eval()
    for layer in model.encoder.layers:
        layer.self_attn.diffusion_lm = attention == 'noncausal'

    @torch.inference_mode()
    def predict(row):
        _, _, labels, _, ids = prepare_questions(tok, dict(row['input'], model='x'), position_ordering=True)[0]
        x = torch.tensor([ids + [model.mask_token_id]], device='cuda')
        h = model.encoder(input_ids=x, use_cache=False, use_causal_mask=attention == 'causal',
                          is_training=False).last_hidden_state
        logits = model.diffusion_head(h[:, -1]).float()[0, [c for _, c in codes[:len(labels)]]]
        return dict(zip(labels, logits.softmax(-1).tolist()))
    return predict


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(records):
    out = {}
    groups = defaultdict(list)
    for r in records:
        groups['all'].append(r); groups['type/' + r['type']].append(r); groups['domain/' + str(r['domain'])].append(r)
    for name, rs in sorted(groups.items()):
        n = len(rs)
        m = dict(n=n, accuracy=round(sum(r['correct'] for r in rs) / n, 4),
                 log_loss=round(sum(-math.log(max(1e-12, r['p_gold'])) for r in rs) / n, 4),
                 brier=round(sum(r['brier'] for r in rs) / n, 4))
        bins = defaultdict(list)
        for r in rs:
            bins[min(9, int(r['confidence'] * 10))].append(r)
        m['ece'] = round(sum(len(b) / n * abs(sum(x['correct'] for x in b) / len(b) - sum(x['confidence'] for x in b) / len(b))
                             for b in bins.values()), 4)
        nouls = [r for r in rs if r['type'] == 'noul']
        if nouls:
            m['noul_mean_p_true'] = round(sum(r['probs']['true'] for r in nouls) / len(nouls), 4)
            a = auc([r['probs']['true'] for r in nouls if r['gold'] == 'true'],
                    [r['probs']['true'] for r in nouls if r['gold'] == 'false'])
            m['noul_auc'] = round(a, 4) if a is not None else None
        out[name] = m
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, nargs='+')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--url')
    g.add_argument('--checkpoint')
    p.add_argument('--model-name', default='Nemotron-Deciffusion-8B', help='"model" field sent to the server')
    p.add_argument('--app', default=str(APP))
    p.add_argument('--attention', choices=['causal', 'noncausal'], default='noncausal')
    p.add_argument('--save', default='', help='write per-row predictions JSONL')
    p.add_argument('--baseline', default='', help='predictions JSONL from another model for a paired comparison')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    rows = [r for f in a.input for r in read_jsonl(f)]
    if a.limit:
        rows = rows[:a.limit]
    predict = server_predict(a.url, a.model_name) if a.url else native_predict(a.checkpoint, a.app, a.attention)
    records, failures = [], 0
    for i, row in enumerate(rows):
        _, q = next(iter(row['input']['questions'].items()))
        gold = gold_label(row)
        try:
            probs = predict(row)
        except Exception as e:  # report and continue (e.g. prompt over the server limit)
            failures += 1
            print(f"{row.get('id')}: {e}", file=sys.stderr)
            continue
        labels = labels_of(q)
        top = max(labels, key=lambda l: probs.get(l, 0))
        records.append(dict(id=row.get('id'), type=q['type'], domain=row.get('domain'), gold=gold, probs=probs,
                            pred=top, correct=top == gold, confidence=probs.get(top, 0), p_gold=probs.get(gold, 0),
                            brier=sum((probs.get(l, 0) - (l == gold)) ** 2 for l in labels)))
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{len(rows)}', file=sys.stderr, flush=True)
    summary = summarize(records) if records else {}
    summary['failures'] = failures
    if a.baseline:
        base = {r['id']: r for r in read_jsonl(a.baseline)}
        pairs = [(r, base[r['id']]) for r in records if r['id'] in base]
        summary['paired_vs_baseline'] = dict(n=len(pairs), wins=sum(r['correct'] and not b['correct'] for r, b in pairs),
                                             losses=sum(b['correct'] and not r['correct'] for r, b in pairs))
    print(json.dumps(summary, indent=2))
    if a.save:
        write_jsonl(a.save, records)


if __name__ == '__main__':
    main()
