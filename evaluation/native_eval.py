"""Native single-pass evaluation (prompt + one mask token; --attention causal or noncausal), any CUDA GPU.

Writes results/<tag>-<split>/predictions.jsonl in think_eval's schema (variant 'direct') so jev16.py and
proxy_v16.py apply unchanged. Uses the serving prompt builder and option ordering.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

APP = os.environ.get('DECIFFUSION_APP', str(Path(__file__).resolve().parents[1] / 'serving' / 'app'))
HARNESS = os.environ.get('JEVBENCH_HARNESS', '')  # path to the JevBench harness checkout (required)
DATA = Path(os.environ.get('DECIFFUSION_EVAL_DATA', 'data/eval'))  # directory with the native eval/audit splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--results', default=os.environ.get('DECIFFUSION_RESULTS', 'results'))
    p.add_argument('--splits', default='jevbench,eval,audit')
    p.add_argument('--attention', choices=['causal', 'noncausal'], default='causal')
    a = p.parse_args()
    sys.path.insert(0, APP); sys.path.insert(0, HARNESS)
    from transformers import AutoModel, AutoTokenizer
    from scoring import candidate_codes, pack_answer, prepare_questions
    from jevbench.tasks import load_jsonl
    from jevbench.adapters.base import build_question
    from jevbench.scoring import score_task
    tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    codes = candidate_codes(tok)
    model = AutoModel.from_pretrained(a.checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                      attn_implementation='sdpa').cuda().eval()
    attn = [layer.self_attn for layer in model.encoder.layers]
    for m in attn:
        m.diffusion_lm = a.attention == 'noncausal'

    @torch.inference_mode()
    def probs_for(ids, n):
        x = torch.tensor([ids + [model.mask_token_id]], device='cuda')
        hidden = model.encoder(input_ids=x, use_cache=False, use_causal_mask=a.attention == 'causal',
                               is_training=False).last_hidden_state
        logits = model.diffusion_head(hidden[:, -1]).float()[0, [c for _, c in codes[:n]]]
        return logits.softmax(-1).tolist()

    for split in a.splits.split(','):
        if split == 'jevbench':
            items = [dict(id=t.id, family=t.family, tier=tier, task=t,
                          payload={'model': 'x', 'state': t.state, 'questions': {'decision': build_question(t)}})
                     for tier in ('easy', 'original', 'hard') for t in load_jsonl(f'{HARNESS}/datasets/public/{tier}.jsonl')]
        else:
            items = []
            for line in (DATA / f'{split}.jsonl').read_text().split('\n'):
                if line.strip():
                    r = json.loads(line)
                    items.append(dict(id=r['id'], family=r.get('source_family'), tier=r.get('domain'),
                                      target=r['reference']['target'], payload=dict(r['input'], model='x')))
        out = Path(a.results) / f'{a.tag}-{split}'
        out.mkdir(parents=True, exist_ok=True)
        correct = 0
        with (out / 'predictions.jsonl').open('w') as f:
            for it in items:
                key, kind, labels, _, ids = prepare_questions(tok, it['payload'], position_ordering=True)[0]
                probs = probs_for(ids, len(labels))
                rec = dict(id=it['id'], family=it['family'], tier=it['tier'], kind=kind, labels=labels, variants={})
                if split == 'jevbench':
                    ans = pack_answer(it['payload']['questions'][key], labels, probs)
                    mapped = {'no': 1 - ans['noul'], 'yes': ans['noul']} if kind == 'noul' else ans['probabilities']
                    ok = bool(score_task(mapped, it['task'])['correct'])
                    rec['variants']['direct'] = dict(probs=probs, correct=ok)
                else:
                    target = str(it['target']).lower() if kind == 'noul' else str(it['target'])
                    g = labels.index(target); ok = probs.index(max(probs)) == g
                    rec['variants']['direct'] = dict(probs=probs, correct=ok, target=g)
                correct += ok
                f.write(json.dumps(rec) + '\n')
        (out / 'summary.json').write_text(json.dumps(dict(split=split, n=len(items), correct={'direct': correct})))
        print(a.tag, split, correct, len(items), flush=True)


if __name__ == '__main__':
    main()
