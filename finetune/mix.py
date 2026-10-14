#!/usr/bin/env python3
"""Mix task rows with replay rows into a train_pack.py / train_lora.py data directory.

Replay = rows from the original decision-training distribution. Without it the model forgets general skills
(our routing-only round lost 4 JevBench points). Use ~40-60% replay. Sources can be weighted:

  python3 finetune/mix.py --task data/task/train.jsonl \\
      --replay replay/generic.jsonl:3 --replay replay/noul.jsonl:1 --replay-fraction 0.5 \\
      --task-eval data/task/heldout.jsonl --replay-eval replay/generic-heldout.jsonl \\
      --batch 256 --output data/mix1

Output directory:
  train.jsonl      task + replay, deterministically shuffled, trimmed to a multiple of --batch
  eval.jsonl       checkpoint-selection split for the trainer. train_pack.py asserts exactly 580 rows, so the
                   default --eval-size is 580 (task held-out first, then replay held-out; sampled with
                   replacement only if both together are smaller, with a warning)
  task_eval.jsonl  the full task held-out set, for evaluation after training (eval_heldout.py)
  manifest.json    counts, sources, hashes
Rows whose state also appears in an eval file are removed from train.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, stable_hash, write_jsonl  # noqa: E402


def parse_source(spec):
    if Path(spec).exists() or ':' not in spec:
        return spec, 1.0
    path, _, w = spec.rpartition(':')
    return path, float(w)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', nargs='+', required=True, help='task train JSONL file(s)')
    p.add_argument('--replay', action='append', default=[], help='replay JSONL, optionally path:weight; repeatable')
    p.add_argument('--replay-fraction', type=float, default=0.5, help='share of final train rows from replay')
    p.add_argument('--task-repeat', type=int, default=1, help='repeat task rows (small tasks); the trainer '
                   're-shuffles options on the fly for half the rows, so repeats are not identical prompts')
    p.add_argument('--task-eval', default='', help='task held-out JSONL')
    p.add_argument('--replay-eval', default='', help='replay held-out JSONL (fills eval.jsonl up to --eval-size)')
    p.add_argument('--eval-size', type=int, default=580, help='rows in eval.jsonl (train_pack.py asserts 580)')
    p.add_argument('--batch', type=int, default=256, help='trim train to a multiple of this')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rng = random.Random(a.seed)

    task = [r for f in a.task for r in read_jsonl(f)] * a.task_repeat
    task_eval = read_jsonl(a.task_eval) if a.task_eval else []
    replay_eval = read_jsonl(a.replay_eval) if a.replay_eval else []

    # replay: weighted draw without replacement from each source
    n_replay = round(len(task) * a.replay_fraction / max(1e-9, 1 - a.replay_fraction)) if a.replay else 0
    sources = [parse_source(s) for s in a.replay]
    wsum = sum(w for _, w in sources) or 1
    replay, by_source = [], Counter()
    for path, w in sources:
        rows = read_jsonl(path)
        want = round(n_replay * w / wsum)
        if want > len(rows):
            print(f'WARNING: {path}: wanted {want} replay rows, only {len(rows)} available; using all',
                  file=sys.stderr)
        take = rng.sample(rows, min(want, len(rows)))
        replay += take; by_source[path] = len(take)

    # eval.jsonl for checkpoint selection
    pool_eval = list(task_eval)
    rng.shuffle(pool_eval)
    sel = pool_eval[:a.eval_size]
    if len(sel) < a.eval_size and replay_eval:
        extra = list(replay_eval); rng.shuffle(extra)
        sel += extra[:a.eval_size - len(sel)]
    if sel and len(sel) < a.eval_size:
        print(f'WARNING: only {len(sel)} eval rows; sampling with replacement to reach {a.eval_size}. '
              'Selection on this split is a sanity check only.', file=sys.stderr)
        base = list(sel)
        sel += [dict(r, id=f"{r['id']}:dup{i}") for i, r in enumerate(rng.choices(base, k=a.eval_size - len(sel)))]

    blocked = {stable_hash(r['input']['state']) for r in task_eval + replay_eval + sel}
    train, leaked = [], 0
    for origin, rows in (('task', task), ('replay', replay)):
        for r in rows:
            if stable_hash(r['input']['state']) in blocked:
                leaked += 1
                continue
            train.append((origin, r))
    if leaked:
        print(f'NOTE: removed {leaked} train rows whose state appears in an eval file', file=sys.stderr)
    rng.shuffle(train)
    trimmed = len(train) - len(train) % a.batch if len(train) >= a.batch else len(train)
    if len(train) < a.batch:
        print(f'WARNING: {len(train)} train rows < batch {a.batch}; train_pack.py would run 0 steps', file=sys.stderr)
    origins = Counter(o for o, _ in train[:trimmed])
    train = [r for _, r in train[:trimmed]]

    out = Path(a.output)
    write_jsonl(out / 'train.jsonl', train)
    if sel:
        write_jsonl(out / 'eval.jsonl', sel)
    if task_eval:
        write_jsonl(out / 'task_eval.jsonl', task_eval)
    ids = Counter(r['id'] for r in train)
    manifest = dict(train=len(train), task_rows=len(task), replay_rows=len(replay), replay_by_source=dict(by_source),
                    replay_share=round(origins['replay'] / max(1, len(train)), 3),
                    removed_overlapping_eval_states=leaked, eval=len(sel), task_eval=len(task_eval),
                    duplicate_ids=sum(1 for v in ids.values() if v > 1),
                    types=dict(Counter(r['input']['questions'][next(iter(r['input']['questions']))]['type'] for r in train)),
                    train_sha256=hashlib.sha256((out / 'train.jsonl').read_bytes()).hexdigest(), seed=a.seed)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
