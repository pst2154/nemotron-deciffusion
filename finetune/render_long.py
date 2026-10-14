#!/usr/bin/env python3
"""Pre-render long rows with the serving compactor so they fit the training length and match serving.

At serving time, prompts over COMPACT_OVER (8192) tokens have their long state content reduced by
serving/app/compaction.py `compact_state`: agent traces (keys like trajectory/log/messages/steps/history) keep
their opening and most recent steps (head+tail); long prose keeps its most question-relevant sentences (BM25).
The trainer drops rows over --max-tokens (8192), and its shuffled-option path re-renders the raw state, so long
rows must be compacted *before* training. This script does that with the exact serving function.

Rows that already fit are copied unchanged. Long rows are compacted with a state budget that starts at --budget
and shrinks until the whole prompt (system + question + options + state + mask) fits --max-tokens. Rows that
cannot be made to fit (e.g. long content in many short fields, which the compactor never touches) are dropped
and counted.

  python3 finetune/render_long.py --checkpoint $CKPT --input long.jsonl --output long.rendered.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import APP, load_tokenizer, read_jsonl, write_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', ''), help='model dir (tokenizer) [$CHECKPOINT]')
    p.add_argument('--app', default=str(APP))
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-tokens', type=int, default=8192, help='prompt + mask limit (train_pack.py --max-tokens)')
    p.add_argument('--budget', type=int, default=7168, help='initial state budget passed to compact_state')
    p.add_argument('--tries', type=int, default=6)
    a = p.parse_args()
    if not a.checkpoint:
        p.error('--checkpoint (or $CHECKPOINT) is required')
    sys.path.insert(0, a.app)
    os.environ['COMPACT_OVER'] = '0'
    import scoring
    from compaction import compact_state
    scoring.COMPACT_OVER = 0
    tok = load_tokenizer(a.checkpoint)

    def length(inp):
        return len(scoring.prepare_questions(tok, dict(inp, model='x'), position_ordering=True)[0][4]) + 1

    out, stats = [], Counter()
    for row in read_jsonl(a.input):
        inp = row['input']
        try:
            n = length(inp)
        except (ValueError, KeyError, TypeError, StopIteration):
            stats['invalid_row'] += 1
            continue
        if n <= a.max_tokens:
            out.append(row); stats['fits'] += 1
            continue
        q = next(iter(inp['questions'].values()))
        options = [m for m in scoring.prepare_questions(tok, dict(inp, model='x'))[0][3]]
        budget, done = a.budget, False
        for _ in range(a.tries):
            state, info = compact_state(tok, inp['state'], scoring._as_text(q['instructions']), options, budget)
            if info is None:
                break
            new = dict(inp, state=state)
            m = length(new)
            if m <= a.max_tokens:
                row = dict(row, input=new)
                row['provenance'] = dict(row.get('provenance') or {},
                                         rendered=dict(compaction=info.get('mode', 'bm25'), budget=budget,
                                                       tokens_before=n, tokens_after=m))
                out.append(row); stats['compacted'] += 1; done = True
                break
            budget = max(256, budget - (m - a.max_tokens) - 128)
        if not done:
            stats['dropped_cannot_fit'] += 1
    write_jsonl(a.output, out)
    print(json.dumps(dict(stats)))


if __name__ == '__main__':
    main()
