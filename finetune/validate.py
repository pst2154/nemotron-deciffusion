#!/usr/bin/env python3
"""Validate training rows exactly as the trainers will see them, and report label balance.

Checks per row (a failing row is reported and, with --output, left out):
  schema      id, input.state, exactly one question, reference.target; the request passes the serving validator
              (serving/app/compat_gateway.py) and prompt builder (serving/app/scoring.py prepare_questions)
  label       target is one of the question's labels (choice key / 'true'|'false' / score level '0'..'n-1')
  options     2..501 options (501 = single-token option codes in the Nemotron tokenizer)
  soft label  teacher_probabilities covers every label, is finite, non-negative and sums to > 0
  length      rendered prompt + 1 mask token <= --max-tokens (default 8192), measured WITHOUT serving compaction,
              because train_pack.py's shuffled-option path re-renders the raw state (pre-render long states with
              render_long.py)
  duplicate   same state + question + options as an earlier row
Warnings (reported, row kept unless --strict):
  leak        a state key named like an answer field (--leak-fields), or the gold option text appearing
              verbatim in the state for choice rows with descriptive options
  overlap     state shares a --ngram n-gram with a row in --against (contamination with your eval set)

Report: counts per type/domain, noul true share, score level histogram, choice gold position and
"gold is the longest option" share, token-length percentiles.

  python3 finetune/validate.py --checkpoint $CKPT data/task/train.jsonl --output data/task/train.valid.jsonl
  python3 finetune/validate.py --no-tokenizer data/task/*.jsonl            # schema/balance only, no model files
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (APP, DEFAULT_LEAK_FIELDS, find_keys, gold_label, labels_of, ngrams, option_texts, read_jsonl,  # noqa: E402
                    stable_hash, state_text, write_jsonl)


class RowChecker:
    def __init__(self, tok, scoring, max_tokens, leak_fields, against=None, n=13):
        self.tok, self.scoring, self.max_tokens = tok, scoring, max_tokens
        self.leak_fields = {f.lower() for f in leak_fields}
        self.seen = set()
        self.against, self.n = against or set(), n
        self.max_options = len(scoring.candidate_codes(tok)) if tok is not None else 501

    def check(self, row):
        """Return (errors, warnings, info)."""
        errors, warnings, info = [], [], {}
        for k in ('id', 'input', 'reference'):
            if k not in row:
                return [f'missing {k}'], warnings, info
        inp = row['input']
        if not isinstance(inp, dict) or 'state' not in inp or not isinstance(inp.get('questions'), dict):
            return ['input must have state and questions'], warnings, info
        if len(inp['questions']) != 1:
            return [f"exactly one question per row (got {len(inp['questions'])})"], warnings, info
        if 'target' not in row['reference']:
            return ['missing reference.target'], warnings, info
        _, q = next(iter(inp['questions'].items()))
        try:
            self.scoring.validate_request(dict(inp, model='x'))
        except ValueError as e:
            return [f'request: {e}'], warnings, info
        kind = q['type']; info['type'] = kind
        labels = labels_of(q)
        if len(labels) > self.max_options:
            errors.append(f'{len(labels)} options > {self.max_options}')
        gold = gold_label(row); info['gold'] = gold
        if gold not in labels:
            errors.append(f'target {gold!r} not in labels {labels[:6]}...')
        tp = row.get('teacher_probabilities')
        if tp is not None:
            vals = [tp.get(l) for l in labels]
            if any(v is None for v in vals):
                errors.append('teacher_probabilities missing labels')
            elif any((not isinstance(v, (int, float))) or not math.isfinite(v) or v < 0 for v in vals) or sum(vals) <= 0:
                errors.append('teacher_probabilities must be finite, >= 0, sum > 0')
            elif gold in labels and max(labels, key=lambda l: tp[l]) != gold:
                warnings.append('soft label argmax != reference.target')
        key = stable_hash([inp['state'], q.get('instructions'), q.get('criteria')])
        if key in self.seen:
            errors.append('duplicate')
        self.seen.add(key)
        # leakage
        st = inp['state']
        leaks = list(find_keys(st, self.leak_fields)) if isinstance(st, (dict, list)) else []
        if leaks:
            warnings.append('leak-field ' + '.'.join(map(str, leaks[0])))
        if kind == 'choice' and gold in labels:
            text = option_texts(q)[gold]
            if len(text) >= 25 and text.lower() in state_text(st).lower():
                warnings.append('gold option text appears verbatim in state')
        if self.against and ngrams(state_text(st), self.n) & self.against:
            warnings.append(f'{self.n}-gram overlap with --against')
        # length, through the real serving prompt builder (no compaction)
        if self.tok is not None and not errors:
            try:
                prepared = self.scoring.prepare_questions(self.tok, dict(inp, model='x'), position_ordering=True)
                _, _, plabels, _, ids = prepared[0]
                info['tokens'] = len(ids) + 1
                if len(ids) + 1 > self.max_tokens:
                    errors.append(f'length {len(ids) + 1} > {self.max_tokens}')
                info['gold_position'] = plabels.index(gold) if gold in plabels else None
            except ValueError as e:
                errors.append(f'prompt: {e}')
        return errors, warnings, info


def longest_is_gold(q, gold):
    lens = {k: len(v) for k, v in option_texts(q).items()}
    top = max(lens.values())
    return lens.get(gold) == top and sum(v == top for v in lens.values()) == 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+')
    p.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', ''), help='model dir (tokenizer) [$CHECKPOINT]')
    p.add_argument('--no-tokenizer', action='store_true', help='skip the token-length check')
    p.add_argument('--app', default=str(APP), help='serving/app directory (prompt builder)')
    p.add_argument('--max-tokens', type=int, default=8192, help='train_pack.py --max-tokens')
    p.add_argument('--leak-fields', default='', help='extra comma-separated answer-like key names')
    p.add_argument('--against', nargs='*', default=[], help='eval JSONL files for n-gram overlap warnings')
    p.add_argument('--ngram', type=int, default=13)
    p.add_argument('--strict', action='store_true', help='drop rows with warnings too')
    p.add_argument('--output', default='', help='write passing rows here (all input files concatenated)')
    p.add_argument('--show', type=int, default=5, help='example ids per problem to print')
    a = p.parse_args()

    sys.path.insert(0, a.app)
    os.environ['COMPACT_OVER'] = '0'  # measure the raw rendering (what the shuffled-option path trains on)
    import scoring
    scoring.COMPACT_OVER = 0
    tok = None
    if not a.no_tokenizer:
        if not a.checkpoint:
            p.error('--checkpoint (or $CHECKPOINT) is required unless --no-tokenizer')
        from common import load_tokenizer
        tok = load_tokenizer(a.checkpoint)
    against = set()
    for f in a.against:
        for r in read_jsonl(f):
            st = r['input']['state'] if 'input' in r else r.get('state')
            against |= ngrams(state_text(st), a.ngram)
    checker = RowChecker(tok, scoring, a.max_tokens,
                         list(DEFAULT_LEAK_FIELDS) + [x for x in a.leak_fields.split(',') if x], against, a.ngram)

    kept, problems, examples = [], Counter(), defaultdict(list)
    stats = defaultdict(Counter)
    lengths, longest, choice_n = [], 0, 0
    total = 0
    for f in a.files:
        for row in read_jsonl(f):
            total += 1
            errors, warnings, info = checker.check(row)
            for e in errors:
                tag = 'ERROR ' + e.split(' ')[0] if e.startswith(('length', 'target', 'request', 'prompt')) else 'ERROR ' + e
                problems[tag] += 1; examples[tag].append(row.get('id'))
            for w in warnings:
                tag = 'WARN ' + w.split(' (')[0]
                tag = 'WARN leak-field' if w.startswith('leak-field') else tag
                problems[tag] += 1; examples[tag].append(f"{row.get('id')} ({w})" if w.startswith('leak') else row.get('id'))
            if errors or (a.strict and warnings):
                continue
            kept.append(row)
            kind = info['type']
            stats['type'][kind] += 1
            stats['domain'][row.get('domain', '?')] += 1
            if kind in ('noul', 'score'):
                stats[f'{kind}_label'][info['gold']] += 1
            else:
                choice_n += 1
                q = next(iter(row['input']['questions'].values()))
                longest += longest_is_gold(q, info['gold'])
                stats['choice_n_options'][min(len(q['criteria']), 10) if len(q['criteria']) < 10 else '10+'] += 1
                if info.get('gold_position') is not None:
                    stats['choice_gold_position_served'][min(info['gold_position'], 9)] += 1
            if 'tokens' in info:
                lengths.append(info['tokens'])
            stats['soft_label']['yes' if 'teacher_probabilities' in row else 'no'] += 1

    report = {'rows': total, 'valid': len(kept), 'problems': dict(problems.most_common())}
    report.update({k: dict(sorted(v.items(), key=lambda kv: str(kv[0]))) for k, v in stats.items()})
    if stats['noul_label']:
        n = sum(stats['noul_label'].values())
        report['noul_true_share'] = round(stats['noul_label']['true'] / n, 3)
    if choice_n:
        report['choice_longest_option_is_gold_share'] = round(longest / choice_n, 3)
    if lengths:
        s = sorted(lengths)
        report['tokens'] = {q: s[min(len(s) - 1, int(q / 100 * len(s)))] for q in (50, 90, 99)} | {'max': s[-1]}
    print(json.dumps(report, indent=2, default=str))
    for tag, ids in examples.items():
        print(f'{tag}: e.g. {ids[:a.show]}', file=sys.stderr)
    hints = []
    if report.get('noul_true_share') is not None and not 0.4 <= report['noul_true_share'] <= 0.6:
        hints.append('noul labels are unbalanced (aim for 40-60% true)')
    if report.get('choice_longest_option_is_gold_share', 0) > 0.35:
        hints.append('the gold choice is often the longest option; downsample those rows (~27% worked for us)')
    if any(t.startswith('ERROR length') for t in problems):
        hints.append('rows over the length limit are dropped by the trainer; pre-render them with render_long.py')
    for h in hints:
        print('HINT:', h, file=sys.stderr)
    if a.output:
        write_jsonl(a.output, kept)
        print(f'wrote {len(kept)} rows to {a.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
