#!/usr/bin/env python3
"""Convert a labelled CSV/JSONL table into Deciffusion training rows, with a task-disjoint held-out split.

One input record = one decision. Columns (rename with the --*-col flags):

  state          JSON object/array as text, or plain text (wrapped as {"text": ...}); or use --state-cols a,b,c
                 to build the state from several columns
  question       the instructions shown to the model
  options        choice: JSON list ["...", ...], JSON object {"key": "description"}, or "a|b|c"
                 score:  JSON list of level descriptions ordered lowest -> highest (level i = index i)
                 noul:   leave empty, or {"true": "...", "false": "..."} descriptions
  label          choice: option key, option text, or 0-based index; noul: yes/no/true/false/1/0;
                 score: level index or level text
  type           optional (choice|noul|score); else --type, else inferred (yes/no label and no options -> noul)
  probabilities  optional soft label: JSON object {label|option text: p} or list aligned with options;
                 for noul also a single number p(true)
  group          optional; rows sharing a group never straddle the train/held-out split (default: hash of state)

Augmentation (train split only): option order shuffled per row (never for score levels, which are ordinal),
option keys re-coded per row (--key-style mixed: semantic slugs, letters, opaque ids, option_i), optional
instruction paraphrases (--instructions-file), optional --variants K copies per row. The trainer additionally
shuffles option order on the fly for half the rows (train_pack.py --canonical-fraction 0.5).

Balancing (train split only, --balance): noul true/false downsampled to equal counts; choice and score classes
capped at --balance-ratio x the median class size; rows whose gold is the strictly longest option capped at
--max-longest-share.

Example:
  python3 finetune/convert_labeled.py --input tickets.csv --type choice --domain support-routing \\
      --instructions-file paraphrases.txt --eval-fraction 0.1 --balance --output data/task
  -> data/task/train.jsonl, data/task/heldout.jsonl, data/task/convert_report.json
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalize_noul, stable_hash, write_jsonl  # noqa: E402


def parse_jsonish(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] in '[{' or text in ('true', 'false', 'null') or re.fullmatch(r'-?\d+(\.\d+)?', text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return value


def read_records(path):
    path = Path(path)
    if path.suffix.lower() in ('.csv', '.tsv'):
        with path.open(newline='') as f:
            return list(csv.DictReader(f, delimiter='\t' if path.suffix.lower() == '.tsv' else ','))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def slug(text, used):
    base = re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')[:40] or 'option'
    key, n = base, 2
    while key in used:
        key, n = f'{base}_{n}', n + 1
    used.add(key)
    return key


def make_keys(texts, style, rng, original=None):
    n = len(texts)
    if style == 'mixed':
        style = rng.choices(['semantic', 'letters', 'opaque', 'numbered'], [0.4, 0.2, 0.2, 0.2])[0]
    if style == 'letters' and n > 26:
        style = 'numbered'
    if style == 'keep' and original:
        return list(original)
    if style == 'letters':
        return [chr(65 + i) for i in range(n)]
    if style == 'opaque':
        keys = set()
        while len(keys) < n:
            keys.add(f'k_{rng.getrandbits(32):08x}')
        return list(keys)
    if style == 'numbered':
        return [f'option_{i}' for i in range(n)]
    used = set()
    return [slug(t, used) for t in texts]


class Converter:
    def __init__(self, a):
        self.a = a
        self.variants_by_question = {}
        self.global_variants = []
        if a.instructions_file:
            text = Path(a.instructions_file).read_text()
            try:
                obj = json.loads(text)
                self.variants_by_question = {k: list(v) for k, v in obj.items()}
            except json.JSONDecodeError:
                self.global_variants = [l.strip() for l in text.splitlines() if l.strip()]

    def col(self, rec, name):
        return rec.get(name) if name else None

    def state_of(self, rec):
        a = self.a
        if a.state_cols:
            return {c: parse_jsonish(rec.get(c)) for c in a.state_cols.split(',')}
        raw = rec.get(a.state_col)
        val = parse_jsonish(raw)
        if isinstance(val, (dict, list)):
            return val
        if val is None:
            raise ValueError('empty state')
        return {a.state_key: str(val)}

    def instructions(self, question, rng, augment):
        if not augment or rng.random() >= self.a.vary_instructions:
            return question
        pool = self.variants_by_question.get(question) or self.global_variants
        return rng.choice(pool) if pool else question

    def kind_of(self, rec, options):
        k = self.col(rec, self.a.type_col) or self.a.type
        if k:
            k = str(k).strip().lower()
            if k not in ('choice', 'noul', 'score'):
                raise ValueError(f'unknown type {k!r}')
            return k
        if not options and normalize_noul(rec.get(self.a.label_col)) is not None:
            return 'noul'
        return 'choice'

    def options_of(self, rec):
        raw = self.col(rec, self.a.options_col)
        val = parse_jsonish(raw)
        if val is None:
            return None
        if isinstance(val, str):
            val = [x.strip() for x in val.split(self.a.option_sep) if x.strip()]
        return val

    def convert(self, rec, idx, rng, augment):
        a = self.a
        state = self.state_of(rec)
        question = self.col(rec, a.question_col) or a.question
        if not question:
            raise ValueError('no question (set --question-col or --question)')
        options = self.options_of(rec)
        kind = self.kind_of(rec, options)
        label_raw = rec.get(a.label_col)
        probs_raw = parse_jsonish(self.col(rec, a.probs_col))
        q = {'type': kind, 'instructions': self.instructions(str(question), rng, augment)}
        teacher = None

        if kind == 'noul':
            if isinstance(options, dict) and set(options) >= {'true', 'false'}:
                q['criteria'] = {'true': options['true'], 'false': options['false']}
            target = normalize_noul(label_raw) if label_raw not in (None, '') else None
            if probs_raw is not None:
                if isinstance(probs_raw, (int, float)):
                    pt = float(probs_raw)
                else:
                    norm = {normalize_noul(k): float(v) for k, v in probs_raw.items()}
                    pt = norm.get('true', 0.0) / max(1e-12, norm.get('true', 0.0) + norm.get('false', 0.0))
                teacher = {'false': 1 - pt, 'true': pt}
                target = target or ('true' if pt >= 0.5 else 'false')
            if target is None:
                raise ValueError(f'noul label {label_raw!r} is not yes/no/true/false/1/0')
            gold_text = target

        elif kind == 'score':
            if not isinstance(options, list) or len(options) < 2:
                raise ValueError('score needs a list of >= 2 level descriptions')
            q['criteria'] = list(options)
            labels = [str(i) for i in range(len(options))]
            target = self.match_index(label_raw, options)
            if probs_raw is not None:
                teacher = self.map_probs(probs_raw, list(options), labels, labels)
                if target is None:
                    target = max(range(len(labels)), key=lambda i: teacher[labels[i]])
            if target is None:
                raise ValueError(f'score label {label_raw!r} matches no level')
            target, gold_text = str(target), str(options[target])

        else:  # choice
            if isinstance(options, dict):
                orig_keys, texts = list(options), [options[k] for k in options]
            elif isinstance(options, list):
                orig_keys, texts = None, list(options)
            else:
                raise ValueError('choice needs options')
            if len(texts) < 2:
                raise ValueError('choice needs >= 2 options')
            gi = None
            if label_raw not in (None, ''):
                gi = self.match_index(label_raw, texts, orig_keys)
                if gi is None:
                    raise ValueError(f'label {label_raw!r} matches no option key, text or index')
            order = list(range(len(texts)))
            if augment and not a.no_shuffle:
                rng.shuffle(order)
            keys = make_keys([texts[i] for i in order], a.key_style, rng,
                             [orig_keys[i] for i in order] if orig_keys else None)
            q['criteria'] = {k: texts[i] for k, i in zip(keys, order)}
            if probs_raw is not None:
                src_labels = orig_keys or [str(t) for t in texts]
                teacher_src = self.map_probs(probs_raw, texts, src_labels, src_labels)
                teacher = {k: teacher_src[src_labels[i]] for k, i in zip(keys, order)}
                if gi is None:
                    gi = max(range(len(texts)), key=lambda i: teacher_src[src_labels[i]])
            if gi is None:
                raise ValueError('no label and no probabilities')
            target = keys[order.index(gi)]
            gold_text = str(texts[gi])

        rid = str(rec.get(a.id_col) or f'{a.id_prefix}:{idx}')
        group = str(rec.get(a.group_col) or '') if a.group_col else ''
        row = {'id': rid,
               'input': {'state': state, 'questions': {'decision': q}},
               'reference': {'target': target, 'source': a.source},
               'provenance': {'source_id': group or stable_hash(state)[:16], 'source_file': Path(a.input).name},
               'domain': str(rec.get(a.domain_col) or a.domain) if a.domain_col else a.domain}
        if teacher is not None:
            s = sum(teacher.values())
            if s <= 0:
                raise ValueError('probabilities sum to 0')
            row['teacher_probabilities'] = {k: v / s for k, v in teacher.items()}
        return row, gold_text

    @staticmethod
    def match_index(label, texts, keys=None):
        if label is None or label == '':
            return None
        s = str(label).strip()
        if keys and s in keys:
            return keys.index(s)
        strs = [str(t).strip() for t in texts]
        if s in strs:
            return strs.index(s)
        low = [t.lower() for t in strs]
        if s.lower() in low:
            return low.index(s.lower())
        if re.fullmatch(r'\d+', s) and int(s) < len(texts):
            return int(s)
        return None

    @staticmethod
    def map_probs(raw, texts, src_labels, out_labels):
        """Map a soft label given by key, option text or position onto out_labels (aligned with texts)."""
        if isinstance(raw, list):
            if len(raw) != len(texts):
                raise ValueError('probability list length != number of options')
            return {out_labels[i]: float(p) for i, p in enumerate(raw)}
        out = {l: 0.0 for l in out_labels}
        for k, p in raw.items():
            i = Converter.match_index(k, texts, src_labels)
            if i is None:
                raise ValueError(f'probability key {k!r} matches no option')
            out[out_labels[i]] += float(p)
        return out


def balance(rows, gold_texts, a, rng, report):
    """Downsample within each question type: equal noul labels, capped choice/score classes, capped longest-correct."""
    by_type = defaultdict(list)
    for r, g in zip(rows, gold_texts):
        by_type[r['input']['questions']['decision']['type']].append((r, g))
    kept = []
    for kind, items in by_type.items():
        classes = defaultdict(list)
        for r, g in items:
            classes[g].append((r, g))
        sizes = [len(v) for v in classes.values()]
        if kind == 'noul' and len(classes) == 2:
            cap = min(sizes)
        elif len(classes) > 1:
            cap = max(1, int(a.balance_ratio * median(sizes)))
        else:
            cap = sizes[0]
        for g, v in classes.items():
            rng.shuffle(v)
            kept += v[:cap]
        report[f'balance/{kind}'] = {'before': len(items), 'after': sum(min(cap, s) for s in sizes), 'cap': cap}
    rng.shuffle(kept)
    return [r for r, _ in kept], [g for _, g in kept]


def longest_is_gold(row):
    q = row['input']['questions']['decision']
    if q['type'] != 'choice':
        return False
    lens = {k: len(json.dumps(v, ensure_ascii=False)) for k, v in q['criteria'].items()}
    gold = str(row['reference']['target'])
    top = max(lens.values())
    return lens.get(gold) == top and sum(1 for v in lens.values() if v == top) == 1


def cap_longest(rows, share, rng, report):
    longest = [r for r in rows if longest_is_gold(r)]
    choice = sum(1 for r in rows if r['input']['questions']['decision']['type'] == 'choice')
    if not choice:
        return rows
    report['longest_option_is_gold_before'] = round(len(longest) / choice, 3)
    others = choice - len(longest)
    allowed = int(share * others / (1 - share)) if share < 1 else len(longest)
    if len(longest) <= allowed:
        return rows
    rng.shuffle(longest)
    drop = {id(r) for r in longest[allowed:]}
    out = [r for r in rows if id(r) not in drop]
    report['longest_option_is_gold_after'] = round(allowed / (others + allowed), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='CSV/TSV or JSONL file')
    p.add_argument('--output', required=True, help='output directory (train.jsonl, heldout.jsonl, convert_report.json)')
    p.add_argument('--state-col', default='state')
    p.add_argument('--state-cols', default='', help='comma list of columns forming the state object')
    p.add_argument('--state-key', default='text', help='key used to wrap a plain-text state')
    p.add_argument('--question-col', default='question')
    p.add_argument('--question', default='', help='fixed instructions when there is no question column')
    p.add_argument('--options-col', default='options')
    p.add_argument('--option-sep', default='|')
    p.add_argument('--label-col', default='label')
    p.add_argument('--type-col', default='type')
    p.add_argument('--type', choices=['choice', 'noul', 'score'], default=None, help='force a question type')
    p.add_argument('--probs-col', default='probabilities')
    p.add_argument('--group-col', default='', help='rows with the same group stay on one side of the split')
    p.add_argument('--id-col', default='id')
    p.add_argument('--id-prefix', default='custom')
    p.add_argument('--domain', default='custom')
    p.add_argument('--domain-col', default='')
    p.add_argument('--source', default='labeled', help='reference.source value')
    p.add_argument('--key-style', choices=['mixed', 'semantic', 'letters', 'opaque', 'numbered', 'keep'],
                   default='mixed', help='choice option keys (chosen per row; keep = original keys of an options object)')
    p.add_argument('--no-shuffle', action='store_true', help='keep the given choice option order')
    p.add_argument('--instructions-file', default='',
                   help='paraphrases: a text file (one per line, applies to every row) or a JSON object '
                        '{original question: [paraphrases]}')
    p.add_argument('--vary-instructions', type=float, default=0.5, help='probability of using a paraphrase')
    p.add_argument('--variants', type=int, default=1, help='augmented copies per train row')
    p.add_argument('--eval-fraction', type=float, default=0.1)
    p.add_argument('--balance', action='store_true')
    p.add_argument('--balance-ratio', type=float, default=1.5)
    p.add_argument('--max-longest-share', type=float, default=0.3,
                   help='cap on choice rows whose gold is the strictly longest option (1 disables)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    rng = random.Random(a.seed)
    conv = Converter(a)
    records = read_records(a.input)
    train, train_gold, held, errors, seen = [], [], [], Counter(), set()
    for idx, rec in enumerate(records):
        try:
            probe, _ = conv.convert(rec, idx, random.Random(f'{a.seed}/{idx}'), augment=False)
        except (ValueError, KeyError, TypeError) as e:
            errors[str(e)[:80]] += 1
            continue
        group = probe['provenance']['source_id']
        in_eval = int(stable_hash(group, f'split/{a.seed}/')[:8], 16) / 0xFFFFFFFF < a.eval_fraction
        if in_eval:
            held.append(probe)
            continue
        for v in range(a.variants):
            row, gold_text = conv.convert(rec, idx, random.Random(f'{a.seed}/{idx}/{v}'), augment=True)
            q = row['input']['questions']['decision']
            key = stable_hash([row['input']['state'], q['instructions'], q.get('criteria')])
            if key in seen:  # identical variant (same keys, order and phrasing): skip
                continue
            seen.add(key)
            if a.variants > 1:
                row['id'] = f"{row['id']}:v{v}"
            train.append(row); train_gold.append(gold_text)

    report = {'records': len(records), 'errors': dict(errors.most_common(20))}
    if a.balance:
        train, train_gold = balance(train, train_gold, a, rng, report)
    train = cap_longest(train, a.max_longest_share, rng, report)
    rng.shuffle(train)

    out = Path(a.output)
    write_jsonl(out / 'train.jsonl', train)
    write_jsonl(out / 'heldout.jsonl', held)

    def dist(rows):
        c = Counter()
        for r in rows:
            q = r['input']['questions']['decision']
            g = str(r['reference']['target']) if q['type'] != 'choice' else 'choice'
            c[f"{q['type']}/{g}"] += 1
        return dict(sorted(c.items()))
    report.update(train=len(train), heldout=len(held), train_labels=dist(train), heldout_labels=dist(held))
    (out / 'convert_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if errors:
        print(f'WARNING: {sum(errors.values())} records skipped; see errors above', file=sys.stderr)


if __name__ == '__main__':
    main()
