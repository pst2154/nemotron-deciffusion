"""Opt-in explanations by span occlusion: which parts of the state made the chosen answer most probable.

Each candidate span of the state text that was sent to the model is removed (default; "mask" instead replaces it
token for token by the mask token), the question is read again, and the span's weight is
delta = p(chosen | full state) - p(chosen | span removed). Positive spans support the decision; negative spans
(counter-evidence) argue against it. All occluded prompts of a request go upstream together, grouped by candidate
set like the questions themselves, so vLLM batches them.

Spans: string values of the state (the state itself if it is a string, else every string leaf of the JSON, with a
JSONPath) are split into sentences; "phrase" splits sentences at clause punctuation and into ~8-word chunks; "word"
reads sentences first, then the words of the three most supportive sentences. EXPLAIN_MAX_SPANS caps the occluded
reads per request (adjacent spans are merged to fit; the response says so).

Deletion is the default (`"occlusion": "mask"` or EXPLAIN_OCCLUSION=mask selects masking). On GB200 with the 8B-v2 checkpoint, its deltas matched re-rendering the request without the span (Pearson 0.96,
mean abs. error 0.003, same top span 15/16 over 16 router/JSON/Japanese questions) better than mask occlusion
(0.54, 0.048, 11/16): extra mask tokens in the context shift some reads on their own.
"""
import bisect
import json
import math
import os
import re

EXPLAIN_MAX_SPANS = int(os.environ.get('EXPLAIN_MAX_SPANS', '64'))
EXPLAIN_OCCLUSION = os.environ.get('EXPLAIN_OCCLUSION', 'delete')
GRANULARITIES = ('sentence', 'phrase', 'word')
OCCLUSIONS = {'mask': 'mask_occlusion', 'delete': 'span_deletion'}
if EXPLAIN_OCCLUSION not in OCCLUSIONS:
    raise ValueError('EXPLAIN_OCCLUSION must be mask or delete')
DEFAULTS = {'granularity': 'sentence', 'top_k': 5, 'questions': None, 'occlusion': None}
PHRASE_WORDS = 8
WORD_STAGE_SENTENCES = 3

# Sentence ends: Latin terminators before whitespace or end, CJK terminators anywhere, and line breaks.
_SENTENCE_END = re.compile(r'[.!?…]+["”’\')\]]*(?=\s|$)|[。！？!?]+[」』”’）)]*|\n+')
_CLAUSE_END = re.compile(r'[,;:—–](?=\s)|[、，；：]')
_CJK = '぀-ヿ㐀-䶿一-鿿豈-﫿가-힯'
# Words: Latin-like words (with inner apostrophes) or runs of one CJK script (kanji, hiragana, katakana, hangul).
_WORD = re.compile(rf'[㐀-䶿一-鿿豈-﫿々〆]+|[぀-ゟ]+|[゠-ヿー]+'
                   rf'|[가-힯]+|[^\W{_CJK}]+(?:[\'’][^\W{_CJK}]+)*')
# Markers that compaction inserts for omitted text are not evidence.
_OMITTED = re.compile(r'\[\.\.\. [^\]]*omitted[^\]]*\]')
_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def parse_explain(value, questions):
    """Request field -> None (off) or {'granularity', 'top_k', 'questions'}; ValueError for invalid values."""
    if value is None or value is False:
        return None
    if value is True:
        return dict(DEFAULTS, occlusion=EXPLAIN_OCCLUSION)
    if not isinstance(value, dict):
        raise ValueError('explain must be true, false or an object')
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'explain: unknown field {sorted(unknown)[0]}')
    config = {**DEFAULTS, 'occlusion': EXPLAIN_OCCLUSION, **value}
    if config['occlusion'] not in OCCLUSIONS:
        raise ValueError('explain.occlusion must be mask or delete')
    if config['granularity'] not in GRANULARITIES:
        raise ValueError('explain.granularity must be sentence, phrase or word')
    top_k = config['top_k']
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError('explain.top_k must be an integer from 1 to 100')
    ids = config['questions']
    if ids is not None:
        if not isinstance(ids, list) or not ids or not all(isinstance(q, str) and q for q in ids):
            raise ValueError('explain.questions must be a non-empty list of question ids')
        if isinstance(questions, dict):
            missing = [q for q in ids if q not in questions]
            if missing:
                raise ValueError(f'explain.questions: unknown question {missing[0]}')
        config['questions'] = list(dict.fromkeys(ids))
    return config


# ---- spans --------------------------------------------------------------------------------------------------------

def _trimmed(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start and re.search(r'\w', text[start:end]) else None


def _split(text, pattern, start=0, end=None):
    end = len(text) if end is None else end
    out, pos = [], start
    for match in pattern.finditer(text, start, end):
        piece = _trimmed(text, pos, match.end())
        if piece:
            out.append(piece)
        pos = match.end()
    piece = _trimmed(text, pos, end)
    if piece:
        out.append(piece)
    return out


def sentences(text):
    """(start, end) character ranges of the sentences of `text`, skipping compaction markers."""
    out, pos = [], 0
    for marker in _OMITTED.finditer(text):
        out += _split(text, _SENTENCE_END, pos, marker.start())
        pos = marker.end()
    return out + _split(text, _SENTENCE_END, pos)


def words(text, start=0, end=None):
    end = len(text) if end is None else end
    return [m.span() for m in _WORD.finditer(text, start, end)]


def phrases(text):
    """Clauses of each sentence, with clauses longer than PHRASE_WORDS words cut into chunks of that many words."""
    out = []
    for s_start, s_end in sentences(text):
        for c_start, c_end in _split(text, _CLAUSE_END, s_start, s_end):
            units = words(text, c_start, c_end)
            if len(units) <= PHRASE_WORDS:
                out.append((c_start, c_end))
                continue
            for k in range(0, len(units), PHRASE_WORDS):
                chunk = units[k:k + PHRASE_WORDS]
                out.append((chunk[0][0], chunk[-1][1]))
    return out


def json_path(path):
    out = '$'
    for key in path:
        if isinstance(key, int):
            out += f'[{key}]'
        elif _IDENTIFIER.match(key):
            out += f'.{key}'
        else:
            out += f'[{json.dumps(key, ensure_ascii=False)}]'
    return out


def locate_leaves(state):
    """(serialized, [(path, offset, text)]): `serialized` equals json.dumps(state, ensure_ascii=False) and `offset`
    is where each non-empty string leaf's escaped text starts in it."""
    parts, leaves, size = [], [], [0]

    def emit(text):
        parts.append(text)
        size[0] += len(text)

    def walk(value, path):
        if isinstance(value, str):
            if value:
                leaves.append((path, size[0] + 1, value))
            emit(json.dumps(value, ensure_ascii=False))
        elif isinstance(value, dict) and value:
            emit('{')
            for i, (key, item) in enumerate(value.items()):
                if i:
                    emit(', ')
                emit(json.dumps({key: None}, ensure_ascii=False)[1:-5])  # '"key": '
                walk(item, path + (str(key) if not isinstance(key, str) else key,))
            emit('}')
        elif isinstance(value, list) and value:
            emit('[')
            for i, item in enumerate(value):
                if i:
                    emit(', ')
                walk(item, path + (i,))
            emit(']')
        else:
            emit(json.dumps(value, ensure_ascii=False))

    walk(state, ())
    serialized = ''.join(parts)
    if serialized != json.dumps(state, ensure_ascii=False):
        raise RuntimeError('State serialization mismatch')
    return serialized, leaves


def _escaped_length(char):
    if char in '"\\\n\r\t\b\f':
        return 2
    return 6 if char < ' ' else 1


def escaped_offsets(text):
    """offsets[i] = position of text[i] inside json.dumps(text, ensure_ascii=False) after the opening quote."""
    out, pos = [0], 0
    for char in text:
        pos += _escaped_length(char)
        out.append(pos)
    return out


def leaf_spans(state, granularity):
    """Candidate spans of every string leaf: dicts with leaf index, path, start, end, text."""
    _, leaves = locate_leaves(state)
    split = phrases if granularity == 'phrase' else sentences
    spans = []
    for index, (path, _, text) in enumerate(leaves):
        for start, end in split(text):
            spans.append({'leaf': index, 'path': json_path(path), 'start': start, 'end': end, 'text': text[start:end]})
    return leaves, spans


def fit_spans(spans, budget, leaves):
    """At most `budget` spans: adjacent spans of the same leaf are merged into equal-size groups (whole leaves, the
    longest first, if there are more leaves than budget). Returns (spans, merged)."""
    if len(spans) <= budget:
        return spans, False
    if budget <= 0:
        return [], True
    by_leaf = {}
    for span in spans:
        by_leaf.setdefault(span['leaf'], []).append(span)
    if len(by_leaf) >= budget:
        longest = sorted(by_leaf, key=lambda i: -(by_leaf[i][-1]['end'] - by_leaf[i][0]['start']))[:budget]
        groups = [by_leaf[i] for i in sorted(longest)]
    else:
        # Per-leaf group counts proportional to span counts, at least one each, summing to the budget.
        counts = {i: max(1, len(group) * budget // len(spans)) for i, group in by_leaf.items()}
        while sum(counts.values()) > budget:  # the at-least-one floors can overshoot
            counts[max(counts, key=counts.get)] -= 1
        spare = budget - sum(counts.values())
        for i in sorted(by_leaf, key=lambda i: -len(by_leaf[i])):
            if spare <= 0:
                break
            if counts[i] < len(by_leaf[i]):
                counts[i] += 1
                spare -= 1
        groups = []
        for i, group in by_leaf.items():
            k = min(counts[i], len(group))
            bounds = [round(j * len(group) / k) for j in range(k + 1)]
            groups += [group[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
    out = []
    for group in groups:
        first, last = group[0], group[-1]
        text = leaves[first['leaf']][2]
        out.append({**first, 'end': last['end'], 'text': text[first['start']:last['end']]})
    return out, True


# ---- masking ------------------------------------------------------------------------------------------------------

class Occluder:
    """Maps leaf character ranges of one scored prompt to its token positions and builds masked copies."""

    def __init__(self, tokenizer, rendered, state, ids, mask_token_id):
        serialized, self.leaves = locate_leaves(state)
        anchor = '{"state": ' + serialized
        at = rendered.find(anchor)
        if at < 0:
            raise RuntimeError('State not found in the rendered prompt')
        self.base = at + len('{"state": ')
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded['input_ids']) != list(ids):
            raise RuntimeError('Tokenization with offsets does not reproduce the scored prompt')
        offsets = encoded['offset_mapping']
        self.starts = [a for a, _ in offsets]
        self.ends = [b for _, b in offsets]
        self.monotonic = all(a <= b for a, b in zip(self.starts, self.starts[1:])) and \
            all(a <= b for a, b in zip(self.ends, self.ends[1:]))
        self.ids, self.mask_token_id = list(ids), mask_token_id
        self._escaped = {}

    def char_range(self, span):
        """Rendered-prompt character range of a leaf span."""
        index = span['leaf']
        if index not in self._escaped:
            self._escaped[index] = escaped_offsets(self.leaves[index][2])
        offsets, origin = self._escaped[index], self.base + self.leaves[index][1]
        return origin + offsets[span['start']], origin + offsets[span['end']]

    def token_range(self, span):
        """Token positions [i, j) overlapping the span's characters."""
        a, b = self.char_range(span)
        if self.monotonic:
            return bisect.bisect_right(self.ends, a), bisect.bisect_left(self.starts, b)
        hits = [k for k, (s, e) in enumerate(zip(self.starts, self.ends)) if s < b and e > a]
        return (hits[0], hits[-1] + 1) if hits else (0, 0)

    def occluded(self, span, occlusion='mask'):
        """Prompt ids with the span's tokens masked (same length) or deleted; None if the span has no tokens."""
        i, j = self.token_range(span)
        if j <= i:
            return None
        fill = [self.mask_token_id] * (j - i) if occlusion == 'mask' else []
        return self.ids[:i] + fill + self.ids[j:]


# ---- scoring ------------------------------------------------------------------------------------------------------

def softmax_at(logprobs, candidate_ids, index):
    values = [logprobs[token] for token in candidate_ids]
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError('Non-finite candidate logprobs')
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    return weights[index] / sum(weights)


def budgets(total, count):
    """Split `total` occluded reads over `count` questions."""
    base, extra = divmod(total, count) if count else (0, 0)
    return [base + (1 if i < extra else 0) for i in range(count)]


def ranked(spans, deltas, top_k):
    rows = [{'text': s['text'], 'path': s['path'], 'start': s['start'], 'end': s['end'], 'delta': round(d, 6)}
            for s, d in zip(spans, deltas)]
    support = sorted((r for r in rows if r['delta'] > 0), key=lambda r: -r['delta'])[:top_k]
    against = sorted((r for r in rows if r['delta'] < 0), key=lambda r: r['delta'])[:top_k]
    return support, against


def word_spans(leaves, sentence_spans, deltas, budget):
    """Second stage of word granularity: the words of the top sentences, best sentence first, up to `budget`."""
    order = sorted(range(len(sentence_spans)), key=lambda i: -deltas[i])
    positive = [i for i in order if deltas[i] > 0]
    chosen = (positive or order)[:WORD_STAGE_SENTENCES]
    out = []
    for i in chosen:
        sentence = sentence_spans[i]
        text = leaves[sentence['leaf']][2]
        out += [{'leaf': sentence['leaf'], 'path': sentence['path'], 'start': a, 'end': b, 'text': text[a:b]}
                for a, b in words(text, sentence['start'], sentence['end'])]
    return out[:budget], len(out) > budget


def skipped(config):
    return {'method': OCCLUSIONS[config['occlusion']], 'granularity': config['granularity'], 'spans': [], 'counter_evidence': [],
            'truncated': False, 'notes': ['Explanations cover choice and noul answers; score answers are an '
                                          'expected level rather than one chosen option.']}


class Job:
    """Explanation of one answered question: spans, their occluded prompts and, once read, their deltas."""

    def __init__(self, key, answer, read, original_state, config):
        chosen = answer['choice'] if answer['type'] == 'choice' else ('true' if answer['noul'] >= 0.5 else 'false')
        if chosen not in read['labels'] and 'first' in read:
            read = read['first']
        self.key, self.read, self.config, self.chosen = key, read, config, chosen
        self.index = read['labels'].index(chosen)
        self.base = softmax_at(read['logprobs'], read['candidate_ids'], self.index)
        self.notes = []
        if read['state'] is not original_state:
            self.notes.append('Spans refer to the compacted state that was scored.')
        if read.get('round') == 'elimination':
            self.notes.append('Explains the elimination read between the two finalists.')
        self.limit, self.truncated, self.prompts, self.pending, self.stages = 0, False, [], None, []
        self.occluder = self.leaves = None

    def _queue(self, spans):
        self.pending, self.prompts = [], []
        for span in spans:
            ids = self.occluder.occluded(span, self.config['occlusion'])
            if ids is not None:
                self.pending.append(span)
                self.prompts.append(ids + [self.occluder.mask_token_id])

    def first_stage(self, tokenizer, mask_token_id):
        read = self.read
        try:
            self.occluder = Occluder(tokenizer, read['prompt'], read['state'], read['ids'][:-1], mask_token_id)
        except RuntimeError as exc:
            self.notes.append(f'No explanation: {exc}.')
            return
        granularity = self.config['granularity']
        self.leaves, spans = leaf_spans(read['state'], 'sentence' if granularity == 'word' else granularity)
        if not spans:
            self.notes.append('The scored state has no text to explain.')
            return
        budget = self.limit if granularity != 'word' else min(len(spans), max(1, self.limit // 2))
        fitted, merged = fit_spans(spans, budget, self.leaves)
        if merged:
            self.truncated = True
            self.notes.append(f'{len(spans)} {granularity if granularity != "word" else "sentence"} spans were '
                              f'grouped into {len(fitted)} to stay within {EXPLAIN_MAX_SPANS} reads per request.')
        self._queue(fitted)

    def record(self, logprobs):
        if self.pending is None:  # nothing was queued in this stage
            return
        deltas = [self.base - softmax_at(values, self.read['candidate_ids'], self.index) for values in logprobs]
        self.stages.append((self.pending, deltas))
        self.prompts, self.pending = [], None

    def second_stage(self):
        """Word granularity: words of the top sentences."""
        self.prompts = []
        if not self.stages:
            return
        sentence_spans, deltas = self.stages[0]
        room = self.limit - len(sentence_spans)
        if room <= 0:
            self.truncated = True
            self.notes.append('No reads were left for words; sentence spans are reported.')
            return
        spans, cut = word_spans(self.leaves, sentence_spans, deltas, room)
        if cut:
            self.truncated = True
            self.notes.append(f'Word reads stopped at {room} to stay within {EXPLAIN_MAX_SPANS} reads per request.')
        self._queue(spans)

    def explanation(self):
        spans, deltas = self.stages[-1] if self.stages else ([], [])
        support, against = ranked(spans, deltas, self.config['top_k'])
        out = {'method': OCCLUSIONS[self.config['occlusion']], 'granularity': self.config['granularity'], 'decision': self.chosen,
               'base_probability': self.base, 'spans': support, 'counter_evidence': against,
               'truncated': self.truncated}
        if self.notes:
            out['notes'] = self.notes
        return out
