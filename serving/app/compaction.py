"""Long-state compaction applied before the single scoring pass, without any model call.

Prose leaves are split into sentences, ranked against the question and options (BM25 + pseudo-relevance term
expansion) and the best sentences are kept in their original order. Trace-like values (keys such as trajectory,
log, messages, steps, history) keep their opening (25% of their share) and their most recent steps instead,
because generic questions about an agent run share no keywords with the trace.
Measured with an 8192-token gate and budget: BABILong 32k/64k 64/54% -> 98/98%, QuALITY unchanged. Compaction
hurts whole-document tasks on short inputs, so it only runs above the length gate.
"""
import json
import math
import re

_STOP = frozenset('the a an is are was were of to in and or for with this that it be as on by from at what where '
                  'who which how did does do has have had not no yes'.split())
_SPLIT = re.compile(r'(?<=[.!?])["”\']?\s+|\n\s*\n')


def _words(text):
    return [w for w in re.findall(r'\w+', text.lower()) if w not in _STOP]


def _sentences(text):
    out = []
    for part in _SPLIT.split(text):
        words = part.split()
        if not words:
            continue
        if len(words) > 90:  # unpunctuated runs: fixed word chunks
            out += [' '.join(words[k:k + 70]) for k in range(0, len(words), 70)]
        else:
            out.append(' '.join(words))
    return out


def _bm25(docs, idf, avgdl, weights, k1=1.2, b=0.75):
    scores = []
    for words in docs:
        tf = {}
        for w in words:
            if w in weights:
                tf[w] = tf.get(w, 0) + 1
        dl = len(words)
        scores.append(sum(weights[w] * idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
                          for w, f in tf.items()))
    return scores


def rank(sentences, instructions, options_text):
    docs = [_words(s) for s in sentences]
    n = len(docs)
    df = {}
    for words in docs:
        for w in set(words):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
    avgdl = max(1.0, sum(map(len, docs)) / max(1, n))
    query = {w: 0.3 for w in _words(options_text)}
    query.update({w: 1.0 for w in _words(instructions)})
    first = _bm25(docs, idf, avgdl, query)
    seeds = [j for j in sorted(range(n), key=lambda j: -first[j])[:5] if first[j] > 0]
    expand = {w: 0.5 for j in seeds for w in docs[j] if w not in query}
    if not expand:
        return first
    return [a + b for a, b in zip(first, _bm25(docs, idf, avgdl, expand))]


def _leaves(value, path=()):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _leaves(v, path + (i,))


def _replace(value, path, new):
    if not path:
        return new
    head, rest = path[0], path[1:]
    if isinstance(value, dict):
        return {k: (_replace(v, rest, new) if k == head else v) for k, v in value.items()}
    return [(_replace(v, rest, new) if i == head else v) for i, v in enumerate(value)]


_TRACE_KEY = re.compile(r'(?:^|[_\W])(traj\w*|transcripts?|traces?|logs?|history|conversations?|messages|steps|events|turns)'
                        r'(?:$|[_\W])', re.I)
HEAD_SHARE = 0.25  # agent traces: keep the opening (task) and the most recent steps


def _is_trace(path):
    return any(isinstance(k, str) and _TRACE_KEY.search(k) for k in path)


def _head_tail_text(tokenizer, text, budget):
    """Keep whole steps (blank-line separated blocks, else lines) from the start (HEAD_SHARE of budget) and the
    end; mark the omitted middle."""
    sep = '\n\n' if text.count('\n\n') >= 2 else '\n'
    lines = text.split(sep)
    ntok = [len(x) for x in tokenizer(lines, add_special_tokens=False)['input_ids']]
    if sum(ntok) <= budget:
        return text
    head, used = [], 0
    for line, n in zip(lines, ntok):
        if used + n > HEAD_SHARE * budget:
            break
        head.append(line); used += n
    tail, used_tail = [], 0
    for line, n in zip(reversed(lines[len(head):]), reversed(ntok[len(head):])):
        if used + used_tail + n > budget:
            break
        tail.append(line); used_tail += n
    omitted = len(lines) - len(head) - len(tail)
    unit = 'earlier steps' if sep == '\n\n' else 'lines'
    return sep.join(head + [f'[... {omitted} {unit} omitted ...]'] + tail[::-1])


def _head_tail_list(tokenizer, items, budget):
    """Trace given as a list of steps/messages: keep the first and the most recent items."""
    texts = [json.dumps(x, ensure_ascii=False) for x in items]
    ntok = [len(x) for x in tokenizer(texts, add_special_tokens=False)['input_ids']]
    if sum(ntok) <= budget or len(items) < 3:
        return items
    head, used = 0, 0
    while head < len(items) - 1 and used + ntok[head] <= HEAD_SHARE * budget:
        used += ntok[head]; head += 1
    head = max(head, 1)
    used = sum(ntok[:head])
    tail = 0
    while head + tail < len(items) and used + ntok[len(items) - 1 - tail] <= budget:
        used += ntok[len(items) - 1 - tail]; tail += 1
    omitted = len(items) - head - tail
    return items[:head] + [f'[... {omitted} items omitted ...]'] + (items[len(items) - tail:] if tail else [])


def _trace_lists(value, path=()):
    if isinstance(value, list):
        if _is_trace(path) and len(value) >= 3:
            yield path, value
            return
        for i, v in enumerate(value):
            yield from _trace_lists(v, path + (i,))
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _trace_lists(v, path + (k,))


def compact_state(tokenizer, state, instructions, options, budget, min_leaf_tokens=256):
    """Return (state, info): long state content reduced to at most `budget` tokens in total.

    Agent traces (keys like trajectory/log/messages) keep their opening and most recent steps; other long
    text keeps its most question-relevant sentences (BM25). The budget is shared in proportion to size.
    """
    lists = list(_trace_lists(state))
    list_tok = [len(tokenizer.encode(json.dumps(v, ensure_ascii=False), add_special_tokens=False)) for _, v in lists]
    leaves = [(p, s) for p, s in _leaves(state) if len(s) >= 4 * min_leaf_tokens
              and not any(p[:len(lp)] == lp for lp, _ in lists)]
    traces = [(p, s) for p, s in leaves if _is_trace(p)]
    leaves = [(p, s) for p, s in leaves if not _is_trace(p)]
    trace_tok = [len(tokenizer.encode(s, add_special_tokens=False)) for _, s in traces]
    prose_tok = sum(len(tokenizer.encode(s, add_special_tokens=False)) for _, s in leaves) if leaves else 0
    total = sum(list_tok) + sum(trace_tok) + prose_tok
    if total <= budget:
        return state, None
    out = state
    for (path, value), n in zip(lists, list_tok):
        out = _replace(out, path, _head_tail_list(tokenizer, value, int(budget * n / total)))
    for (path, text), n in zip(traces, trace_tok):
        out = _replace(out, path, _head_tail_text(tokenizer, text, int(budget * n / total)))
    if not leaves:
        return out, dict(mode='head_tail', tokens_before=total)
    out, info = _compact_prose(tokenizer, out, leaves, instructions, options, int(budget * prose_tok / total) or 1)
    return out, dict(info or {}, tokens_before=total)


def _compact_prose(tokenizer, state, leaves, instructions, options, budget):
    units = []  # (leaf index, sentence index, text)
    for li, (_, text) in enumerate(leaves):
        units += [(li, si, s) for si, s in enumerate(_sentences(text))]
    if not units:
        return state, None
    texts = [u[2] for u in units]
    ntok = [len(x) for x in tokenizer(texts, add_special_tokens=False)['input_ids']]
    if sum(ntok) <= budget:
        return state, None
    scores = rank(texts, instructions, json.dumps(options, ensure_ascii=False))
    keep, used = set(), 0
    for j in sorted(range(len(units)), key=lambda j: -scores[j]):
        if used + ntok[j] > budget:
            if used >= 0.9 * budget:
                break
            continue
        keep.add(j)
        used += ntok[j]
    out = state
    for li, (path, _) in enumerate(leaves):
        kept = [units[j][2] for j in sorted(keep) if units[j][0] == li]
        out = _replace(out, path, ' '.join(kept))
    return out, dict(sentences=len(units), kept=len(keep), tokens_before=sum(ntok), tokens_after=used)
