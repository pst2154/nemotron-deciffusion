"""Opt-in mask-occlusion explanations. CPU only: a character-level tokenizer and a fake upstream whose candidate logits
count keywords in the unmasked prompt, so the expected evidence is known exactly."""
import asyncio
import copy
import math
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import adapter  # noqa: E402
import explain  # noqa: E402


@pytest.fixture(autouse=True)
def _mask_occlusion_default(monkeypatch):
    """The fake upstream's exact deltas below assume masking; the server default (deletion) is tested separately."""
    monkeypatch.setattr(explain, 'EXPLAIN_OCCLUSION', 'mask')
import scoring  # noqa: E402

MASK = 100
BASE = 2000


class Tokenizer:
    """One token per character (ids >= BASE), so offsets are exact and the mask id never collides."""

    def encode(self, text, **kwargs):
        return [BASE + ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        # Like a Jinja chat template: message contents are inserted verbatim.
        return ''.join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + '<assistant>'

    def __call__(self, prompts, return_offsets_mapping=False, **kwargs):
        if isinstance(prompts, str):
            out = {'input_ids': self.encode(prompts)}
            if return_offsets_mapping:
                out['offset_mapping'] = [(i, i + 1) for i in range(len(prompts))]
            return out
        return {'input_ids': [self.encode(p) for p in prompts]}


def decode(ids):
    return ''.join(chr(t - BASE) if t >= BASE else '\0' for t in ids)


# Candidate i's logit is 2 x the number of unmasked occurrences of KEYWORDS[i] (by code position). Two-candidate
# reads are noul (false, true) and elimination finalists (billing, shipping).
KEYWORDS = {2: ['refund', 'blocked'], 3: ['refund', 'delivery', 'crash'], 4: ['refund', 'delivery', 'crash', 'login']}


class Upstream:
    def __init__(self):
        self.calls = []

    async def post(self, path, body):
        self.calls.append(copy.deepcopy(body))
        ids = body.get('logprob_token_ids') or body['allowed_token_ids']
        words = KEYWORDS[len(ids)]
        choices = []
        for i, prompt in enumerate(body['prompt']):
            # Only the state is evidence: count inside the user message's "state" value.
            text = decode(prompt)
            state = text[text.find('"state"'):text.find('"question"')]
            choices.append({'index': i, 'logprobs': {'top_logprobs': [
                {f'token_id:{t}': 2.0 * state.count(w) for t, w in zip(ids, words)}]}})
        return 200, {'choices': choices}


TICKET = ('Customer was charged twice for order 1182. They want the duplicate refund today. '
          'The delivery itself arrived on time. Thanks for the help!')


def payload(**extra):
    return {'model': 'm', 'state': {'ticket': TICKET}, 'questions': {
        'route': {'type': 'choice', 'instructions': 'Which team?',
                  'criteria': {'billing': 'Payments', 'shipping': 'Deliveries', 'tech': 'Bugs'}},
        'urgent': {'type': 'noul', 'instructions': 'Is it urgent?'},
        'priority': {'type': 'score', 'instructions': 'How urgent?', 'criteria': ['low', 'high']}}, **extra}


@pytest.fixture
def scorer():
    executor = ThreadPoolExecutor(2)
    upstream = Upstream()
    yield adapter.Scorer(Tokenizer(), upstream.post, 'served', executor=executor, position_ordering=False,
                         mask_token_id=MASK), upstream
    executor.shutdown()


def run(scorer, body):
    return asyncio.run(scorer.evaluate(body))


# ---- spans --------------------------------------------------------------------------------------------------------

def texts(text, ranges):
    return [text[a:b] for a, b in ranges]


def test_sentences_english_japanese_lines_and_markers():
    assert texts(TICKET, explain.sentences(TICKET)) == [
        'Customer was charged twice for order 1182.', 'They want the duplicate refund today.',
        'The delivery itself arrived on time.', 'Thanks for the help!']
    ja = '注文が二重に請求されました。返金してください！よろしく'
    assert texts(ja, explain.sentences(ja)) == ['注文が二重に請求されました。', '返金してください！', 'よろしく']
    lines = 'step 1: open app\nstep 2: crash\n\n[... 12 lines omitted ...]\nstep 15: done'
    assert texts(lines, explain.sentences(lines)) == ['step 1: open app', 'step 2: crash', 'step 15: done']
    assert explain.sentences('  ...  ') == []


def test_words_and_phrases():
    text = "Don't refund order 1182, please."
    assert texts(text, explain.words(text)) == ["Don't", 'refund', 'order', '1182', 'please']
    ja = '注文が二重に請求された'
    assert texts(ja, explain.words(ja)) == ['注文', 'が', '二重', 'に', '請求', 'された']
    long = 'one two three four five six seven eight nine ten, eleven twelve.'
    assert texts(long, explain.phrases(long)) == [
        'one two three four five six seven eight', 'nine ten', 'eleven twelve.']


def test_json_leaves_paths_and_escaped_offsets():
    state = {'ticket': 'Say "hi"\nto\tme. ✓ 終わり。', 'n': 3, 'tags': ['a', ''],
             'messages': [{'role': 'user', 'content': 'Refund? Yes.'}], 'odd key': 'x', 'none': None}
    serialized, leaves = explain.locate_leaves(state)
    assert serialized == json.dumps(state, ensure_ascii=False)
    assert [explain.json_path(p) for p, _, _ in leaves] == [
        '$.ticket', '$.tags[0]', '$.messages[0].role', '$.messages[0].content', '$["odd key"]']
    for _, offset, text in leaves:
        escaped = explain.escaped_offsets(text)
        for a, b in explain.sentences(text) + explain.words(text):
            assert serialized[offset + escaped[a]:offset + escaped[b]] == json.dumps(text[a:b], ensure_ascii=False)[1:-1]
    _, spans = explain.leaf_spans('Plain string state. Two.', 'sentence')
    assert [(s['path'], s['text']) for s in spans] == [('$', 'Plain string state.'), ('$', 'Two.')]


def test_masking_keeps_token_count_and_hits_only_the_span():
    tokenizer = Tokenizer()
    state = {'ticket': 'Hello "world".\nSecond line ✓.'}
    question = {'instructions': 'q'}
    rendered = scoring._render(tokenizer, state, question, [{'code': 'A', 'label': 'x', 'meaning': 'X'}])
    ids = tokenizer.encode(rendered)
    occluder = explain.Occluder(tokenizer, rendered, state, ids, MASK)
    leaves, spans = explain.leaf_spans(state, 'sentence')
    assert [s['text'] for s in spans] == ['Hello "world".', 'Second line ✓.']
    for span in spans:
        masked = occluder.occluded(span)
        assert len(masked) == len(ids)
        changed = [i for i, (a, b) in enumerate(zip(ids, masked)) if a != b]
        assert all(masked[i] == MASK for i in changed)
        assert decode(ids[changed[0]:changed[-1] + 1]) == json.dumps(span['text'], ensure_ascii=False)[1:-1]
        deleted = occluder.occluded(span, 'delete')
        assert deleted == ids[:changed[0]] + ids[changed[-1] + 1:]
    with pytest.raises(RuntimeError, match='reproduce'):
        explain.Occluder(tokenizer, rendered, state, ids[:-1], MASK)


def test_fit_spans_merges_adjacent_spans_within_leaves():
    text = ' '.join(f'Sentence {i}.' for i in range(10))
    leaves, spans = explain.leaf_spans({'a': text, 'b': 'One. Two.'}, 'sentence')
    fitted, merged = explain.fit_spans(spans, 4, leaves)
    assert merged and len(fitted) == 4
    a = [s for s in fitted if s['path'] == '$.a']
    assert a[0]['start'] == 0 and a[-1]['end'] == len(text) and all(s['text'] == text[s['start']:s['end']] for s in a)
    assert explain.fit_spans(spans, 12, leaves) == (spans, False)
    many = {'a': text, **{f'k{i}': 'Short one.' for i in range(3)}}
    leaves2, spans2 = explain.leaf_spans(many, 'sentence')
    fitted, _ = explain.fit_spans(spans2, 5, leaves2)
    assert len(fitted) == 5 and {s['path'] for s in fitted} == {'$.a', '$.k0', '$.k1', '$.k2'}
    whole, merged = explain.fit_spans(spans, 1, leaves)
    assert merged and [s['path'] for s in whole] == ['$.a']


def test_parse_explain_validation():
    questions = {'q': {}}
    assert explain.parse_explain(False, questions) is None
    assert explain.parse_explain(True, questions) == {'granularity': 'sentence', 'top_k': 5, 'questions': None,
                                                      'occlusion': 'mask'}
    assert explain.parse_explain({'occlusion': 'delete'}, questions)['occlusion'] == 'delete'
    assert explain.parse_explain({'granularity': 'word', 'questions': ['q', 'q']}, questions)['questions'] == ['q']
    for bad, message in [('yes', 'true, false'), ({'granularity': 'char'}, 'granularity'),
                         ({'top_k': 0}, 'top_k'), ({'top_k': True}, 'top_k'), ({'questions': []}, 'questions'),
                         ({'questions': ['nope']}, 'unknown question'), ({'k': 1}, 'unknown field'),
                         ({'occlusion': 'blank'}, 'occlusion')]:
        with pytest.raises(ValueError, match=message):
            explain.parse_explain(bad, questions)


# ---- end to end through the adapter ---------------------------------------------------------------------------------

def without_timing(result):
    out = copy.deepcopy(result)
    for key in ('seconds', 'prepare_seconds', 'engine_seconds'):
        out['metrics'].pop(key)
    return out


def test_off_by_default_is_identical(scorer):
    scorer, upstream = scorer
    plain = run(scorer, payload())
    calls = copy.deepcopy(upstream.calls)
    upstream.calls.clear()
    off = run(scorer, payload(explain=False))
    assert json.dumps(without_timing(plain)) == json.dumps(without_timing(off))
    assert upstream.calls == calls and len(calls) == 2
    assert set(plain['metrics']) == {'seconds', 'prepare_seconds', 'engine_seconds', 'questions', 'upstream_calls'}
    assert all('explanation' not in a for a in plain['answers'].values())


def test_sentence_explanations_delta_math_and_shape(scorer):
    scorer, upstream = scorer
    plain = run(scorer, payload())
    upstream.calls.clear()
    result = run(scorer, payload(explain=True))
    # Answers are unchanged; only explanations, metrics and usage are added.
    for key, answer in plain['answers'].items():
        assert {k: v for k, v in result['answers'][key].items() if k != 'explanation'} == answer
    route = result['answers']['route']['explanation']
    assert route['decision'] == 'billing' and route['base_probability'] == pytest.approx(
        result['answers']['route']['confidence'])
    assert route['spans'][0]['text'] == 'They want the duplicate refund today.'
    assert route['spans'][0]['path'] == '$.ticket'
    assert TICKET[route['spans'][0]['start']:route['spans'][0]['end']] == route['spans'][0]['text']
    # Exact delta: full logits (billing, shipping, tech) = (2, 2, 0); with 'refund' masked (0, 2, 0).
    full, masked = math.exp(2) / (2 * math.exp(2) + 1), 1 / (math.exp(2) + 2)
    assert route['base_probability'] == pytest.approx(full)
    assert route['spans'][0]['delta'] == pytest.approx(full - masked, abs=1e-6)
    # Masking the delivery sentence makes billing more likely: counter-evidence.
    assert route['counter_evidence'][0]['text'] == 'The delivery itself arrived on time.'
    assert route['counter_evidence'][0]['delta'] < 0
    assert set(route) == {'method', 'granularity', 'decision', 'base_probability', 'spans', 'counter_evidence',
                          'truncated'} and route['method'] == 'mask_occlusion' and not route['truncated']
    urgent = result['answers']['urgent']['explanation']
    # 'refund' argues for 'false' here: the decision is no, and masking the refund sentence raises p_yes.
    assert urgent['decision'] == 'false' and urgent['spans'][0]['text'] == 'They want the duplicate refund today.'
    assert urgent['base_probability'] == pytest.approx(1 - result['answers']['urgent']['noul'])
    assert result['answers']['priority']['explanation']['notes'][0].startswith('Explanations cover choice')
    # 4 sentences x 2 explained questions, batched into one call per candidate set.
    assert result['metrics']['explain_forwards'] == 8 and result['metrics']['explain_upstream_calls'] == 2
    assert len(upstream.calls) == 4 and {len(c['prompt']) for c in upstream.calls[2:]} == {4}
    for call in upstream.calls[2:]:
        assert len({len(p) for p in call['prompt']}) == 1  # occlusion keeps every prompt's length
    assert result['usage']['input_tokens'] == plain['usage']['input_tokens'] + sum(
        len(p) - 1 for c in upstream.calls[2:] for p in c['prompt'])
    assert result['metrics']['explain_seconds'] >= 0


def test_each_question_gets_its_own_reads(scorer):
    """Questions explained together must each be scored on their own occluded prompts."""
    scorer, _ = scorer
    body = payload(explain=True)
    body['state'] = {'ticket': 'Account is blocked. Please refund me.'}
    together = run(scorer, body)['answers']
    for key in ('route', 'urgent'):
        alone = run(scorer, dict(body, explain={'questions': [key]}))['answers'][key]['explanation']
        assert together[key]['explanation'] == alone
    # Exact values: route (billing) depends only on 'refund'; urgent ('false', from 'refund' vs 'blocked').
    route = together['route']['explanation']
    assert [s['text'] for s in route['spans']] == ['Please refund me.'] and route['counter_evidence'] == []
    urgent = together['urgent']['explanation']
    assert urgent['decision'] == 'true' and urgent['base_probability'] == pytest.approx(0.5)
    assert [s['text'] for s in urgent['spans']] == ['Account is blocked.']
    assert urgent['spans'][0]['delta'] == pytest.approx(0.5 - 1 / (1 + math.exp(2)), abs=1e-6)
    assert [s['text'] for s in urgent['counter_evidence']] == ['Please refund me.']


def test_delete_occlusion_drops_span_tokens(scorer):
    scorer, upstream = scorer
    masked = run(scorer, payload(explain={'questions': ['route']}))
    upstream.calls.clear()
    deleted = run(scorer, payload(explain={'questions': ['route'], 'occlusion': 'delete'}))
    explanation = deleted['answers']['route']['explanation']
    assert explanation['method'] == 'span_deletion'
    full = len(upstream.calls[0]['prompt'][0])
    sentences = explain.sentences(TICKET)
    assert sorted(len(p) for p in upstream.calls[-1]['prompt']) == sorted(full - (b - a) for a, b in sentences)
    # The keyword fake cannot tell masking from deletion: same evidence, same deltas.
    ours = masked['answers']['route']['explanation']
    assert explanation['spans'] == ours['spans'] and explanation['counter_evidence'] == ours['counter_evidence']


def test_noul_sign_follows_the_decision(scorer):
    scorer, _ = scorer
    body = payload(explain={'questions': ['urgent']})
    body['state'] = {'ticket': 'Account is blocked. Nothing else.', 'note': 'Fine. Not blocked now.'}
    result = run(scorer, body)
    assert set(k for k, a in result['answers'].items() if 'explanation' in a) == {'urgent'}
    urgent = result['answers']['urgent']['explanation']
    assert urgent['decision'] == 'true' and urgent['base_probability'] == pytest.approx(result['answers']['urgent']['noul'])
    assert {s['text'] for s in urgent['spans']} == {'Account is blocked.', 'Not blocked now.'}
    assert {s['path'] for s in urgent['spans']} == {'$.ticket', '$.note'}
    assert urgent['counter_evidence'] == []


def test_word_granularity_two_stages(scorer):
    scorer, upstream = scorer
    result = run(scorer, payload(explain={'granularity': 'word', 'questions': ['route'], 'top_k': 2}))
    route = result['answers']['route']['explanation']
    assert route['granularity'] == 'word' and route['spans'][0]['text'] == 'refund'
    assert len(route['spans']) == 1  # only the keyword word changes the read
    # Stage one: 4 sentences; stage two: words of the one supportive sentence (6 words).
    assert result['metrics']['explain_forwards'] == 4 + 6
    assert route['counter_evidence'] == []


def test_cap_merges_spans_and_reports_truncation(scorer, monkeypatch):
    scorer, _ = scorer
    monkeypatch.setattr(explain, 'EXPLAIN_MAX_SPANS', 3)
    body = payload(explain={'questions': ['route']})
    body['state'] = {'ticket': ' '.join(f'Filler sentence {i}.' for i in range(9)) + ' Please refund me.'}
    result = run(scorer, body)
    route = result['answers']['route']['explanation']
    assert result['metrics']['explain_forwards'] == 3 and route['truncated']
    assert '10 sentence spans were grouped into 3' in route['notes'][0]
    assert route['spans'][0]['text'].endswith('Please refund me.')


def test_cap_is_shared_across_questions(scorer, monkeypatch):
    scorer, _ = scorer
    monkeypatch.setattr(explain, 'EXPLAIN_MAX_SPANS', 5)
    result = run(scorer, payload(explain=True))
    assert result['metrics']['explain_forwards'] <= 5
    assert result['answers']['urgent']['explanation']['truncated']


def test_elimination_explains_the_final_read(scorer):
    scorer, upstream = scorer
    body = payload(elimination=True, explain={'questions': ['route']})
    body['questions']['route']['criteria'] = {'billing': 'P', 'shipping': 'D', 'tech': 'B', 'access': 'L'}
    result = run(scorer, body)
    route = result['answers']['route']['explanation']
    assert route['notes'] == ['Explains the elimination read between the two finalists.']
    assert route['decision'] == result['answers']['route']['choice'] == 'billing'
    # The elimination read has two candidates (refund vs delivery logits): 4 occluded prompts in a 2-id call.
    explain_calls = upstream.calls[-1:]
    assert len(explain_calls[0]['logprob_token_ids']) == 2 and len(explain_calls[0]['prompt']) == 4
    assert route['spans'][0]['text'] == 'They want the duplicate refund today.'


def test_compacted_state_is_explained_as_scored(scorer, monkeypatch):
    scorer, _ = scorer
    monkeypatch.setattr(scoring, 'COMPACT_OVER', 600)
    monkeypatch.setattr(scoring, 'COMPACT_BUDGET', 400)
    filler = ' '.join(f'Weather report {i} says sunny skies.' for i in range(60))
    body = payload(explain={'questions': ['route']})
    body['state'] = {'ticket': filler + ' Which team? Payments refund needed for the team.'}
    result = run(scorer, body)
    route = result['answers']['route']['explanation']
    assert route['notes'] == ['Spans refer to the compacted state that was scored.']
    assert route['spans'][0]['text'] == 'Payments refund needed for the team.'
    assert result['metrics']['explain_forwards'] < 20  # spans of the compacted text, not of all 61 sentences


def test_http_rejects_invalid_explain(scorer):
    from aiohttp.test_utils import TestClient, TestServer
    scorer, _ = scorer

    async def go():
        app = adapter.make_app(scorer, lambda: {}, lambda: True)
        async with TestClient(TestServer(app)) as client:
            r = await client.post('/v1/systemone', data=json.dumps(payload(explain={'granularity': 'x'})),
                                  headers={'Content-Type': 'application/json'})
            return r.status, await r.json()
    status, body = asyncio.run(go())
    assert status == 400 and 'granularity' in body['error']['message']


def test_server_default_is_deletion(monkeypatch):
    import importlib
    monkeypatch.delenv('EXPLAIN_OCCLUSION', raising=False)
    module = importlib.reload(explain)
    assert module.EXPLAIN_OCCLUSION == 'delete'
    assert module.parse_explain(True, {'q': {}})['occlusion'] == 'delete'
