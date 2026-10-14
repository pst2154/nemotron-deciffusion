import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import scoring  # noqa: E402


class Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ' '.join(m['content'] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {'input_ids': [1] * len(texts.split())}
        return {'input_ids': [[1] * len(t.split()) for t in texts]}


def test_unschedulable_prompt_is_rejected(monkeypatch):
    monkeypatch.setattr(scoring, 'candidate_codes', lambda tok: tuple((c, i) for i, c in enumerate('ABCD')))
    monkeypatch.setattr(scoring, 'COMPACT_OVER', 0)
    monkeypatch.setattr(scoring, '_prompt_limit', scoring.MAX_PROMPT_TOKENS)
    scoring.set_prompt_limit(50)
    payload = {'model': 'm', 'state': {'facts': ['word'] * 200},
               'questions': {'q': {'type': 'noul', 'instructions': 'Is it?'}}}
    with pytest.raises(ValueError, match='scoring limit'):
        scoring.prepare_questions(Tok(), payload)


def _long_payload():
    story = ' '.join(f'Filler sentence number {i} about nothing.' for i in range(600))
    story += ' Mary went to the kitchen. John took the apple to the garden.'
    return {'model': 'm', 'state': {'story': story},
            'questions': {'mary': {'type': 'noul', 'instructions': 'Is Mary in the kitchen?'},
                          'apple': {'type': 'noul', 'instructions': 'Is the apple in the garden?'}}}


def _prefix_len(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


@pytest.mark.parametrize('scope', ['question', 'request'])
def test_compaction_scope_controls_shared_prefix(monkeypatch, scope):
    monkeypatch.setattr(scoring, 'candidate_codes', lambda tok: tuple((c, i) for i, c in enumerate('ABCD')))
    monkeypatch.setattr(scoring, 'COMPACT_OVER', 300)
    monkeypatch.setattr(scoring, 'COMPACT_BUDGET', 60)
    monkeypatch.setattr(scoring, '_prompt_limit', scoring.MAX_PROMPT_TOKENS)
    monkeypatch.setattr(scoring, '_scope', scope)
    tok = Tok()
    rendered = []
    monkeypatch.setattr(scoring, '_render', lambda t, state, q, o: (state != {} and rendered.append(state)) or
                        ' '.join([str(state), q['instructions']]))
    out = scoring.prepare_questions(tok, _long_payload())
    assert len(out) == 2
    compacted = []  # compacted states in order (a fit check renders the same state as the final prompt)
    for state in rendered[2:]:  # first two renders are the uncompacted prompts
        if not compacted or state is not compacted[-1]:
            compacted.append(state)
    assert len(compacted) == (1 if scope == 'request' else 2)
    compacted = compacted * (2 if scope == 'request' else 1)
    if scope == 'request':
        assert compacted[0] == compacted[1]
        assert 'Mary went to the kitchen.' in compacted[0]['story']
        assert 'John took the apple' in compacted[0]['story']
    else:
        assert 'Mary went to the kitchen.' in compacted[0]['story']
        assert 'John took the apple' in compacted[1]['story']


def test_auto_scope_follows_prefix_caching(monkeypatch):
    monkeypatch.setattr(scoring, 'COMPACT_SCOPE', 'auto')
    scoring.set_prefix_caching(True)
    assert scoring._scope == 'request'
    scoring.set_prefix_caching(False)
    assert scoring._scope == 'question'


@pytest.mark.parametrize('scope', ['question', 'request'])
@pytest.mark.parametrize('limit', [8190, 16382])
def test_compacted_prompt_fits_small_context(monkeypatch, scope, limit):
    """MAX_MODEL_LEN 8192/16384: the state budget leaves room for question, options and template."""
    monkeypatch.setattr(scoring, 'candidate_codes', lambda tok: tuple((c, i) for i, c in enumerate('ABCD')))
    monkeypatch.setattr(scoring, 'COMPACT_OVER', 8192)
    monkeypatch.setattr(scoring, 'COMPACT_BUDGET', 8192)
    monkeypatch.setattr(scoring, 'COMPACT_MARGIN', 64)
    monkeypatch.setattr(scoring, '_prompt_limit', scoring.MAX_PROMPT_TOKENS)
    monkeypatch.setattr(scoring, '_scope', scope)
    scoring.set_prompt_limit(limit)
    story = ' '.join(f'Filler sentence number {i} about nothing in particular here.' for i in range(3000))
    story += ' Mary went to the kitchen.'
    long_question = 'Is Mary in the kitchen? ' + ' '.join(['Consider every clause of the evidence.'] * 120)
    payload = {'model': 'm', 'state': {'story': story},
               'questions': {'q': {'type': 'choice', 'instructions': long_question,
                                   'criteria': {f'opt{i}': ' '.join(['option meaning words'] * 40)
                                                for i in range(4)}}}}
    out = scoring.prepare_questions(Tok(), payload)
    ids = out[0][-1]
    assert len(ids) <= limit, (len(ids), limit)


def test_small_context_compacts_below_gate(monkeypatch):
    """A prompt under COMPACT_OVER but over the scoring limit is compacted instead of rejected."""
    monkeypatch.setattr(scoring, 'candidate_codes', lambda tok: tuple((c, i) for i, c in enumerate('ABCD')))
    monkeypatch.setattr(scoring, 'COMPACT_OVER', 8192)
    monkeypatch.setattr(scoring, 'COMPACT_BUDGET', 8192)
    monkeypatch.setattr(scoring, '_prompt_limit', scoring.MAX_PROMPT_TOKENS)
    monkeypatch.setattr(scoring, '_scope', 'question')
    scoring.set_prompt_limit(2000)
    story = ' '.join(f'Filler sentence number {i} about nothing.' for i in range(600))
    payload = {'model': 'm', 'state': {'story': story + ' Mary went to the kitchen.'},
               'questions': {'q': {'type': 'noul', 'instructions': 'Is Mary in the kitchen?'}}}
    out = scoring.prepare_questions(Tok(), payload)
    assert len(out[0][-1]) <= 2000


class EscapeTok(Tok):
    """Every JSON escape costs a token, as in real tokenizers: raw text undercounts the rendered prompt."""

    def encode(self, text, add_special_tokens=False):
        return [1] * (len(text.split()) + text.count('\\'))

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {'input_ids': self.encode(texts)}
        return {'input_ids': [self.encode(t) for t in texts]}


def _small_context(monkeypatch, scope, limit):
    monkeypatch.setattr(scoring, 'candidate_codes', lambda tok: tuple((c, i) for i, c in enumerate('ABCD')))
    monkeypatch.setattr(scoring, 'COMPACT_OVER', 8192)
    monkeypatch.setattr(scoring, 'COMPACT_BUDGET', 8192)
    monkeypatch.setattr(scoring, 'COMPACT_MARGIN', 64)
    monkeypatch.setattr(scoring, '_prompt_limit', scoring.MAX_PROMPT_TOKENS)
    monkeypatch.setattr(scoring, '_scope', scope)
    scoring.set_prompt_limit(limit)


@pytest.mark.parametrize('scope', ['question', 'request'])
def test_budget_counts_json_escaping(monkeypatch, scope):
    """Quoted, multi-line traces grow when JSON-escaped into the prompt; the rendered prompt must still fit."""
    _small_context(monkeypatch, scope, 2000)
    steps = '\n\n'.join(f'step {i}: tool said "ok" and wrote "done"\n"result": "fine"' for i in range(800))
    payload = {'model': 'm', 'state': {'trajectory': steps},
               'questions': {'q': {'type': 'noul', 'instructions': 'Did the agent finish?'}}}
    out = scoring.prepare_questions(EscapeTok(), payload)
    assert len(out[0][-1]) <= 2000


@pytest.mark.parametrize('scope', ['question', 'request'])
def test_uncompactable_state_is_truncated_to_fit(monkeypatch, scope):
    """Many short items are never compacted; head+tail truncation still makes the prompt fit."""
    _small_context(monkeypatch, scope, 500)
    payload = {'model': 'm', 'state': {'items': ['w w w w'] * 500},
               'questions': {'q': {'type': 'noul', 'instructions': 'Is it?'}}}
    out = scoring.prepare_questions(EscapeTok(), payload)
    assert len(out[0][-1]) <= 500


def test_question_alone_over_limit_is_rejected(monkeypatch):
    _small_context(monkeypatch, 'question', 100)
    payload = {'model': 'm', 'state': {'text': 'short'},
               'questions': {'q': {'type': 'noul', 'instructions': 'word ' * 300}}}
    with pytest.raises(ValueError, match='scoring limit'):
        scoring.prepare_questions(Tok(), payload)
