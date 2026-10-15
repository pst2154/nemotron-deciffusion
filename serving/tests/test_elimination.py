"""Opt-in two-round elimination: second read over the top two choice options, merged into the first-round answer."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import adapter  # noqa: E402

PAYLOAD = {'state': {'ticket': 'x'}, 'questions': {
    'route': {'type': 'choice', 'instructions': 'Which team?', 'criteria': {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}},
    'pair': {'type': 'choice', 'instructions': 'Which?', 'criteria': {'x': 'X', 'y': 'Y'}},
    'urgent': {'type': 'noul', 'instructions': 'Urgent?'}}}
FIRST = {'answers': {
    'route': {'type': 'choice', 'probabilities': {'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1}, 'confidence': 0.4, 'choice': 'a'},
    'pair': {'type': 'choice', 'probabilities': {'x': 0.6, 'y': 0.4}, 'confidence': 0.6, 'choice': 'x'},
    'urgent': {'type': 'noul', 'noul': 0.7}}, 'usage': {'input_tokens': 100, 'output_tokens': 3, 'decision_positions': 3}}


def test_second_round_payload():
    second = adapter.elimination_payload(PAYLOAD, FIRST['answers'])
    assert set(second['questions']) == {'route'}  # two-option and non-choice questions are not re-read
    route = second['questions']['route']
    assert route['criteria'] == {'a': 'A', 'b': 'B'}
    assert route['instructions'].endswith('(Options already ruled out: C; D)')
    assert 'elimination' not in second and second['state'] == PAYLOAD['state']


def test_merge_keeps_mass_and_untouched_options():
    result = {'answers': {k: dict(v, probabilities=dict(v.get('probabilities', {}))) if 'probabilities' in v else dict(v)
                          for k, v in FIRST['answers'].items()}, 'usage': dict(FIRST['usage'])}
    refined = {'answers': {'route': {'type': 'choice', 'probabilities': {'a': 0.25, 'b': 0.75}}},
               'usage': {'input_tokens': 40, 'output_tokens': 1, 'decision_positions': 1}}
    adapter.merge_elimination(result, refined)
    route = result['answers']['route']
    assert route['probabilities'] == pytest.approx({'a': 0.175, 'b': 0.525, 'c': 0.2, 'd': 0.1})
    assert route['choice'] == 'b' and route['confidence'] == pytest.approx(0.525)
    assert sum(route['probabilities'].values()) == pytest.approx(1.0)
    assert result['answers']['pair']['choice'] == 'x'
    assert result['usage'] == {'input_tokens': 140, 'output_tokens': 3, 'decision_positions': 3}


@pytest.mark.parametrize('flag,calls', [(False, 1), (True, 2), (None, 1)])
def test_evaluate_runs_second_round_only_when_enabled(flag, calls):
    scorer = adapter.Scorer.__new__(adapter.Scorer)
    seen = []

    async def once(payload):
        seen.append(payload)
        if len(seen) == 1:
            return {'answers': {k: dict(v) for k, v in FIRST['answers'].items()}, 'usage': dict(FIRST['usage'])}
        return {'answers': {'route': {'type': 'choice', 'probabilities': {'a': 0.9, 'b': 0.1}}}, 'usage': {'input_tokens': 1}}

    scorer._evaluate_once = once
    payload = dict(PAYLOAD) if flag is None else dict(PAYLOAD, elimination=flag)
    out = asyncio.run(scorer.evaluate(payload))
    assert len(seen) == calls
    if calls == 2:
        assert out['answers']['route']['choice'] == 'a'


def test_rejects_non_boolean_flag():
    scorer = adapter.Scorer.__new__(adapter.Scorer)
    with pytest.raises(ValueError, match='elimination'):
        asyncio.run(scorer.evaluate(dict(PAYLOAD, elimination='yes')))
