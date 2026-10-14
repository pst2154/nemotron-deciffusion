"""The vllm-serve adapter must reproduce the offline path's prompts and answers, batch across requests,
and isolate failures. CPU only: a fake OpenAI completions upstream returns deterministic logits."""
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import adapter  # noqa: E402
import scoring  # noqa: E402
from vllm_scoring import evaluate  # noqa: E402


class Tokenizer:
    def encode(self, text, **kwargs):
        if text.isdigit():
            return [1000 + int(text)]
        return [ord(c) % 997 for c in text] if len(text) > 1 else [ord(text)]

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, prompts, **kwargs):
        return {'input_ids': [self.encode(p) for p in prompts]}


def logit(prompt, token):
    """Deterministic, prompt-dependent candidate logit."""
    return ((sum(prompt) * 31 + token * 17) % 1000) / 100.0


class Upstream:
    """Fake /v1/completions with vLLM's response shape (token_id:N keys)."""

    def __init__(self, fail_if=None, delay=0.0):
        self.calls, self.fail_if, self.delay = [], fail_if, delay
        self.in_flight = self.max_in_flight = 0

    async def post(self, path, body):
        assert path == '/v1/completions'
        self.calls.append(body)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if self.fail_if and self.fail_if(body):
                return 400, {'error': {'message': 'prompt too long for one scheduler step'}}
            ids = body.get('logprob_token_ids') or body['allowed_token_ids']
            assert body['max_tokens'] == 1 and body['return_tokens_as_token_ids']
            assert body['logprobs'] == len(ids)
            choices = [{'index': i, 'logprobs': {'top_logprobs': [
                {f'token_id:{t}': logit(p, t) for t in ids}]}}
                for i, p in enumerate(body['prompt'])]
            return 200, {'choices': choices[::-1]}  # order must come from 'index'
        finally:
            self.in_flight -= 1


def payload(tag='x', n=3, wide=False):
    questions = {f'q{i}': {'type': 'noul', 'instructions': f'{tag} condition {i}?'} for i in range(n)}
    questions['pick'] = {'type': 'choice', 'instructions': f'{tag} pick',
                         'criteria': {'red': 'warm', 'blue': 'cold', 'green': 'grass'}}
    questions['rate'] = {'type': 'score', 'instructions': 'rate', 'criteria': ['low', 'mid', 'high']}
    if wide:
        questions['wide'] = {'type': 'choice', 'instructions': 'many',
                             'criteria': {f'o{i}': f'option {i}' for i in range(200)}}
    return {'model': 'jev-latest', 'state': {'text': f'state {tag}'}, 'questions': questions}


def offline(body, tokenizer):
    def generate(prompts, params, **kwargs):
        out = []
        for prompt, p in zip(prompts, params):
            ids = p.logprob_token_ids if hasattr(p, 'logprob_token_ids') else p.allowed_token_ids
            out.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
                t: SimpleNamespace(logprob=logit(prompt['prompt_token_ids'], t)) for t in ids}])]))
        return out
    with patch.dict(sys.modules, {'vllm': SimpleNamespace(
            SamplingParams=lambda **kw: SimpleNamespace(**kw))}):
        return evaluate(SimpleNamespace(generate=generate), tokenizer, body,
                        batch_tokenize=True, position_ordering=True)


@pytest.fixture
def make_scorer():
    executor = ThreadPoolExecutor(4)
    tokenizer = Tokenizer()

    def make(upstream, **kwargs):
        return adapter.Scorer(tokenizer, upstream.post, 'served', executor=executor, **kwargs), tokenizer
    yield make
    executor.shutdown()


def test_answers_match_offline_path_including_wide_choices(make_scorer):
    upstream = Upstream()
    scorer, tokenizer = make_scorer(upstream)
    body = payload(wide=True)
    result = asyncio.run(scorer.evaluate(body))
    reference = offline(body, tokenizer)
    assert result['answers'] == reference['answers']
    assert list(result['answers']) == list(body['questions'])
    assert result['usage'] == reference['usage']
    assert result['scoring'] == reference['scoring']
    assert set(result['metrics']) >= {'seconds', 'prepare_seconds', 'engine_seconds', 'questions'}
    # noul (2 codes), choice/score (3 codes), wide (200 codes) -> three grouped calls.
    assert result['metrics']['upstream_calls'] == len(upstream.calls) == 3
    wide = [c for c in upstream.calls if 'allowed_token_ids' in c]
    assert len(wide) == 1 and wide[0]['logprobs'] == 200 and 'logprob_token_ids' not in wide[0]
    for call in upstream.calls:
        assert all(prompt[-1] == 100 for prompt in call['prompt'])
        assert call['model'] == 'served' and 'cache_salt' not in call


def test_concurrent_requests_are_in_flight_together_and_split_back(make_scorer):
    upstream = Upstream(delay=0.05)
    scorer, tokenizer = make_scorer(upstream)
    bodies = [payload(tag=f'r{i}', n=1 + i % 4) for i in range(16)]

    async def run():
        return await asyncio.gather(*[scorer.evaluate(b) for b in bodies])
    results = asyncio.run(run())
    assert upstream.max_in_flight > 2  # requests are not serialized
    for body, result in zip(bodies, results):
        assert result['answers'] == offline(body, tokenizer)['answers']


def test_one_bad_request_does_not_fail_the_others(make_scorer):
    upstream = Upstream(fail_if=lambda body: any(len(p) > 1500 for p in body['prompt']))
    scorer, _ = make_scorer(upstream)
    good = [payload(tag=f'g{i}') for i in range(4)]
    bad = payload(tag='bad')
    bad['state'] = {'text': 'long ' * 500}

    async def run():
        return await asyncio.gather(*[scorer.evaluate(b) for b in good + [bad]],
                                    return_exceptions=True)
    results = asyncio.run(run())
    assert all(isinstance(r, dict) for r in results[:4])
    assert isinstance(results[4], adapter.UpstreamError) and results[4].status == 400


def test_isolated_cache_salts_each_request(make_scorer):
    upstream = Upstream()
    scorer, _ = make_scorer(upstream, isolate_cache=True)
    asyncio.run(scorer.evaluate(payload('a')))
    asyncio.run(scorer.evaluate(payload('b')))
    salts = [c['cache_salt'] for c in upstream.calls]
    assert len(set(salts[:2])) == 1 and salts[0] != salts[-1]


def test_missing_token_ids_and_extra_positions_are_errors():
    with pytest.raises(RuntimeError, match='token ids'):
        adapter.mask_logprobs({'logprobs': {'top_logprobs': [{'A': 1.0}]}})
    with pytest.raises(RuntimeError, match='one scored mask'):
        adapter.mask_logprobs({'logprobs': {'top_logprobs': [{}, {}]}})


def http(make_scorer, upstream, requests, ready=True):
    from aiohttp.test_utils import TestClient, TestServer
    scorer, _ = make_scorer(upstream)

    async def upstream_ready():
        return ready

    async def run():
        app = adapter.make_app(scorer, lambda: {'model': 'm'}, upstream_ready)
        async with TestClient(TestServer(app)) as client:
            out = []
            for method, path, body in requests:
                r = await client.request(method, path, data=body,
                                         headers={'Content-Type': 'application/json'})
                out.append((r.status, r.headers.get('Access-Control-Allow-Origin'),
                            await r.text()))
            return out
    return asyncio.run(run())


def test_http_status_codes_and_cors(make_scorer, monkeypatch):
    monkeypatch.setattr(scoring, '_prompt_limit', 1500)
    long_body = payload('long')
    # The question alone is over the scoring limit; no state reduction can fit it: 400.
    long_body['questions']['q0']['instructions'] = 'w ' * 2000
    prose_body = payload('prose')
    # One long prose leaf is compacted to fit the scoring limit instead of being rejected: 200.
    prose_body['state'] = {'text': 'w ' * 2000}
    out = http(make_scorer, Upstream(), [
        ('POST', '/v1/systemone', json.dumps(payload())),
        ('POST', '/v1/systemone', '{not json'),
        ('POST', '/v1/systemone', json.dumps({'model': 'm', 'state': 's'})),
        ('POST', '/v1/systemone', json.dumps(long_body)),
        ('GET', '/health', None),
        ('OPTIONS', '/v1/systemone', None),
        ('POST', '/v1/other', '{}'),
        ('POST', '/v1/systemone', json.dumps(prose_body)),
    ])
    assert [status for status, _, _ in out] == [200, 400, 400, 400, 200, 204, 404, 200]
    assert all(origin == '*' for _, origin, _ in out)
    assert 'scoring limit' in json.loads(out[3][2])['error']['message']
    assert json.loads(out[4][2]) == {'model': 'm', 'ready': True}


def test_http_upstream_errors(make_scorer):
    bad = http(make_scorer, Upstream(fail_if=lambda body: True),
               [('POST', '/v1/systemone', json.dumps(payload()))])
    assert bad[0][0] == 400 and 'scheduler step' in bad[0][2]
    out = http(make_scorer, Upstream(), [('GET', '/health', None)], ready=False)
    assert out[0][0] == 503
