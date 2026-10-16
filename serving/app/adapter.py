"""SystemOne API in front of `vllm serve`: production prompts, vLLM's async engine batches across requests.

The adapter keeps everything server.py does before and after the engine call (validation, long-state compaction,
position ordering, wide option codes, prompt-limit 400s, the response schema and timing, CORS, the explorer), and
replaces the offline `LLM` + global lock with HTTP calls to an OpenAI-compatible vLLM server. Each request's
questions go to /v1/completions as token-id prompts ending in the mask token, max_tokens=1, with the candidate
logits returned at the mask (`logprob_token_ids`, or `logprobs`+`allowed_token_ids` above 128 candidates). vLLM's
scheduler then batches reads from all in-flight requests, so concurrent users no longer queue behind each other.
"""
import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from model_config import MODEL_NAME

TOKEN_PREFIX = 'token_id:'


def group_reads(reads):
    """Questions with the same candidate set share one /v1/completions call: {candidate_ids: [read index]}."""
    groups = {}
    for index, (_, candidate_ids) in enumerate(reads):
        groups.setdefault(tuple(candidate_ids), []).append(index)
    return groups


def completion_body(model, prompts, candidate_ids, cache_salt=None):
    from vllm_scoring import candidate_logprob_args
    args = candidate_logprob_args(list(candidate_ids))
    body = {'model': model, 'prompt': prompts, 'max_tokens': 1, 'temperature': 0.0,
            'logprobs': args.get('logprobs', len(candidate_ids)),
            'return_tokens_as_token_ids': True, **args}
    if cache_salt:
        body['cache_salt'] = cache_salt
    return body


def mask_logprobs(choice):
    """Token id -> value at the single scored mask position of one completion choice."""
    logprobs = choice.get('logprobs') or {}
    top = logprobs.get('top_logprobs') or []
    if len(top) != 1 or top[0] is None:
        raise RuntimeError('Expected exactly one scored mask position')
    out = {}
    for token, value in top[0].items():
        if not token.startswith(TOKEN_PREFIX):
            raise RuntimeError('Upstream did not return token ids')
        out[int(token[len(TOKEN_PREFIX):])] = value
    return out


class UpstreamError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


ELIMINATION_DEFAULT = os.environ.get('ELIMINATION', '0') == '1'


def elimination_payload(payload, answers):
    """Second-round request: each choice question with more than two options, restricted to its first-round top two,
    with the other options named as ruled out. Returns None when no question qualifies."""
    questions = {}
    for qid, question in (payload.get('questions') or {}).items():
        answer = answers.get(qid) or {}
        criteria = question.get('criteria')
        if question.get('type') != 'choice' or not isinstance(criteria, dict) or len(criteria) <= 2:
            continue
        probs = answer.get('probabilities') or {}
        ranked = sorted(criteria, key=lambda label: probs.get(label, 0.0), reverse=True)
        keep, gone = ranked[:2], ranked[2:]
        instructions = question.get('instructions')
        if isinstance(instructions, str):
            note = '; '.join(str(criteria[label]) for label in gone[:30])
            instructions = f'{instructions}\n(Options already ruled out: {note})'
        questions[qid] = {**question, 'instructions': instructions, 'criteria': {label: criteria[label] for label in keep}}
    if not questions:
        return None
    return {**{k: v for k, v in payload.items() if k not in ('questions', 'elimination')}, 'questions': questions}


def merge_elimination(result, refined):
    """Fold second-round reads into first-round answers: the top two split their first-round mass by the second read;
    every other option keeps its first-round probability."""
    for qid, second in refined['answers'].items():
        answer = result['answers'][qid]
        probs = dict(answer['probabilities'])
        p2 = second.get('probabilities') or {}
        mass = sum(probs.get(label, 0.0) for label in p2)
        for label, value in p2.items():
            probs[label] = value * mass
        answer['probabilities'] = probs
        top = max(probs, key=probs.get)
        answer['choice'], answer['confidence'] = top, probs[top]
    # The same decisions are returned, so only the extra prompt tokens are added to usage.
    usage, extra = result.get('usage'), refined.get('usage')
    if isinstance(usage, dict) and isinstance(extra, dict) and isinstance(extra.get('input_tokens'), int):
        usage['input_tokens'] = usage.get('input_tokens', 0) + extra['input_tokens']


class Scorer:
    """Request -> answers. `post(path, body)` is an async JSON client returning (status, json)."""

    def __init__(self, tokenizer, post, model, *, executor, position_ordering=True, batch_tokenize=True,
                 isolate_cache=False, mask_token_id=100):
        self.tokenizer, self.post, self.model, self.executor = tokenizer, post, model, executor
        self.position_ordering, self.batch_tokenize = position_ordering, batch_tokenize
        self.isolate_cache, self.mask_token_id = isolate_cache, mask_token_id

    def _prepare(self, payload, trace=None):
        from vllm_scoring import build_reads
        return build_reads(self.tokenizer, payload, self.mask_token_id, self.batch_tokenize,
                           self.position_ordering, trace)

    async def _group(self, candidate_ids, prompts, salt):
        status, data = await self.post('/v1/completions',
                                       completion_body(self.model, prompts, candidate_ids, salt))
        if status != 200:
            error = data.get('error') if isinstance(data, dict) else None
            message = (error.get('message') if isinstance(error, dict) else error) or str(data)[:500]
            raise UpstreamError(status, message)
        choices = sorted(data.get('choices') or [], key=lambda c: c['index'])
        if len(choices) != len(prompts):
            raise RuntimeError('Incomplete scoring batch')
        return [mask_logprobs(choice) for choice in choices]

    async def evaluate(self, payload):
        """Score a request. With elimination (request field `"elimination": true`, or ELIMINATION=1 as the server
        default), choice questions with more than two options get a second read over their top two, with the other
        options named as ruled out; the top two then share their first-round probability mass by the second read."""
        flag = payload.get('elimination', ELIMINATION_DEFAULT) if isinstance(payload, dict) else False
        if not isinstance(flag, bool):
            raise ValueError('elimination must be true or false')
        config = None
        if isinstance(payload, dict) and 'explain' in payload:
            from explain import parse_explain
            config = parse_explain(payload['explain'], payload.get('questions'))
        reads = {} if config else None
        result = await self._evaluate_once(payload, reads)
        if flag:
            second = elimination_payload(payload, result['answers'])
            if second is not None:
                refined_reads = {} if config else None
                refined = await self._evaluate_once(second, refined_reads)
                merge_elimination(result, refined)
                if config:
                    # Explain the read that decided between the two finalists.
                    for key, read in refined_reads.items():
                        reads[key] = dict(read, round='elimination', first=reads[key])
        if config:
            await self._explain(payload, result, config, reads)
        return result

    async def _forward(self, reads):
        """Logprobs at the mask for [(prompt_ids, candidate_ids)], one upstream call per candidate set."""
        # A per-request salt keeps one request's prefix blocks invisible to every other request.
        salt = uuid.uuid4().hex if self.isolate_cache else None
        groups = group_reads(reads)
        results = await asyncio.gather(*[
            self._group(candidate_ids, [reads[i][0] for i in indices], salt)
            for candidate_ids, indices in groups.items()])
        logprobs = [None] * len(reads)
        for indices, values in zip(groups.values(), results):
            for i, value in zip(indices, values):
                logprobs[i] = value
        return logprobs, len(groups)

    async def _explain(self, payload, result, config, reads):
        """Occlusion explanations for the chosen answers (request field `"explain"`); see explain.py."""
        import explain
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        answers = result['answers']
        keys = config['questions'] or list(payload['questions'])
        jobs = []
        for key in keys:
            answer = answers[key]
            if answer['type'] == 'score':
                answer['explanation'] = explain.skipped(config)
                continue
            jobs.append(explain.Job(key, answer, reads[key], payload['state'], config))
        limits = explain.budgets(explain.EXPLAIN_MAX_SPANS, len(jobs))
        word = config['granularity'] == 'word'
        for job, limit in zip(jobs, limits):
            job.limit = limit
        prepare = (lambda job: job.first_stage(self.tokenizer, self.mask_token_id))
        await asyncio.gather(*[loop.run_in_executor(self.executor, prepare, job) for job in jobs])
        forwards = tokens = calls = 0
        for stage in (1, 2) if word else (1,):
            if stage == 2:
                await asyncio.gather(*[loop.run_in_executor(self.executor, job.second_stage) for job in jobs])
            batch = [(prompt, job.read['candidate_ids']) for job in jobs for prompt in job.prompts]
            if not batch:
                continue
            logprobs, n_calls = await self._forward(batch)
            forwards, calls = forwards + len(batch), calls + n_calls
            tokens += sum(len(prompt) - 1 for prompt, _ in batch)
            at = 0
            for job in jobs:
                count = len(job.prompts)
                job.record(logprobs[at:at + count])
                at += count
        for job in jobs:
            answers[job.key]['explanation'] = job.explanation()
        usage = result.get('usage')
        if isinstance(usage, dict) and isinstance(usage.get('input_tokens'), int):
            usage['input_tokens'] += tokens
        result['metrics'].update(explain_seconds=time.perf_counter() - started, explain_forwards=forwards,
                                 explain_upstream_calls=calls)

    async def _evaluate_once(self, payload, trace=None):
        """Score one request. A `trace` dict receives, per question, the read behind its answer (scored state,
        rendered prompt, token ids, candidate ids and logprobs) for explanations."""
        from vllm_scoring import answer_from_logprobs, response
        started = time.perf_counter()
        rendered = [] if trace is not None else None
        prepared, reads = await asyncio.get_running_loop().run_in_executor(
            self.executor, self._prepare, payload, rendered)
        prepared_at = time.perf_counter()
        logprobs, calls = await self._forward(reads)
        inferred_at = time.perf_counter()
        answers = {item[0]: answer_from_logprobs(payload, item, candidate_ids, values)
                   for item, (_, candidate_ids), values in zip(prepared, reads, logprobs)}
        if trace is not None:
            for item, (ids, candidate_ids), values, (state, prompt) in zip(prepared, reads, logprobs, rendered):
                trace[item[0]] = {'labels': item[2], 'ids': ids, 'candidate_ids': candidate_ids,
                                  'logprobs': values, 'state': state, 'prompt': prompt, 'round': 'first'}
        return response(prepared, answers, started, prepared_at, inferred_at, self.position_ordering,
                        upstream_calls=calls)


def make_app(scorer, health_info, upstream_ready, cors_origin='*'):
    from aiohttp import web

    def headers():
        if not cors_origin:
            return {}
        return {'Access-Control-Allow-Origin': cors_origin,
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Max-Age': '600'}

    def send(status, value):
        return web.Response(status=status, body=json.dumps(value, allow_nan=False).encode(),
                            content_type='application/json', headers=headers())

    async def options(request):
        return web.Response(status=204, headers=headers())

    async def explorer(request):
        return web.Response(body=Path(__file__).with_name('explorer.html').read_bytes(),
                            content_type='text/html', charset='utf-8',
                            headers={'Cache-Control': 'no-store'})

    async def health(request):
        ready = await upstream_ready()
        return send(200 if ready else 503, {**health_info(), 'ready': ready})

    async def systemone(request):
        try:
            if not 0 < (request.content_length or 0) <= 16000000:
                raise ValueError('Invalid request size')
            body = json.loads(await request.read())
            return send(200, await scorer.evaluate(body))
        except (ValueError, KeyError, TypeError) as exc:
            return send(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
        except UpstreamError as exc:
            if 400 <= exc.status < 500:
                return send(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
            return send(500, {'error': str(exc)})
        except Exception as exc:
            return send(500, {'error': str(exc)})

    async def not_found(request):
        return send(404, {'error': 'Only /v1/systemone is supported'
                          if request.method == 'POST' else 'Not found'})

    app = web.Application(client_max_size=16000000)
    app.router.add_route('OPTIONS', '/{tail:.*}', options)
    app.router.add_get('/', explorer)
    app.router.add_get('/explorer', explorer)
    app.router.add_get('/health', health)
    app.router.add_post('/v1/systemone', systemone)
    app.router.add_route('*', '/{tail:.*}', not_found)
    return app


def main():
    import aiohttp
    from aiohttp import web
    from transformers import AutoTokenizer
    import scoring
    from scoring import POSITION_ORDERING, candidate_codes

    env = os.environ
    for name in ('POSITION_ORDERING', 'BATCH_TOKENIZE', 'ISOLATE_REQUEST_CACHE'):
        if env.get(name, '1' if name != 'ISOLATE_REQUEST_CACHE' else '0') not in ('0', '1'):
            raise ValueError(f'{name} must be 0 or 1')
    position_ordering = env.get('POSITION_ORDERING', '1') == '1'
    attention = env.get('ATTENTION', 'noncausal')
    if attention not in ('causal', 'noncausal'):
        raise ValueError('ATTENTION must be causal or noncausal')
    noncausal = attention == 'noncausal'
    max_batched = int(env.get('MAX_BATCHED_TOKENS', '65536' if noncausal else '8192'))
    if noncausal:
        # No chunked prefill for bidirectional prompts: reject what one scheduler step cannot hold.
        scoring.set_prompt_limit(max_batched - 2)
    scoring.set_prefix_caching(not noncausal)
    isolate_cache = env.get('ISOLATE_REQUEST_CACHE', '0') == '1'
    upstream = env.get('UPSTREAM', 'http://127.0.0.1:8200').rstrip('/')
    model = env.get('SERVED_MODEL_NAME', MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(env.get('CHECKPOINT_DIR', '/models/checkpoint'),
                                              trust_remote_code=True, local_files_only=True)
    executor = ThreadPoolExecutor(int(env.get('PREP_WORKERS', '4')), thread_name_prefix='prep')
    timeout = aiohttp.ClientTimeout(total=float(env.get('UPSTREAM_TIMEOUT', '600')))
    state = {}

    async def session():
        if 'session' not in state:
            state['session'] = aiohttp.ClientSession(
                timeout=timeout, connector=aiohttp.TCPConnector(limit=0))
        return state['session']

    async def post(path, body):
        async with (await session()).post(upstream + path, json=body) as r:
            try:
                return r.status, await r.json(content_type=None)
            except ValueError:
                return r.status, {'error': (await r.text())[:500]}

    async def upstream_ready():
        try:
            async with (await session()).get(upstream + '/health',
                                             timeout=aiohttp.ClientTimeout(total=4)) as r:
                return r.status == 200
        except Exception:
            return False

    def health_info():
        return {'model': MODEL_NAME, 'backend': 'vllm', 'engine': 'vllm_serve', 'modes': ['systemone'],
                'quantization': env.get('QUANTIZATION', 'none'), 'candidate_only_head': False,
                'cache_policy': 'isolated_request' if isolate_cache else 'shared',
                'position_ordering': POSITION_ORDERING if position_ordering else 'off',
                'attention': attention,
                'limits': {'max_questions': 512, 'max_choices': len(candidate_codes(tokenizer)),
                           'max_prompt_tokens': scoring.prompt_limit()}}

    scorer = Scorer(tokenizer, post, model, executor=executor, position_ordering=position_ordering,
                    batch_tokenize=env.get('BATCH_TOKENIZE', '1') == '1', isolate_cache=isolate_cache)
    port = int(env.get('PORT', '8000'))
    print(f'SystemOne adapter on port {port}, upstream {upstream}', flush=True)
    web.run_app(make_app(scorer, health_info, upstream_ready, env.get('CORS_ORIGIN', '*')),
                host='0.0.0.0', port=port, print=None, access_log=None)


if __name__ == '__main__':
    main()
