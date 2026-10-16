"""Batched single-mask scoring; no autoregressive answer rollout.

Each input ends with the checkpoint's mask token. vLLM returns the requested
candidate logprobs at that slot. Its single sampled token is discarded and is
never fed back into the model. All questions retain the native prompt format.
"""

import math
import time
from usage import decision_usage
from model_config import MODEL_NAME

from scoring import POSITION_ORDERING, candidate_codes, prepare_questions, pack_answer


def candidate_logprob_args(candidate_ids):
    """vLLM's indexed-logprob path supports up to 128 ids. With larger
    choice sets, masked top-k covers every candidate without omission."""
    return ({'logprob_token_ids': candidate_ids}
            if len(candidate_ids) <= 128 else
            {'logprobs': len(candidate_ids), 'allowed_token_ids': candidate_ids})


def build_reads(tokenizer, payload, mask_token_id=100, batch_tokenize=False,
                position_ordering=True, trace=None):
    """Validated, compacted, ordered prompts: (prepared, [(prompt_ids, candidate_ids)])."""
    prepared = prepare_questions(tokenizer, payload, batch_tokenize=batch_tokenize,
                                 position_ordering=position_ordering, trace=trace)
    codes = candidate_codes(tokenizer)
    reads = [(ids + [mask_token_id], [token for _, token in codes[:len(labels)]])
             for _, _, labels, _, ids in prepared]
    return prepared, reads


def answer_from_logprobs(payload, item, candidate_ids, logprobs):
    """Candidate softmax for one read; logprobs maps token id -> logprob (or logit)."""
    key, _, labels, _, _ = item
    if logprobs is None or any(token not in logprobs for token in candidate_ids):
        raise RuntimeError('Missing candidate scores')
    values = [logprobs[token] for token in candidate_ids]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('Non-finite candidate logprobs')
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    denominator = sum(weights)
    return pack_answer(payload['questions'][key], labels, [w / denominator for w in weights])


def response(prepared, answers, started, prepared_at, inferred_at, position_ordering, **metrics):
    return {
        'model': MODEL_NAME, 'answers': answers,
        'usage': decision_usage(sum(len(item[4]) for item in prepared),
                                len(answers), sampled_tokens=len(answers)),
        'metrics': {'seconds': time.perf_counter() - started,
                    'prepare_seconds': prepared_at - started,
                    'engine_seconds': inferred_at - prepared_at,
                    'questions': len(answers), **metrics},
        'scoring': {'method': 'diffusion_masked_token_candidate_softmax',
                    'position_ordering': POSITION_ORDERING if position_ordering else 'off',
                    'backend': 'vllm', 'confidence': 'maximum_candidate_probability',
                    'calibrated': False},
    }


def evaluate(engine, tokenizer, payload, mask_token_id=100, prime_shared_prefix=False,
             batch_tokenize=False, position_ordering=True):
    from vllm import SamplingParams

    started = time.perf_counter()
    prepared, reads = build_reads(tokenizer, payload, mask_token_id, batch_tokenize,
                                  position_ordering)
    prompts = [{'prompt_token_ids': ids} for ids, _ in reads]
    params = [SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                             **candidate_logprob_args(candidate_ids))
              for _, candidate_ids in reads]
    prepared_at = time.perf_counter()
    # A single engine call lets the scheduler batch question suffixes and reuse
    # already-computed prefix blocks. No prompt_logprobs: it defeats cache reads.
    if prime_shared_prefix and len(prompts) > 1:
        # Make common prefix blocks available before scheduling the fan-out.
        # Merely enabling APC need not avoid repeated prefill in the first batch.
        outputs = engine.generate(prompts[:1], params[:1], use_tqdm=False)
        outputs += engine.generate(prompts[1:], params[1:], use_tqdm=False)
    else:
        outputs = engine.generate(prompts, params, use_tqdm=False)
    inferred_at = time.perf_counter()
    if len(outputs) != len(prepared):
        raise RuntimeError('Incomplete scoring batch')
    answers = {}
    for item, (_, candidate_ids), output in zip(prepared, reads, outputs):
        if (len(output.outputs) != 1 or not output.outputs[0].logprobs
                or len(output.outputs[0].logprobs) != 1):
            raise RuntimeError('Expected exactly one scored mask position')
        logprobs = output.outputs[0].logprobs[0]
        answers[item[0]] = answer_from_logprobs(
            payload, item, candidate_ids,
            None if logprobs is None else {token: lp.logprob for token, lp in logprobs.items()})
    return response(prepared, answers, started, prepared_at, inferred_at, position_ordering)
