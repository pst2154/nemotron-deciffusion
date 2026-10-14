"""Decision Index engine: Nemotron Deciffusion in-process (no HTTP), same prompt builder and readout as serving.

Use: python -m decision_index run --engine di_engine:DeciffusionEngine --option checkpoint=... --option attention=noncausal
"""
import os
import sys
from pathlib import Path
import time

import torch
from decision_index.engines.base import Engine, Unsupported

APP = os.environ.get('DECIFFUSION_APP', str(Path(__file__).resolve().parents[1] / 'serving' / 'app'))


class DeciffusionEngine(Engine):
    name = 'deciffusion-native'
    latency = 'In-process wall time per request (all questions), excluding model loading.'

    def __init__(self, checkpoint, attention='noncausal', **options):
        super().__init__(**options)
        sys.path.insert(0, APP)
        from transformers import AutoModel, AutoTokenizer
        from scoring import candidate_codes, pack_answer, prepare_questions
        self.prepare_questions, self.pack_answer = prepare_questions, pack_answer
        self.tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.codes = candidate_codes(self.tok)
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                               attn_implementation='sdpa').cuda().eval()
        self.causal = attention == 'causal'
        for layer in self.model.encoder.layers:
            layer.self_attn.diffusion_lm = not self.causal
        self.provenance = {'kind': 'full fine-tune', 'checkpoint': checkpoint, 'attention': attention}

    @torch.inference_mode()
    def _probs(self, ids, n):
        x = torch.tensor([ids + [self.model.mask_token_id]], device='cuda')
        h = self.model.encoder(input_ids=x, use_cache=False, use_causal_mask=self.causal, is_training=False).last_hidden_state
        logits = self.model.diffusion_head(h[:, -1]).float()[0, [c for _, c in self.codes[:n]]]
        return logits.softmax(-1).tolist()

    def __call__(self, state, questions):
        payload = {'model': 'deciffusion', 'state': state, 'questions': questions}
        try:
            prepared = self.prepare_questions(self.tok, payload, position_ordering=True)
        except ValueError as e:  # over the option/prompt limits or rejected by request validation
            raise Unsupported(str(e))
        started = time.perf_counter()
        answers = {key: self.pack_answer(questions[key], labels, self._probs(ids, len(labels)))
                   for key, _, labels, _, ids in prepared}
        response = {'answers': answers}
        return response, dict(response, seconds=time.perf_counter() - started)

    def synchronize(self):
        torch.cuda.synchronize()
