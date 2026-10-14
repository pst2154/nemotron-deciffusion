#!/usr/bin/env python3
"""LoRA fine-tuning of Nemotron Deciffusion 8B on the serving decision graph (single node, 1..8 GPUs, bf16).

Checked on one H100: on a 100-row eval split the step-0 validation (86/100, CE 0.419) equals
finetune/eval_heldout.py --checkpoint (native, noncausal) on the same rows, and an 8-row overfit run drives the
loss from 0.42 to 0.0002 in 6 steps. Check the step-0 match on your own eval split before a long run.

Forward (identical to train_pack.py / native_eval.py / serving with ATTENTION=noncausal):
  ids = serving prompt (serving/app/scoring.py prepare_questions, position_ordering=True) + [mask token]
  one bidirectional pass of the decoder (attention diffusion_lm=True, no causal mask)
  logits = diffusion_head(hidden at the mask position), restricted to the option-code token ids
  loss = soft-target cross-entropy (teacher_probabilities when present, else one-hot reference)
Option augmentation as train_pack.py: a --canonical-fraction of rows keep the serving order, the rest are
shuffled and re-coded by position.

Only LoRA adapters on the decoder's q/k/v/o/gate/up/down projections are trained; embeddings and the diffusion
head stay frozen. Examples run one per forward (no padding: padding would change a bidirectional pass), with
gradient accumulation to --batch rows per optimizer step, gradient checkpointing, and DDP across GPUs.

  # 1 GPU
  python3 finetune/train_lora.py --checkpoint $CKPT --data data/mix1 --output runs/lora1
  # 8 GPUs
  torchrun --standalone --nproc-per-node=8 finetune/train_lora.py --checkpoint $CKPT --data data/mix1 --output runs/lora1

Data dir: train.jsonl (required), eval.jsonl (optional; any size). Outputs: best/ and last/ (PEFT adapters),
training.jsonl, validation.jsonl, selected.json. Merge with finetune/merge_lora.py.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', ''), help='HF model dir [$CHECKPOINT]')
    p.add_argument('--app', default=str(REPO / 'serving' / 'app'), help='serving/app (prompt builder)')
    p.add_argument('--data', required=True, help='dir with train.jsonl and optional eval.jsonl')
    p.add_argument('--output', required=True)
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--alpha', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--targets', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--batch', type=int, default=16, help='rows per optimizer step (global)')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--warmup', type=float, default=0.05)
    p.add_argument('--canonical-fraction', type=float, default=0.5)
    p.add_argument('--max-tokens', type=int, default=8192, help='drop rows whose prompt + mask exceeds this')
    p.add_argument('--validate-every', type=int, default=100)
    p.add_argument('--limit-steps', type=int, default=0)
    p.add_argument('--attention', choices=['causal', 'noncausal'], default='noncausal')
    p.add_argument('--no-grad-checkpointing', action='store_true')
    p.add_argument('--seed', type=int, default=101)
    a = p.parse_args()
    if not a.checkpoint:
        p.error('--checkpoint (or $CHECKPOINT) is required')

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    distributed = world > 1
    if distributed:
        dist.init_process_group('nccl')
    device = torch.device('cuda', local)
    torch.manual_seed(a.seed)

    def all_reduce(t):
        if distributed:
            dist.all_reduce(t)
        return t

    def barrier():
        if distributed:
            dist.barrier()

    out = Path(a.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    barrier()

    sys.path.insert(0, a.app)
    import scoring
    from scoring import candidate_codes, prepare_questions

    tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True)
    codes = candidate_codes(tok)
    data = Path(a.data)
    rows = [json.loads(l) for l in (data / 'train.jsonl').read_text().splitlines() if l.strip()]
    val_rows = ([json.loads(l) for l in (data / 'eval.jsonl').read_text().splitlines() if l.strip()]
                if (data / 'eval.jsonl').exists() else [])

    def prepare(row, aug_seed=None):
        """Same as training/train_pack.py prepare(): serving prompt, optional shuffled/re-coded options."""
        payload = dict(row['input'], model='x')
        try:
            _, kind, labels, descs, ids = prepare_questions(tok, payload, position_ordering=True)[0]
        except ValueError:
            return None
        if aug_seed is not None:
            order = list(range(len(labels))); random.Random(aug_seed).shuffle(order)
            labels = [labels[i] for i in order]; descs = [descs[i] for i in order]
            q = next(iter(payload['questions'].values()))
            options = [dict(code=codes[i][0], label=l, meaning=d) for i, (l, d) in enumerate(zip(labels, descs))]
            ids = tok.encode(scoring._render(tok, payload['state'], q, options), add_special_tokens=False)
        t = row['reference']['target']; label = str(t).lower() if kind == 'noul' else str(t)
        if label not in labels or len(ids) + 1 > a.max_tokens:
            return None
        if 'teacher_probabilities' in row:
            target = [float(row['teacher_probabilities'].get(l, 0.0)) for l in labels]
            s = sum(target)
            if s <= 0:
                return None
            target = [x / s for x in target]
        else:
            target = [float(l == label) for l in labels]
        return dict(ids=ids, target=target, gold=labels.index(label), cand=[c for _, c in codes[:len(labels)]])

    validation = [x for x in (prepare(r) for r in val_rows) if x is not None]
    filler = validation[0] if validation else next(x for x in map(prepare, rows) if x is not None)
    if rank == 0 and len(validation) < len(val_rows):
        print(f'eval: {len(val_rows) - len(validation)} rows skipped (invalid or too long)', flush=True)

    model = AutoModel.from_pretrained(a.checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
                                      attn_implementation='sdpa').to(device)
    mask_id = model.mask_token_id
    causal = a.attention == 'causal'
    for layer in model.encoder.layers:
        layer.self_attn.diffusion_lm = not causal  # bidirectional: attention runs with no mask, is_causal=False
    if not a.no_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        except ValueError as e:  # remote code without checkpointing support: long rows may run out of memory
            print(f'gradient checkpointing unavailable: {e}', flush=True)
    names = set(a.targets.split(','))
    targets = [n for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and n.startswith('encoder.layers.') and n.rsplit('.', 1)[-1] in names]
    assert targets, 'no target modules found'
    peft = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                            target_modules=targets, bias='none'))
    trainable = [q for q in peft.parameters() if q.requires_grad]
    assert all('lora_' in n for n, q in peft.named_parameters() if q.requires_grad)

    class Scorer(torch.nn.Module):
        def __init__(self, peft_model):
            super().__init__()
            self.peft = peft_model

        def forward(self, ids):
            # No torch.autocast: the base weights are already bf16 and PEFT casts to/from the fp32 adapters
            # itself. Autocast also recasts the remote code's fp32 rotary/scaling math to bf16, which moved
            # the noncausal readout (smoke test: 86/100 -> 79/100 on the same eval rows).
            base = self.peft.get_base_model()
            h = base.encoder(input_ids=ids, use_cache=False, use_causal_mask=causal,
                             is_training=False).last_hidden_state
            return base.diffusion_head(h[:, -1]).float()[0]

    scorer = Scorer(peft)
    net = DDP(scorer, device_ids=[local], find_unused_parameters=False) if distributed else scorer
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)

    def run(ex, module):
        x = torch.tensor([ex['ids'] + [mask_id]], device=device)
        return module(x)[ex['cand']]

    def log(name, record):
        if rank == 0:
            with (out / name).open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(name, json.dumps(record), flush=True)

    def validate(step):
        if not validation:
            return None
        scorer.eval()
        stats = torch.zeros(3, device=device, dtype=torch.float64)
        with torch.no_grad():
            for ex in validation[rank::world]:
                lg = run(ex, scorer)
                ce = -lg.log_softmax(-1)[ex['gold']].item()
                stats += torch.tensor([int(lg.argmax().item() == ex['gold']), ce, 1], device=device, dtype=torch.float64)
        all_reduce(stats)
        rec = dict(step=step, correct=int(stats[0]), total=int(stats[2]), cross_entropy=(stats[1] / stats[2]).item())
        log('validation.jsonl', rec)
        scorer.train()
        return rec

    def save(tag):
        if rank == 0:
            peft.save_pretrained(out / tag)
        barrier()

    steps_per_epoch = len(rows) // a.batch
    total = a.limit_steps or steps_per_epoch * a.epochs
    assert a.batch % world == 0, '--batch must be divisible by the number of GPUs'
    accum = a.batch // world
    warm = max(1, int(a.warmup * total))
    log('config.json.log', dict(vars(a), world=world, rows=len(rows), eval=len(validation), total_steps=total,
                                trainable=sum(q.numel() for q in trainable), lora_modules=len(targets)))
    scorer.train()
    best = validate(0)
    best_key = (-best['correct'], best['cross_entropy']) if best else None
    best_step = 0
    save('best')
    step = 0
    for epoch in range(a.epochs):
        order = list(range(len(rows))); random.Random(a.seed + epoch).shuffle(order)
        for update in range(steps_per_epoch):
            if step >= total:
                break
            t0 = time.monotonic(); losses = []; skipped = 0
            for j in range(accum):
                idx = order[update * a.batch + j * world + rank]
                aug = idx + a.seed * 1000003 + epoch
                ex = prepare(rows[idx], None if random.Random(aug).random() < a.canonical_fraction else aug)
                weight = 1.0
                if ex is None:  # keep DDP in lockstep: run a zero-weight example instead
                    ex, weight, skipped = filler, 0.0, skipped + 1
                sync = j == accum - 1
                ctx = net.no_sync() if distributed and not sync else nullcontext()
                with ctx:
                    lg = run(ex, net)
                    loss = -(torch.tensor(ex['target'], device=device) * lg.log_softmax(-1)).sum() * weight
                    (loss / accum).backward()
                losses.append(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            factor = ((step + 1) / warm if step + 1 <= warm
                      else .1 + .9 * (1 + math.cos(math.pi * (step + 1 - warm) / max(1, total - warm))) / 2)
            for g in optimizer.param_groups:
                g['lr'] = a.lr * factor
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            lm = all_reduce(torch.stack(losses).mean()) / world
            if step % 10 == 0 or step <= 10:
                log('training.jsonl', dict(step=step, epoch=epoch + 1, loss=lm.item(), grad_norm=float(norm),
                                           seconds=time.monotonic() - t0, lr=optimizer.param_groups[0]['lr'],
                                           skipped_rows_this_rank=skipped,
                                           peak_gib=torch.cuda.max_memory_allocated() / 2**30))
            if step % a.validate_every == 0 or step == total:
                rec = validate(step)
                if rec is not None:
                    key = (-rec['correct'], rec['cross_entropy'])
                    if key < best_key:
                        best_key, best_step = key, step
                        save('best')
    save('last')
    if best is None:  # no eval split: best = last
        save('best'); best_step = step
    if rank == 0:
        (out / 'selected.json').write_text(json.dumps(dict(step=best_step, validation_correct=-best_key[0] if best_key else None)))
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
