#!/usr/bin/env python3
"""Merge a LoRA adapter from train_lora.py into the base weights (W += scale * B @ A, computed in fp32).

Writes, without loading the model code:
  --output DIR    standalone checkpoint: every non-weight file of --checkpoint copied, plus model.safetensors
                  (the layout training/merge_full.py produces). Serve it with CHECKPOINT_DIR=DIR SKIP_DOWNLOAD=1.
  --trained FILE  optional: decoder tensors (encoder.layers.*, encoder.norm.*) in bf16, the same format as
                  train_pack.py's trained.safetensors, so the result can be souped with training/soup.py.

  python3 finetune/merge_lora.py --checkpoint $CKPT --adapter runs/lora1/best \\
      --output merged/lora1 --trained runs/lora1/best/trained.safetensors

Needs ~2x the checkpoint size in CPU RAM (about 35 GB for 8B in bf16).
"""
import argparse
import json
import math
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='the base the adapter was trained on')
    p.add_argument('--adapter', required=True, help='PEFT adapter dir (adapter_config.json, adapter_model.safetensors)')
    p.add_argument('--output', default='', help='merged checkpoint dir (must not exist)')
    p.add_argument('--trained', default='', help='also write decoder tensors here')
    a = p.parse_args()
    if not a.output and not a.trained:
        p.error('give --output and/or --trained')
    import torch
    from safetensors.torch import load_file, save_file

    cfg = json.loads((Path(a.adapter) / 'adapter_config.json').read_text())
    if cfg.get('use_dora') or cfg.get('rank_pattern') or cfg.get('alpha_pattern'):
        raise SystemExit('DoRA / rank_pattern / alpha_pattern adapters are not supported by this merger')
    r, alpha = cfg['r'], cfg['lora_alpha']
    scale = alpha / math.sqrt(r) if cfg.get('use_rslora') else alpha / r
    adapter = load_file(str(Path(a.adapter) / 'adapter_model.safetensors'))

    weights = {}
    for f in sorted(Path(a.checkpoint).glob('*.safetensors')):
        weights.update(load_file(str(f)))
    merged = 0
    for key, A in adapter.items():
        if not key.endswith('.lora_A.weight'):
            continue
        module = key[:-len('.lora_A.weight')]
        B = adapter[module + '.lora_B.weight']
        name = module.removeprefix('base_model.model.') + '.weight'
        if name not in weights:
            raise SystemExit(f'{name} not in checkpoint (adapter trained on a different base?)')
        W = weights[name]
        delta = (B.float() @ A.float()) * scale
        assert delta.shape == W.shape, (name, delta.shape, W.shape)
        weights[name] = (W.float() + delta).to(W.dtype)
        merged += 1
    print(f'merged {merged} LoRA modules (r={r}, alpha={alpha}, scale={scale:g})')

    if a.trained:
        keep = {k: v.to(torch.bfloat16).contiguous() for k, v in weights.items()
                if k.startswith('encoder.layers.') or k.startswith('encoder.norm')}
        Path(a.trained).parent.mkdir(parents=True, exist_ok=True)
        save_file(keep, a.trained)
        print(f'wrote {len(keep)} decoder tensors to {a.trained}')
    if a.output:
        out = Path(a.output)
        assert not out.exists(), f'{out} exists'
        shutil.copytree(a.checkpoint, out, ignore=shutil.ignore_patterns('*.safetensors', '*.safetensors.index.json'))
        save_file({k: v.contiguous() for k, v in weights.items()}, str(out / 'model.safetensors'), metadata={'format': 'pt'})
        (out / 'merge-receipt.json').write_text(json.dumps(dict(base=str(a.checkpoint), adapter=str(a.adapter),
                                                                lora_modules=merged, r=r, alpha=alpha, scale=scale)))
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
