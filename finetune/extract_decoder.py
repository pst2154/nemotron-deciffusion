#!/usr/bin/env python3
"""Extract a checkpoint's decoder tensors in train_pack.py's trained.safetensors format (for souping).

training/soup.py averages trained.safetensors files that share the same keys. To soup a fine-tuned run with
the checkpoint it started from, extract the starting checkpoint's decoder tensors first:

  python3 finetune/extract_decoder.py --checkpoint $CKPT --output runs/start.safetensors
  python3 training/soup.py runs/soup.safetensors runs/start.safetensors runs/fft1/best/trained.safetensors
  python3 training/merge_full.py --checkpoint $CKPT --trained runs/soup.safetensors --output merged/soup

Keys: encoder.layers.* and encoder.norm.* (add --with-head if the run used train_pack.py --train-head).
"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--with-head', action='store_true', help='also include diffusion_head.* tensors')
    a = p.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    prefixes = ('encoder.layers.', 'encoder.norm') + (('diffusion_head',) if a.with_head else ())
    keep = {}
    for f in sorted(Path(a.checkpoint).glob('*.safetensors')):
        keep.update({k: v.to(torch.bfloat16).contiguous() for k, v in load_file(str(f)).items() if k.startswith(prefixes)})
    assert keep, 'no decoder tensors found'
    save_file(keep, a.output)
    print(f'wrote {len(keep)} tensors to {a.output}')


if __name__ == '__main__':
    main()
