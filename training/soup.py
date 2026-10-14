"""Uniform weight average of fine-tuned decoder tensors from several runs (same base)."""
import argparse
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser(usage='soup.py OUT IN [IN ...]', description=__doc__)
ap.add_argument('out', metavar='OUT', help='output .safetensors file')
ap.add_argument('paths', metavar='IN', nargs='+', help='input .safetensors files with identical keys (list one twice to weight it 2x)')
args = ap.parse_args()
out, paths = args.out, args.paths
acc = None
for p in paths:
    t = load_file(p)
    acc = {k: v.float() for k, v in t.items()} if acc is None else {k: acc[k] + t[k].float() for k in acc}
save_file({k: (v / len(paths)).to(torch.bfloat16).contiguous() for k, v in acc.items()}, out)
print('averaged', len(paths), 'runs,', len(acc), 'tensors')
