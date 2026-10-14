"""Write a standalone checkpoint: base files plus fully fine-tuned decoder tensors."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
for k in ('checkpoint', 'trained', 'output'):
    p.add_argument('--' + k, required=True)
a = p.parse_args()
out = Path(a.output); assert not out.exists()
shutil.copytree(a.checkpoint, out, ignore=shutil.ignore_patterns('*.safetensors', '*.safetensors.index.json'))
trained = load_file(a.trained)
weights = {}
for f in sorted(Path(a.checkpoint).glob('*.safetensors')):
    weights.update(load_file(str(f)))
missing = set(trained) - set(weights)
assert not missing, list(missing)[:5]
for k, v in trained.items():
    assert weights[k].shape == v.shape
    weights[k] = v.to(weights[k].dtype)
save_file(weights, str(out / 'model.safetensors'), metadata={'format': 'pt'})
(out / 'merge-receipt.json').write_text(json.dumps(dict(replaced=len(trained), total=len(weights))))
print('replaced', len(trained), 'of', len(weights))
