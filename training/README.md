# Training

The trainer behind Nemotron Deciffusion 8B v2, plus the two tools that turn its output into a servable model.
To fine-tune the release on your own decisions, start with [`../finetune/`](../finetune/); it wraps these files.

| file | purpose |
|---|---|
| [`train_pack.py`](train_pack.py) | Full fine-tune of every decoder layer with FSDP2: fp32 master weights and AdamW state sharded across GPUs, bf16 compute, embeddings and output head frozen. Rows are packed into fixed 8,192-token sequences with a per-document FlexAttention block mask, so packed rows never attend to each other. Loss: cross-entropy at the mask slot against `teacher_probabilities` when present, otherwise one-hot `reference.target`. |
| [`soup.py`](soup.py) | Uniform weight average of `trained.safetensors` files from runs on the same base (list a file twice to weight it 2:1). |
| [`merge_full.py`](merge_full.py) | Writes trained decoder tensors onto a base checkpoint to make a standalone, servable model directory. |

## Run

```bash
torchrun --standalone --nproc-per-node=4 training/train_pack.py \
    --checkpoint /models/Nemotron-Deciffusion-8B-v2 --app serving/app --data data/mix --output runs/r1 \
    --attention noncausal --lr 7e-7 --warmup 0.15 --epochs 1 --batch 256 --micro 8 \
    --pack-len 8192 --max-tokens 8192 --validate-every 300
python3 training/merge_full.py --checkpoint /models/Nemotron-Deciffusion-8B-v2 \
    --trained runs/r1/best/trained.safetensors --output merged/r1
```

- `--data` is a directory with `train.jsonl` and `eval.jsonl`. `eval.jsonl` is the checkpoint-selection split and
  must have exactly 580 rows; [`../finetune/mix.py`](../finetune/mix.py) builds both. Row format:
  [`../finetune/schema.md`](../finetune/schema.md).
- `--app serving/app` makes training use the serving prompt builder, so what you train is what you serve.
- `--attention noncausal` trains the bidirectional read used in serving. Keep it on.
- `--init runs/r0/best/trained.safetensors` continues from an earlier run on the same base.
- Output: `best/` and `last/trained.safetensors` (decoder tensors only), `training.jsonl`, `validation.jsonl`,
  `selected.json`. Selection is by validation accuracy, then cross-entropy.
- Requirements: PyTorch with FlexAttention, `transformers` able to load the checkpoint's remote code,
  `safetensors`. Rows over `--max-tokens` are skipped, so validate first with
  [`../finetune/validate.py`](../finetune/validate.py).
