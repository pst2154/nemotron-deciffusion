# Fine-tuning Nemotron Deciffusion 8B on your own decisions

This folder shows how to adapt the released checkpoint to a new decision task: how to build the training data,
how to train (LoRA or full fine-tuning), and how to ship the result without losing what the model already does
well. Everything here uses the same prompt builder as serving (`serving/app/scoring.py`), so what you train is
what you serve.

| file | purpose |
|---|---|
| [`schema.md`](schema.md) | Exact row schema, label rules per question type, and how rows map to API requests and responses |
| [`examples/`](examples/) | Small synthetic rows: `choice`, `noul`, `score`, `long_context`, `agent_trace` (head+tail rendered) |
| `convert_labeled.py` | Labelled CSV/JSONL → rows, with option shuffling, re-keying, paraphrases, label balancing and a group-disjoint held-out split |
| `teacher_label.py` | Soft labels, or new blind-verified items, from any OpenAI-compatible endpoint (e.g. `vllm serve`) |
| `render_long.py` | Pre-render long states with serving's compactor so they fit 8192 tokens and match serving |
| `validate.py` | Checks rows exactly as the trainer sees them (schema, label, length, leakage) and reports label balance |
| `mix.py` | Task + replay mix, deterministic shuffle, batch-multiple trim, the 580-row `eval.jsonl` the FFT trainer needs |
| `train_fft.sh` | Full fine-tune: wrapper around `training/train_pack.py` with continuation defaults |
| `train_lora.py`, `merge_lora.py` | LoRA trainer on the same forward graph (smoke-tested on 1 GPU), and the merge back into a full checkpoint |
| `extract_decoder.py` | Starting checkpoint → `trained.safetensors` format, so you can soup with it |
| `eval_heldout.py` | Accuracy, log loss, Brier, ECE and noul AUC on any held-out rows, via a running server or a local checkpoint |

All Python scripts print their full usage with `--help`. None contain paths or credentials; model and data
locations come from arguments or environment variables (`CHECKPOINT`, `TEACHER_BASE_URL`, …).

## What you are fine-tuning

The model answers typed questions over a JSON `state` in **one forward pass**:

- `choice`: pick one of 2 to 501 options (`criteria: {key: description}`), returns a probability per key.
- `noul`: yes/no, returns p(true). Gold labels are `"true"` / `"false"`.
- `score`: ordered levels (`criteria: [level 0, level 1, ...]`), returns a probability per level and the
  expected level.

The prompt is a fixed system message plus a JSON user turn `{state, question, options}` where each option has a
single-token code. One mask token is appended, attention is **bidirectional** over the whole prompt, and the
answer distribution is the diffusion head's softmax over the option codes at the mask position. Training uses
exactly this graph with a soft-target cross-entropy loss (`teacher_probabilities` when present, otherwise
one-hot on `reference.target`). The method is described in the [release post](../docs/BLOG_v2.md).

## LoRA or full fine-tuning?

| | LoRA (`train_lora.py`) | Full fine-tune (`train_fft.sh` → `training/train_pack.py`) |
|---|---|---|
| Use when | One narrow new task, a few hundred to a few tens of thousands of rows, modest shift from what the model already does | Large shifts (new input shapes such as agent traces, new long-context behavior, many tasks), 50k+ rows |
| Typical recipe | The lightweight option for a narrow task | Full passes over the mix, then a weight average with the starting checkpoint (step 6) |
| Hardware | 1 to 8 GPUs with 80 GB (bf16 base + adapters, gradient checkpointing) | 1 node, 8× H100/H200 80 GB+ (FSDP2, fp32 master weights and AdamW state sharded) |
| Speed | One example per forward (no padding, since padding would change a bidirectional pass) | ~5–12 s per step at 256 rows packed into 8k-token sequences: ~200 steps (50k rows) in 20–40 min |
| Status | Mirrors `train_pack.py`; smoke-tested on 1 GPU (step-0 readout equals native eval) | The trainer used for the released model |

Either way, keep attention bidirectional when training (`--attention noncausal`, the default in both entry
points here) and serve with `ATTENTION=noncausal` (the container default). Flipping the base to bidirectional
without training scores 154/231; trained bidirectional is the single biggest win we measured.

## Step by step

```
define decisions → build rows → validate → mix replay → train → soup + merge → evaluate → serve
```

### 1. Define the decisions

Write down each decision as the API call you will make in production: what goes in `state`, the question
text, the type, and the options. Rules that paid off:

- **One decision per row.** A production request with several questions becomes several rows with the same
  state.
- **Choose the type by what you will do with the output.** A threshold on a probability → `noul`. One of a set
  of actions → `choice`. A graded quantity you will average or rank by → `score`.
- **Options carry meaning.** The model reads the descriptions; the keys are just what comes back. Use
  descriptive text even if your keys are opaque ids. Add an explicit "none of these" option if it can happen.
- **The state is evidence only.** Never put the label, the outcome, or anything computed from them (`outcome`,
  `resolved_as`, `tests_passed_after_merge`, …) in the state. Leakage trains a shortcut that fails in
  production. `validate.py` and `teacher_label.py` flag answer-like keys; extend the list with `--leak-fields`.
- **Fix the held-out split now.** Split by the underlying item (document, customer, repository, trajectory),
  never by row, so the same state cannot appear on both sides. Keep it frozen for the whole project.

### 2. Build the rows

Format: [`schema.md`](schema.md). Examples: [`examples/`](examples/). Three common sources:

**a) You already have labels (CSV/JSONL, human labels, rule labels).**

```bash
python3 finetune/convert_labeled.py --input tickets.csv \
    --state-cols subject,body,plan --question "Which support queue should handle this ticket?" \
    --options-col queues --label-col resolved_queue --group-col customer_id \
    --instructions-file paraphrases.txt --variants 2 --balance \
    --eval-fraction 0.1 --domain support-routing --output data/task
# -> data/task/train.jsonl, data/task/heldout.jsonl, data/task/convert_report.json
```

It shuffles choice options and re-keys them per row (semantic slugs, letters, opaque ids, `option_i`), swaps
in paraphrased instructions (`--instructions-file`: one per line, or JSON `{original: [paraphrases]}`),
balances labels within each type (noul exactly 50/50; choice/score classes capped at 1.5× the median), caps
rows whose gold is the longest option (`--max-longest-share 0.3`; the model will otherwise learn "longest is
right" — we downsampled that to ~27%), and splits by `--group-col` (or by state hash).

**b) From logs and outcomes.** Take the production request (`state` + question) and attach what actually
happened later as `reference.target`: the ticket's final queue, whether the agent run passed hidden tests,
whether the claim was paid. These are hard labels; they are fine. Make sure the outcome is not visible in the
state you would have had at decision time. For agent traces, render the trace as serving will (step 2d).

**c) From an LLM teacher.** Serve a teacher (any strong open model served with vLLM) and either
soft-label your rows or author new ones:

```bash
export TEACHER_BASE_URL=http://localhost:8000/v1      # TEACHER_API_KEY / TEACHER_MODEL if needed
# soft labels for existing rows: 5 blind samples each; target = 0.5 one-hot + 0.5 teacher
python3 finetune/teacher_label.py --samples 5 label --input data/task/train.jsonl \
    --output data/task/train.soft.jsonl --drop-below 0.1
# new items: author → blind verify → no-context check → n-gram filter against your eval set
python3 finetune/teacher_label.py --workers 64 generate --task-file my_task.md --type choice,noul \
    --examples finetune/examples/choice.jsonl --domains "insurance claims,travel expenses" \
    --count 5000 --verify-samples 3 --soft --against data/task/heldout.jsonl --output data/gen/items.jsonl
```

The generation recipe is the one that worked for us: the intended answer is assigned
up front (balanced labels and positions), the item is kept only if a second blind pass reproduces it, items the
teacher can answer **without** the state are dropped, and items sharing a 13-gram with your eval set are
dropped. Expect to keep 20–50% of authored items. Store `teacher_probabilities` (vote shares across several
samples, or a verifier's confidence): the soft-target loss is what trains calibration. Read a sample of the
kept items yourself before training on thousands of them.

**d) Long inputs and agent traces.** Training drops rows over 8192 tokens, and serving compacts prompts over
8192 tokens (BM25 sentence selection for prose; head+tail for trace-like keys such as `trajectory`, `log`,
`messages`, `steps`, `history`). Train on the same rendering:

```bash
python3 finetune/render_long.py --checkpoint $CKPT --input data/task/train.jsonl --output data/task/train.r.jsonl
```

`examples/agent_trace.jsonl` shows the result for traces: the opening steps, an
`[... N items omitted ...]` marker, then the most recent steps. BM25 ranks prose by overlap with the question
and option text only, so a decisive sentence that shares no words with the question can be cut; prefer specific
questions, and keep states under 8k tokens when the whole document matters.

### 3. Validate

```bash
export CHECKPOINT=/models/Nemotron-Deciffusion-8B     # any local copy of the checkpoint you start from
python3 finetune/validate.py data/task/train.r.jsonl --against data/task/heldout.jsonl \
    --output data/task/train.valid.jsonl
python3 finetune/validate.py data/task/heldout.jsonl          # held-out rows must be valid too
```

Errors (row dropped): missing fields, more than one question, label not in the labels, bad soft labels,
prompt + mask over 8192 tokens (measured with the real tokenizer and serving prompt builder), duplicates.

### 4. Mix in replay

Training only on the new task makes the model forget general skills. Mix **40–60% replay**: general decision
rows in the same format, for example earlier tasks of your own or openly licensed typed-decision data converted with
`convert_labeled.py`. Mix several kinds (`choice`, `noul`, `score`; short and long states) and keep part of the
replay aside as `replay/heldout.jsonl`, so you can check general skills after training.

```bash
python3 finetune/mix.py --task data/task/train.valid.jsonl data/gen/items.jsonl \
    --replay replay/general.jsonl:3 --replay replay/noul.jsonl:1 \
    --replay-fraction 0.5 --task-eval data/task/heldout.jsonl --replay-eval replay/heldout.jsonl \
    --batch 256 --output data/mix1
python3 finetune/validate.py data/mix1/train.jsonl data/mix1/eval.jsonl     # once more on the final mix
```

`eval.jsonl` is the trainer's checkpoint-selection split. `train_pack.py` asserts it has exactly 580 rows, all
valid, so `mix.py` builds it from your task held-out, topped up with replay held-out. `task_eval.jsonl` keeps
your full task held-out for step 7. For small tasks, `--task-repeat 2` (or `convert_labeled.py --variants`)
repeats task rows; the trainer re-shuffles options for half the rows, so repeats are not identical prompts.

### 5. Train

Recommended settings:

| setting | full fine-tune | LoRA |
|---|---|---|
| learning rate | 5e-7 to 1e-6 (1e-6 default) | 3e-5 |
| warmup | 5–15% | 5% |
| epochs | 1 (a full pass over the mix) | 2 |
| batch | 256 rows | 16 rows |
| length | packed 8192 tokens; longer rows dropped | 8192 (one row per forward) |
| LoRA | – | rank 32, alpha 64, q/k/v/o/gate/up/down |
| option augmentation | 50% canonical, 50% shuffled + re-coded | same |

**Full fine-tune** (one node, 4–8 GPUs):

```bash
CHECKPOINT=/models/Nemotron-Deciffusion-8B-v2 finetune/train_fft.sh data/mix1 runs/fft1
# smoke test first:  ... finetune/train_fft.sh data/mix1 runs/smoke --limit-steps 5 --validate-every 5
# override via env: LR=7e-7 NPROC=4 BATCH=256 PYTHON=/path/to/venv/bin/python
```

Output: `runs/fft1/best/trained.safetensors` (decoder tensors at the best validation step),
`last/trained.safetensors`, `training.jsonl`, `validation.jsonl`, `selected.json`. Selection is by validation
accuracy, then cross-entropy. Evaluate both `best` and `last` in step 7.

Requirements: PyTorch with FlexAttention (the trainer compiles a per-document block mask), `transformers` able
to load the checkpoint's remote code, `safetensors`. Memory: 8B parameters × (fp32 weight + grad + 2 Adam
moments) ≈ 128 GB, sharded over the GPUs, plus activations for an 8k pack with checkpointing.

**LoRA** (1 GPU or `torchrun` on up to 8):

```bash
pip install peft
python3 finetune/train_lora.py --checkpoint $CHECKPOINT --data data/mix1 --output runs/lora1
torchrun --standalone --nproc-per-node=8 finetune/train_lora.py --checkpoint $CHECKPOINT --data data/mix1 --output runs/lora1
```

`train_lora.py` runs the same forward as `evaluation/native_eval.py` (prompt + mask, bidirectional decoder,
`diffusion_head` at the mask, restricted to option codes), with the same augmentation and soft-target loss as
`train_pack.py`. Its step-0 validation matches `eval_heldout.py --checkpoint` on the same rows. Run
`--limit-steps 5` first. Do not wrap its forward in `torch.autocast`: it recasts the model's fp32 rotary math to
bf16.

### 6. Soup with the starting checkpoint, then merge

A uniform weight average of the starting checkpoint and the fine-tuned run keeps most of the task gain and
protects general skills. Build the run and its soup, and evaluate both:

```bash
# full fine-tune run
python3 finetune/extract_decoder.py --checkpoint $CHECKPOINT --output runs/start.safetensors
python3 training/soup.py runs/fft1/soup.safetensors runs/start.safetensors runs/fft1/best/trained.safetensors
python3 training/merge_full.py --checkpoint $CHECKPOINT --trained runs/fft1/best/trained.safetensors --output merged/fft1
python3 training/merge_full.py --checkpoint $CHECKPOINT --trained runs/fft1/soup.safetensors --output merged/fft1-soup

# LoRA run: merge (also writes trained.safetensors so it can be souped the same way)
python3 finetune/merge_lora.py --checkpoint $CHECKPOINT --adapter runs/lora1/best \
    --output merged/lora1 --trained runs/lora1/trained.safetensors
python3 training/soup.py runs/lora1/soup.safetensors runs/start.safetensors runs/lora1/trained.safetensors
python3 training/merge_full.py --checkpoint $CHECKPOINT --trained runs/lora1/soup.safetensors --output merged/lora1-soup
```

`soup.py` is a uniform average of files with identical keys, all fine-tuned from the same base. List a file twice
to weight it 2:1. Merging needs CPU RAM of about twice the checkpoint size.

### 7. Evaluate

Evaluate the starting checkpoint, the run and the soup on the same files:

```bash
python3 finetune/eval_heldout.py --url http://localhost:8770 --input data/mix1/task_eval.jsonl \
    --save results/fft1-soup.task.jsonl --baseline results/start.task.jsonl
python3 finetune/eval_heldout.py --url http://localhost:8770 --input replay/heldout.jsonl --save results/fft1-soup.replay.jsonl
# or without a server, one GPU:
python3 finetune/eval_heldout.py --checkpoint merged/fft1-soup --input data/mix1/task_eval.jsonl
```

Look at task accuracy and calibration (log loss, Brier, ECE; noul AUC and mean p(true)), and at general skills
on your replay held-out set. Ship the candidate that gains on the task while holding the general sets; that is
often the soup. Do not pick checkpoints, mixes or learning rates on the set you report.

### 8. Serve

Run the release image with your merged checkpoint mounted at `CHECKPOINT_DIR`; the image holds no weights and serves
the checkpoint it finds there:

```bash
docker run -d --gpus all --shm-size=8g -p 8770:8770 \
    -v "$PWD/merged/fft1-soup:/models/custom:ro" -e CHECKPOINT_DIR=/models/custom \
    ghcr.io/pst2154/nemotron-deciffusion:8b-v2
curl -s localhost:8770/health
```

Keep `ATTENTION=noncausal` (default) and the compaction defaults (`COMPACT_OVER=8192`, `COMPACT_BUDGET=8192`)
that your training data was rendered for. FP8 serving is the default; `QUANTIZATION=none` serves BF16.

## Checklist

- Full fine-tuning for large shifts; LoRA for a narrow new task with little data.
- Train and serve bidirectional (`--attention noncausal`, `ATTENTION=noncausal`).
- Mix 40–60% replay; evaluate general skills, not just the task.
- Train full passes, then soup with the starting checkpoint; evaluate both.
- Low learning rate for continuation (5e-7 to 1e-6), 1 epoch, 256-row batches, 8192-token packs.
- Rows over 8192 tokens are dropped: pre-render long states and traces the way serving does.
- Balance labels within each type; shuffle and re-key options; vary instruction phrasing; cap
  "longest option is correct".
- Never leak the label into the state; keep a task-disjoint, frozen held-out split.
- Prefer soft labels from several teacher samples or verifier confidence; hard rule/outcome labels are fine.
- `train_pack.py` requires exactly 580 rows in `eval.jsonl`; `mix.py` produces that.
