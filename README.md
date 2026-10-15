# Nemotron Deciffusion 8B

### Big-model decisions. One GPU. One pass.


**Typed decisions in one forward pass.** Give the model a JSON `state` and up to 512 questions. It returns a
probability for every option of every `choice`, `noul` (yes/no) or `score` question: no generated text, no reasoning
tokens, one bidirectional prefill on an NVIDIA GPU.

Built on [NVIDIA Nemotron-Labs-Diffusion-8B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B). The current release is **v2**
(2026-10-14), trained on a fully traced data lineage in which every source is licensed for commercial training.

| Get it | |
|---|---|
| **Container** | `ghcr.io/pst2154/nemotron-deciffusion:8b-v2`: `vllm serve` + SystemOne adapter, downloads the weights on first start, FP8 by default |
| **Weights** | [`nemotron-deciffusion/Nemotron-Deciffusion-8B-v2`](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2) (BF16, revision `1dea38015e86e101eb978a33fa493818fc71e366`); NVFP4: [nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4), mixed NVFP4/FP8 at 6.6 effective bits for Blackwell data-center GPUs (validated on GB200), faster than FP8 there |
| **Release post** | [https://pst2154.github.io/nemotron-deciffusion/](https://pst2154.github.io/nemotron-deciffusion/) (source: [docs/BLOG_v2.md](docs/BLOG_v2.md)) |
| **Model card** | [MODEL_CARD_v2.md](MODEL_CARD_v2.md) |

<p align="center"><img src="docs/site/assets/gifs/one_forward_pass.gif" width="760" alt="State and question go through one forward pass; the mask slot reads out a probability for every option."></p>

## Results

All numbers are BF16 except the Decision Index (FP8 container), scored on fixed items
([sources](release/v2/SOURCES.md)):

| benchmark | Nemotron Deciffusion 8B |
|---|---|
| JevBench public, top-1 | **210/231** |
| JevBench public, hard tier | **90/111** |
| BABILong 32k, production compaction | **99.6** |
| QuALITY, native | **80.3** |
| Agent-run verify AUC (held-out coding-agent runs) | **0.725** |
| Prompt injection: injected answer chosen, % (lower is better) | **4.93** |
| Decision Index 0.3 public index (full public suite) | **53.9** |
| Aegis 2.0 prompt safety F1 | **87.4** |
| Aegis 2.0 response safety F1 | **83.0** |
| BEIR rerank nDCG@10 (FiQA, NFCorpus, SciFact) | **53.1** |

The full results, the AA Decision Index open components (HLE-MC, When2Call, WildGuardTest and calibration) and the
evaluation methodology are in the [release post](docs/BLOG_v2.md) and the [model card](MODEL_CARD_v2.md).

## Run it

```bash
docker run -d --name deciffusion --gpus all --shm-size=8g -p 8770:8770 \
  -v $HOME/.cache/deciffusion:/models -e HF_TOKEN ghcr.io/pst2154/nemotron-deciffusion:8b-v2
until curl -sf localhost:8770/health > /dev/null; do sleep 5; done  # first start downloads the weights (about 17 GB)
curl -s localhost:8770/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json
```

[`examples/request.json`](examples/request.json) routes a support ticket and asks whether it is urgent. Every option
in the response has a probability. [`examples/quickstart.sh`](examples/quickstart.sh) runs all three steps.

The image runs `vllm serve` on a private port and the SystemOne adapter on `:8770`. See
[`serving/start.sh`](serving/start.sh) and [`serving/app/adapter.py`](serving/app/adapter.py). `/health` returns 503
until vLLM is ready.

| Setting | Default | |
|---|---|---|
| `QUANTIZATION` | `fp8` | `none` serves BF16, the precision of every published evaluation |
| `ATTENTION` | `noncausal` | the model is trained with bidirectional attention |
| `COMPACT_OVER` / `COMPACT_BUDGET` | `8192` / `8192` | prompts above `COMPACT_OVER` tokens are compacted to `COMPACT_BUDGET`; `0` disables compaction |
| `MAX_BATCHED_TOKENS` | `65536` | |

### NVFP4 on Blackwell

[`nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4`](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4) is a mixed NVFP4/FP8 checkpoint (6.6 effective bits) for
Blackwell data-center GPUs, validated on GB200 (sm_100). Run the same image with `MODEL_ID` set to it and its own cache volume:

```bash
docker run -d --name deciffusion-nvfp4 --gpus all --shm-size=8g -p 8770:8770 \
  -v $HOME/.cache/deciffusion-nvfp4:/models -e HF_TOKEN \
  -e MODEL_ID=nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4 -e QUANTIZATION=none ghcr.io/pst2154/nemotron-deciffusion:8b-v2
```

`QUANTIZATION=none` because the quantization config is read from the checkpoint. On sm_100 the container selects the
CUTLASS FP4 GEMM (`LINEAR_BACKEND=cutlass`).

### Two-round elimination (optional)

Add `"elimination": true` to a request (or start the container with `-e ELIMINATION=1` to make it the default). Each
choice question with more than two options gets a second read over its top two options, with the other options named as
ruled out; the two then split their first-round probability. On held-out sets from hard reasoning tasks this adds
1.3 points of accuracy (BBH +5.3) for about 15% more latency per request (19 ms to 22 ms median on one H200).

## Build the container

The bidirectional decision reads need vLLM changes that are being upstreamed (vllm-project/vllm#60283). Until they are released,
the image applies them as an overlay on a vLLM nightly. The overlay is in
[`serving/vllm-overlay/`](serving/vllm-overlay/). [`serving/Dockerfile.release`](serving/Dockerfile.release) builds the
image, and [`serving/build_release.sh`](serving/build_release.sh) runs the build (amd64 and arm64 by default). The image
holds no weights: on first start it downloads the pinned Hugging Face revision into `/models`, or serves a checkpoint
you mount at `CHECKPOINT_DIR`.

```bash
git clone https://github.com/pst2154/nemotron-deciffusion && cd nemotron-deciffusion
serving/build_release.sh deciffusion:local --platforms linux/amd64 --load
```

## How it works

- **Bidirectional attention** over the whole prompt: every token sees the question and every option.
- **Masked readout:** options get single-token codes, and one mask slot reads a distribution over them.
- **Full fine-tuning** with soft-target cross-entropy on the exact serving graph.
- **Compaction** for long prompts, in [`serving/app/compaction.py`](serving/app/compaction.py):
  - prose keeps its most question-relevant sentences (BM25);
  - agent traces keep their head and tail.

## Reproduce the evaluation

[`evaluation/`](evaluation/) has the native single-pass evaluator, JevBench scoring and the Decision Index engine.
See [evaluation/README.md](evaluation/README.md).

## Layout

- `serving/`: `vllm serve` start script, SystemOne adapter, compaction, vLLM overlay, Dockerfile, tests
- `finetune/`: fine-tune the model on your own decisions, with LoRA or full fine-tuning
- `training/train_pack.py`: the full fine-tuning trainer used for v2
- `examples/`: a minimal request and quickstart
- `evaluation/`: evaluators used for the published numbers, including the AA Decision Index open components
- `docs/`: release post, and the launch site with the Decision Playground (`docs/site/`)
- `release/v2/`: every published number with its source (`SOURCES.md`, `numbers.json`)

## License

- **Model:** use of this model is governed by the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).
- **Code:** Apache License 2.0; see [LICENSE](LICENSE).
- **vLLM overlay:** the files in `serving/vllm-overlay/` come from vLLM and are under the Apache License 2.0.

## Disclaimer

Nemotron Deciffusion is a personal project. It is not an NVIDIA product and is not released, endorsed or supported by NVIDIA.
