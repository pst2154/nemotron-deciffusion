---
library_name: transformers
license: other
license_name: nvidia-nemotron-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
base_model: nvidia/Nemotron-Labs-Diffusion-8B
pipeline_tag: text-classification
tags:
  - nvidia
  - nemotron
  - decision
  - classification
  - routing
  - long-context
---


# Nemotron Deciffusion 8B v2

**Big-model decisions. One GPU. One pass.**

Typed decisions in one forward pass. Give the model a JSON `state` and one or more questions, and it returns a
probability for every option of each question:
- `choice`: pick one of up to 501 options;
- `noul`: yes/no;
- `score`: ordered levels.

No text is generated, so a decision costs one prefill.

Built on [nvidia/Nemotron-Labs-Diffusion-8B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B) with full fine-tuning, bidirectional attention over
the whole prompt, and a masked-token readout over option codes.
It is trained on a fully traced data lineage in which every source is licensed for commercial training.

## Model summary

| | |
|---|---|
| Developer | Alex Steiner |
| Base model | `nvidia/Nemotron-Labs-Diffusion-8B` @ `16c67f0560b912e93e0cabb6e0c4f5c3086d95fc` |
| Parameters | 8B |
| Architecture | Nemotron-Labs-Diffusion decoder, served with bidirectional attention and a single mask-slot readout |
| Input | JSON `state` plus typed questions; prompts over 8,192 tokens are compacted by the server |
| Output | a probability for every option of every question |
| Precision | BF16 weights; NVFP4 in production on Blackwell ([nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4), mixed NVFP4/FP8 at 6.6 effective bits for Blackwell data-center GPUs (validated on GB200), faster than FP8 there); FP8 on other GPUs (container default) |
| Revision | `1dea38015e86e101eb978a33fa493818fc71e366`; `model.safetensors` sha256 `722396e8b46e34ba27db49584fea3b2be6c1797f62be58029c6ff142db072bc2` |
| Release date | 2026-10-14 |
| License | [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) |

## Evaluation

All numbers are BF16 except the Decision Index (FP8 container), scored on fixed items.

| benchmark | Nemotron Deciffusion 8B |
|---|---|
| JevBench public, top-1 | **210/231** |
| JevBench public, hard tier | **90/111** |
| Native eval split (decontaminated) | **528/578** |
| Native audit split | **618/678** |
| BABILong 4k, native | **99.2** |
| BABILong 8k, native | **98.8** |
| BABILong 16k, native | **98.4** |
| BABILong 16k, production compaction | **99.2** |
| BABILong 32k, production compaction | **99.6** |
| QuALITY, native | **80.3** |
| QuALITY, production path | **80.3** |
| Agent-run verify AUC (held-out coding-agent runs) | **0.725** |
| Continue-vs-escalate AUC (held-out coding-agent runs) | **0.575** |
| SWE-Atlas verify AUC (unseen harness) | **0.631** |
| Prompt injection: injected answer chosen, % (lower is better) | **4.93** |
| Decision Index 0.3 public index (full public suite) | **53.9** |
| Decision Index 0.3: Tools & Automation | **80.2** |
| Decision Index 0.3: Retrieval & Classification | **58.0** |
| Decision Index 0.3: Language Understanding | **56.9** |
| Decision Index 0.3: Arts & Human Taste | **37.8** |
| Decision Index 0.3: Knowledge & Reasoning | **35.3** |
| Aegis 2.0 prompt safety F1 | **87.4** |
| Aegis 2.0 response safety F1 | **83.0** |
| ToxicChat F1 | **73.5** |
| BEIR rerank nDCG@10 (FiQA, NFCorpus, SciFact) | **53.1** |
| Policy judge AUC: tool calls | **0.981** |
| Policy judge AUC: trajectories | **0.947** |

- **JevBench:** the 231 public items.
- **Production compaction:** the shipped server path. Prompts over 8,192 tokens are reduced to the most relevant 8,192
  tokens: BM25 for prose, head and tail for agent traces.
- **Sources:** every number's source file is listed at https://github.com/pst2154/nemotron-deciffusion/blob/main/release/v2/SOURCES.md.

### AA Decision Index: open components

We ran the three public components with AA's published methodology. AA-Omniscience-Decision is private and was not
run, so no AA index is listed for this model. The reference rows are AA's published scores (snapshot 2026-10-13).
When2Call's train split is part of our training data. Calibration uses AA's binning: 0.1-wide bins with at least 20
answers.

| model | AA index (published) | HLE-MC acc | When2Call acc | WildGuardTest F1 | mean of the 3 open components | ECE HLE-MC | ECE When2Call | ECE WildGuardTest |
|---|---|---|---|---|---|---|---|---|
| **Nemotron Deciffusion 8B v2** | – | 11.7 | 94.2 | 91.1 | 65.7 | 0.314 | 0.030 | 0.017 |
| Jev 1.13 | 57.5 | 20.9 | 80.7 | 93.6 | 65.0 | 0.402 | 0.053 | 0.048 |
| GLiDE | 53.9 | 10.5 | 83.7 | 95.1 | 63.1 | 0.381 | 0.012 | 0.012 |
| Microsoft-Decision-1 | 53.8 | 10.9 | 81.9 | 88.2 | 60.3 | 0.468 | 0.103 | 0.028 |
| Claude Haiku 5.5 (Low) | 60.6 | 23.6 | 72.6 | 92.6 | 62.9 | – | – | – |

### Latency

Latency of the v2 container (`vllm serve` + adapter):

| GPU | precision | prompt tokens | concurrency | p50 ms | p95 ms | decisions/s | measurement |
|---|---|---|---|---|---|---|---|
| L40S | FP8 (dynamic) | 362 | 1 | 29.8 | 93.5 | 24.05 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| L40S | FP8 (dynamic) | 362 | 16 | 539.1 | 893.1 | 28.56 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| L40S | FP8 (dynamic) | 850 | 1 | 59.5 | 129.4 | 14.19 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |
| L40S | FP8 (dynamic) | 850 | 16 | 984.3 | 1413.5 | 15.2 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |
| GB200 | FP8 (dynamic) | 362 | 1 | 24.9 | 29.0 | 28.75 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| GB200 | FP8 (dynamic) | 362 | 16 | 91.2 | 142.5 | 166.8 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| GB200 | FP8 (dynamic) | 850 | 1 | 26.1 | 31.3 | 37.01 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |
| GB200 | FP8 (dynamic) | 850 | 16 | 168.3 | 266.8 | 85.95 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |
| GB200 | BF16 | 362 | 1 | 25.8 | 33.7 | 26.86 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| GB200 | BF16 | 362 | 16 | 142.2 | 243.2 | 105.32 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| GB200 | BF16 | 850 | 1 | 26.7 | 41.2 | 33.6 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |
| GB200 | BF16 | 850 | 16 | 265.6 | 389.1 | 56.83 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |
| GB200 | NVFP4 | 362 | 1 | 24.1 | 28.7 | 29.0 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| GB200 | NVFP4 | 362 | 16 | 73.1 | 143.1 | 201.1 | vllm serve + adapter, AA DI items n=300; hle_mc (1 decision per task) |
| GB200 | NVFP4 | 850 | 1 | 25.1 | 45.7 | 35.53 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |
| GB200 | NVFP4 | 850 | 16 | 137.4 | 319.8 | 95.33 | vllm serve + adapter, AA DI items n=300; when2call (1 decision per task) |

## Intended use

- Routing and verification of agent runs: did the run solve the task, and should it continue or escalate.
- Tool selection and tool-call decisions: call a tool, ask, or answer.
- Policy, safety and compliance checks over a provided state, including long documents.
- Judging and reranking with probabilities.
- Classification and triage, with many questions about one state in a single request.

The model answers typed questions over the state it is given. It does not generate text. For automated decisions,
tune thresholds per deployment.

## Training

- **Method:** full fine-tuning of every decoder layer (PyTorch FSDP2, BF16 compute, FP32 reduction).
- **Objective:** soft-target cross-entropy at the mask slot, on the exact serving forward graph with bidirectional
  attention. Rows are packed to 8,192 tokens.

## Training data

- **Licensing.** Every training source carries a license that permits commercial training. Training data includes no
  non-commercial, research-only or unlicensed data, and no benchmark test, validation or dev splits.
- **Verification.** Every row passed a strict row-level check covering its license, every model that generated or
  labelled it, and overlap with protected evaluation sets.
- **Labels.** Labels come from four places: dataset gold or human labels, deterministic rules, base-model soft
  targets, or votes of open-weight models with commercially usable terms.

Datasets used under attribution licenses:

- [Aegis 2.0 / Nemotron Content Safety Dataset V2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0): CC-BY-4.0
- [When2Call](https://huggingface.co/datasets/nvidia/When2Call): CC-BY-4.0
- [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2): CC-BY-4.0
- [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3): CC-BY-4.0
- [NVIDIA OpenScience](https://huggingface.co/datasets/nvidia/OpenScience): CC-BY-4.0
- [Nemotron-RL knowledge MCQA](https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa): CC-BY-4.0
- [Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1): CC-BY-4.0
- [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces): CC-BY-4.0
- [SWE-rebench OpenHands trajectories (Nebius)](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories): CC-BY-4.0
- [MASSIVE (Amazon)](https://huggingface.co/datasets/AmazonScience/massive): CC-BY-4.0
- [QuALITY (Pang et al., NAACL 2022), train split](https://github.com/nyu-mll/quality): CC-BY-4.0
- [MuSiQue](https://github.com/StonyBrookNLP/musique): CC-BY-4.0
- [HotpotQA](https://hotpotqa.github.io/): CC-BY-SA-4.0
- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/): CC-BY-SA-4.0
- [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc): CC-BY-SA-4.0
- [Natural Questions (Google)](https://ai.google.com/research/NaturalQuestions): CC-BY-SA-3.0
- [BoolQ](https://github.com/google-research-datasets/boolean-questions): CC-BY-SA-3.0
- [WikiHop / QAngaroo](https://qangaroo.cs.ucl.ac.uk/): CC-BY-SA-3.0
- [ShARC](https://sharc-data.github.io/): CC-BY-SA-3.0
- [tasksource collection (CC-BY, CC-BY-SA and ODC-BY subsources)](https://huggingface.co/tasksource): per subsource
- [PRM800K (OpenAI)](https://github.com/openai/prm800k): MIT
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf): MIT

## Serving

The supported runtime is `vllm serve`, with the decision-read changes being upstreamed in vllm-project/vllm#60283, plus the
SystemOne adapter, packaged in one container.

| image | weights | precision |
|---|---|---|
| `ghcr.io/pst2154/nemotron-deciffusion:8b-v2` | downloads `nemotron-deciffusion/Nemotron-Deciffusion-8B-v2` (pinned revision) into `/models` on first start | FP8 by default (`QUANTIZATION=fp8`); `QUANTIZATION=none` for BF16 |
| `ghcr.io/pst2154/nemotron-deciffusion:8b-v2` with `MODEL_ID=nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4`, `QUANTIZATION=none` and `-v $HOME/.cache/deciffusion-nvfp4:/models` | [`nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4`](https://huggingface.co/nemotron-deciffusion/Nemotron-Deciffusion-8B-v2-NVFP4) | mixed NVFP4/FP8 for Blackwell data-center GPUs (validated on GB200) |

```bash
docker run -d --gpus all --shm-size=8g -p 8770:8770 -v $HOME/.cache/deciffusion:/models -e HF_TOKEN ghcr.io/pst2154/nemotron-deciffusion:8b-v2
until curl -sf localhost:8770/health > /dev/null; do sleep 5; done  # 503 until vLLM is ready
curl -s localhost:8770/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "nemotron-deciffusion-8b",
  "state": {"ticket": "Customer was charged twice for order 1182."},
  "questions": {"route": {"type": "choice", "instructions": "Which team should handle this?",
    "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "tech": "Bugs"}}}}'
```

Inside the container, `serving/start.sh` runs (simplified):

```bash
vllm serve <checkpoint> --trust-remote-code --dtype bfloat16 --quantization fp8 \
  --hf-overrides '{"ar_mode": true, "is_causal": false}' --no-enable-prefix-caching \
  --max-logprobs 1052 --logprobs-mode processed_logits --port 8200 &
UPSTREAM=http://127.0.0.1:8200 PORT=8770 python3 adapter.py
```

- **Defaults:** `ATTENTION=noncausal`, `COMPACT_OVER=8192`, `COMPACT_BUDGET=8192`, `MAX_BATCHED_TOKENS=65536`.
- **Request limits:** up to 512 questions per request and 501 options per question.
- **Code:** the adapter and the vLLM overlay files are at https://github.com/pst2154/nemotron-deciffusion.

## License/Terms of Use

Use of this model is governed by the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).
