# vLLM overlay

These 15 files are copied into the vLLM installation by [`../Dockerfile.release`](../Dockerfile.release). They add the
Nemotron-Labs-Diffusion model, bidirectional decision reads (`ar_mode` with `is_causal=False`), wide option sets and
admission control for prompts that cannot be scheduled.

- **Base:** vLLM nightly `fe4df3df0eb703862c686b256e9fe44fa94c8945` (the `VLLM_IMAGE` in the Dockerfile).
- **Upstream:** the same changes are proposed under vllm-project/vllm#60283. Once they are released, the overlay is no longer needed.
- **License:** the files come from the vLLM project and are under the Apache License 2.0; each file keeps its
  `SPDX-License-Identifier` header.
