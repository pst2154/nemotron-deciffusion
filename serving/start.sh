#!/usr/bin/env bash
# vLLM's OpenAI server (async engine, cross-request batching) on a private port, the SystemOne adapter on $PORT.
# Extra arguments are passed to `vllm serve`.
# /health on $PORT reports 503 until vLLM is ready, so it doubles as the readiness probe.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

# Find the checkpoint: the configured CHECKPOINT_DIR, or a platform model mount ($NIM_MODEL_NAME or /config/models,
# including the same snapshot path under it). If a mount is expected, wait for it.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/models/checkpoint}"
find_checkpoint() {
  local snap="${CHECKPOINT_DIR#/config/models/*/}" c root
  for c in "$CHECKPOINT_DIR" "${NIM_MODEL_NAME:-}/$snap" "${NIM_MODEL_NAME:-}"; do
    [[ -n "$c" && "$c" != / && -f "$c/config.json" ]] && { echo "$c"; return 0; }
  done
  for root in "${NIM_MODEL_NAME:-}" /config/models; do
    [[ -n "$root" && -d "$root" ]] || continue
    c=$(find -L "$root" -maxdepth 7 -name config.json -print -quit 2>/dev/null || true)
    [[ -n "$c" ]] && { dirname "$c"; return 0; }
  done
  return 1
}
found=""
for ((i = 0; i <= ${CHECKPOINT_WAIT_SECS:-600}; i += 10)); do
  found=$(find_checkpoint) && break
  [[ -z "${NIM_MODEL_NAME:-}" && ! -d /config/models ]] && break  # nothing will be mounted later
  sleep 10
done
if [[ -z "$found" && -z "${NIM_MODEL_NAME:-}" && "${DOWNLOAD_CHECKPOINT:-1}" == 1 ]]; then
  # Weights are not in the image: fetch the pinned MODEL_ID revision into CHECKPOINT_DIR (mount a volume there to keep
  # it). A private repo needs HF_TOKEN in the environment.
  echo "No checkpoint at $CHECKPOINT_DIR: downloading ${MODEL_ID:-<default MODEL_ID>} (pinned revision) ..."
  mkdir -p "$CHECKPOINT_DIR"
  HF_HUB_OFFLINE=0 HF_HOME="${HF_HOME:-/tmp/hf}" python3 -u "$here/download.py" \
    || { echo "FATAL: download of ${MODEL_ID:-the checkpoint} failed (set HF_TOKEN for a private repo, or mount a checkpoint at CHECKPOINT_DIR)" >&2; exit 2; }
  found=$(find_checkpoint) || true
fi
if [[ -z "$found" ]]; then
  echo "FATAL: no config.json for the Nemotron Deciffusion checkpoint (looked at CHECKPOINT_DIR=$CHECKPOINT_DIR," \
       "NIM_MODEL_NAME=${NIM_MODEL_NAME:-<unset>} and /config/models)" >&2
  find /config/models -maxdepth 6 2>/dev/null | head -80 >&2 || true
  exit 2
fi
[[ "$found" != "$CHECKPOINT_DIR" ]] && echo "Using checkpoint at $found (configured: $CHECKPOINT_DIR)"
export CHECKPOINT_DIR="$found"

# The model mount is read-only; remote-code modules and every JIT or compile cache live on writable scratch.
# read-only root / arbitrary UID: libraries that cache under $HOME (FlashInfer JIT) need a writable home
if ! { mkdir -p "${HOME:-/nonexistent}/.cache" && [ -w "$HOME/.cache" ]; } 2>/dev/null; then export HOME=/tmp/home; fi
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$HOME}"
export HF_HOME="${HF_HOME:-/tmp/hf}" HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf/modules}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"  # bidirectional reads use the V2 runner
mkdir -p "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

upstream_port="${UPSTREAM_PORT:-8200}"
export ATTENTION="${ATTENTION:-noncausal}"
if [[ "$ATTENTION" == noncausal ]]; then
  # One bidirectional forward over prompt + mask; KV cannot be shared across prompts, so no prefix caching.
  overrides='{"ar_mode": true, "is_causal": false}'; caching=--no-enable-prefix-caching; default_batched=65536
elif [[ "$ATTENTION" == causal ]]; then
  overrides='{"ar_mode": true}'; caching=--enable-prefix-caching; default_batched=8192
else
  echo "FATAL: ATTENTION must be causal or noncausal" >&2; exit 2
fi
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$default_batched}"
logprobs_mode=$([[ "${DIRECT_LOGITS:-1}" == 1 ]] && echo processed_logits || echo processed_logprobs)
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_ID:-Nemotron-Deciffusion-8B}")}"
export QUANTIZATION="${QUANTIZATION:-none}"
case "$QUANTIZATION" in
  none) quant=() ;;
  fp8) quant=(--quantization fp8) ;;  # dynamic per-tensor FP8 weights+activations; attention stays BF16
  *) echo "FATAL: QUANTIZATION must be none or fp8" >&2; exit 2 ;;
esac

# DGX Spark / GB10 (sm_121, unified memory, passively limited cooling): FlashInfer's startup autotune runs every
# token bucket up to MAX_BATCHED_TOKENS at full power for minutes and drove a GB10 to 86 C and a hard freeze. Off by
# default there; also leave memory headroom for the host and use a 16k context (compacted prompts are ~8.6k).
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ "$compute_cap" == 12.1 ]]; then
  : "${FLASHINFER_AUTOTUNE:=0}" "${GPU_MEMORY_UTILIZATION:=0.7}" "${MAX_MODEL_LEN:=16384}"
  # GB10's FP8/FP4 GEMMs lose 2-4x efficiency once one scheduler step exceeds ~4k tokens (CUTLASS FP8 at M=6144 is
  # slower than BF16), so batching 8-16 router-sized (~1k token) requests made concurrency slower than sequential
  # (FP8: 6.9 req/s at c=1, 3.6 at c=16). Four sequences per step keep steps small: FP8 c=16 6.7 req/s, NVFP4 8.4,
  # with unchanged single-request latency. Bidirectional reads have no KV cache, so this costs no memory.
  : "${MAX_NUM_SEQS:=4}"
fi
# NVFP4 checkpoints on B200/GB200 (sm_100): vLLM's default FlashInfer CuTe-DSL FP4 GEMM adds ~12 ms per request at low
# concurrency; the CUTLASS FP4 GEMM matches FP8 latency and serves 11-21% more tasks/s than FP8 at 16 concurrent
# (GB200). LINEAR_BACKEND overrides (auto, cutlass, flashinfer_cutlass, flashinfer_cutedsl, ...).
# GB10 (sm_121): the CUTLASS FP4/FP8 GEMMs are ~5% faster per request than the FlashInfer ones (111 vs 117 ms p50).
if [[ -z "${LINEAR_BACKEND:-}" && ( "$compute_cap" == 10.* || "$compute_cap" == 12.1 ) ]] \
   && grep -qs '"quant_algo": *"NVFP4"' "$CHECKPOINT_DIR/hf_quant_config.json"; then
  LINEAR_BACKEND=cutlass
fi
backend=()
[[ -n "${LINEAR_BACKEND:-}" && "${LINEAR_BACKEND}" != auto ]] && backend=(--linear-backend "$LINEAR_BACKEND")
kernel=()
case "${FLASHINFER_AUTOTUNE:-1}" in
  1) ;;
  0) kernel=(--kernel-config '{"enable_flashinfer_autotune": false}') ;;
  *) echo "FATAL: FLASHINFER_AUTOTUNE must be 0 or 1" >&2; exit 2 ;;
esac

cleanup() { kill ${adapter_pid:-} ${upstream_pid:-} 2>/dev/null || true; }
trap cleanup EXIT INT TERM

vllm serve "$CHECKPOINT_DIR" --served-model-name "$SERVED_MODEL_NAME" --host 127.0.0.1 --port "$upstream_port" \
  --trust-remote-code --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-logprobs 1052 --logprobs-mode "$logprobs_mode" \
  --max-num-seqs "${MAX_NUM_SEQS:-128}" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --hf-overrides "$overrides" $caching "${quant[@]}" "${kernel[@]}" "${backend[@]}" --disable-uvicorn-access-log "$@" &
upstream_pid=$!

# The adapter starts at once (health 503 until vLLM is up) so the platform sees a listening port.
UPSTREAM="http://127.0.0.1:${upstream_port}" PORT="${PORT:-8000}" python3 -u "$here/adapter.py" &
adapter_pid=$!
wait -n "$upstream_pid" "$adapter_pid"
