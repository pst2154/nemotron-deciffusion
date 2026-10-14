#!/usr/bin/env bash
# Full fine-tune (FFT) of Nemotron Deciffusion 8B: a thin wrapper around training/train_pack.py
# (FSDP2, packed FlexAttention, bidirectional attention, soft-target CE) with continuation defaults.
#
# Usage:
#   CHECKPOINT=/models/Nemotron-Deciffusion-8B finetune/train_fft.sh <data-dir> <output-dir> [extra train_pack.py args]
#
# <data-dir>   train.jsonl + eval.jsonl (exactly 580 rows: train_pack.py asserts it; finetune/mix.py writes it)
# <output-dir> must not exist. Gets best/trained.safetensors, last/trained.safetensors, training.jsonl,
#              validation.jsonl, selected.json. Log: <output-dir>.log
#
# Environment (defaults in brackets):
#   CHECKPOINT       model dir to start from (HF layout with remote code)   [required]
#   NPROC            GPUs on this node                                        [8]
#   PYTHON           python with torch (FlexAttention), transformers, safetensors   [python3]
#   LR [1e-6]  WARMUP [0.05]  EPOCHS [1]  BATCH [256 rows]  PACK_LEN [8192]  MAX_TOKENS [=PACK_LEN]
#   VALIDATE_EVERY [100]  CANONICAL_FRACTION [0.5]  SEED [101]
#   INIT             optional trained.safetensors to start from (decoder tensors of a run on the same CHECKPOINT)
#
# Extra args go to train_pack.py, e.g. --limit-steps 20 for a smoke run.
set -euo pipefail
if [ $# -lt 2 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,22p' "$0"; exit 0
fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$1 OUT=$2; shift 2
: "${CHECKPOINT:?set CHECKPOINT to the model dir to start from, e.g. a local copy of Nemotron-Deciffusion-8B}"
NPROC=${NPROC:-8}; PYTHON=${PYTHON:-python3}
LR=${LR:-1e-6}; WARMUP=${WARMUP:-0.05}; EPOCHS=${EPOCHS:-1}; BATCH=${BATCH:-256}
PACK_LEN=${PACK_LEN:-8192}; MAX_TOKENS=${MAX_TOKENS:-$PACK_LEN}; VALIDATE_EVERY=${VALIDATE_EVERY:-100}
CANONICAL_FRACTION=${CANONICAL_FRACTION:-0.5}; SEED=${SEED:-101}

test -f "$DATA/train.jsonl" || { echo "missing $DATA/train.jsonl" >&2; exit 1; }
test -f "$DATA/eval.jsonl" || { echo "missing $DATA/eval.jsonl (580 rows; see finetune/mix.py)" >&2; exit 1; }
n_eval=$(grep -c . "$DATA/eval.jsonl" || true)
[ "$n_eval" = 580 ] || { echo "$DATA/eval.jsonl has $n_eval rows; train_pack.py asserts 580 (mix.py --eval-size 580)" >&2; exit 1; }
n_train=$(grep -c . "$DATA/train.jsonl" || true)
[ "$n_train" -ge "$BATCH" ] || { echo "train.jsonl has $n_train rows < BATCH=$BATCH" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 1; }
[ $((BATCH % NPROC)) = 0 ] || { echo "BATCH must be divisible by NPROC" >&2; exit 1; }

extra=()
[ -n "${INIT:-}" ] && extra+=(--init "$INIT")

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} \
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
mkdir -p "$(dirname "$OUT")"
echo "train_pack: $n_train rows, $((n_train / BATCH * EPOCHS)) steps, lr $LR, $NPROC GPUs -> $OUT (log $OUT.log)"
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$NPROC" "$REPO/training/train_pack.py" \
  --checkpoint "$CHECKPOINT" --app "$REPO/serving/app" --data "$DATA" --output "$OUT" \
  --attention noncausal --lr "$LR" --warmup "$WARMUP" --epochs "$EPOCHS" --batch "$BATCH" \
  --pack-len "$PACK_LEN" --max-tokens "$MAX_TOKENS" --validate-every "$VALIDATE_EVERY" \
  --canonical-fraction "$CANONICAL_FRACTION" --seed "$SEED" ${extra[@]+"${extra[@]}"} "$@" 2>&1 | tee "$OUT.log"
echo "done. Next: finetune/README.md step 6 (soup + merge)."
