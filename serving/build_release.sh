#!/usr/bin/env bash
# Build the release serving image (vllm serve + SystemOne adapter). No weights: the container downloads the pinned
# MODEL_ID revision at start, or serves a mounted checkpoint.
# usage: [MODEL_ID=org/name] build_release.sh <tag> [--platforms linux/amd64,linux/arm64] [--push] [--load]
#   Without --push the image stays in the build cache (or --load for a single platform) for local testing.
#   Pushing publishes the image.
set -euo pipefail
if [[ $# -lt 1 || $1 == -h || $1 == --help ]]; then
  echo "usage: [MODEL_ID=org/name] $(basename "$0") <tag> [--platforms linux/amd64,linux/arm64] [--push] [--load]" >&2
  exit 2
fi
tag=$1; shift
platforms=linux/amd64,linux/arm64; out=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --platforms) platforms=$2; shift 2 ;;
    --push) out+=(--push); shift ;;
    --load) out+=(--load); shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
here="$(cd "$(dirname "$0")" && pwd)"
overlay=${OVERLAY_DIR:-}  # vLLM checkout with the Deciffusion changes at commit 552f27b9; unset = reuse the staged vllm-overlay/

# Overlay: the 15 changed vLLM files (see vllm-overlay/README.md).
if [[ -n "$overlay" ]]; then
  rm -rf "$here/vllm-overlay"; mkdir -p "$here/vllm-overlay"
  (cd "$overlay" && git diff --name-only 552f27b9~1 552f27b9 -- vllm | tar cf - -T -) | tar xf - -C "$here/vllm-overlay"
fi
[[ -d "$here/vllm-overlay/vllm" ]] || { echo "no vllm-overlay/: set OVERLAY_DIR" >&2; exit 1; }
echo "overlay files: $(find "$here/vllm-overlay" -type f | wc -l)"
docker buildx build --platform "$platforms" --build-arg MODEL_ID="${MODEL_ID:-nemotron-deciffusion/Nemotron-Deciffusion-8B-v2}" \
  -f "$here/Dockerfile.release" -t "$tag" "${out[@]}" "$here"
