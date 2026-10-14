#!/usr/bin/env bash
# Start the Nemotron Deciffusion 8B container, wait until it is ready, and send examples/request.json.
# Usage: examples/quickstart.sh [image]   (needs Docker with the NVIDIA Container Toolkit and one NVIDIA GPU)
set -euo pipefail
image=${1:-ghcr.io/pst2154/nemotron-deciffusion:8b-v2}
here="$(cd "$(dirname "$0")" && pwd)"
docker run -d --name deciffusion --gpus all --shm-size=8g -p 8770:8770 \
  -v "$HOME/.cache/deciffusion:/models" -e HF_TOKEN "$image"
until curl -sf localhost:8770/health > /dev/null; do sleep 5; done  # first start downloads the weights (about 17 GB)
curl -s localhost:8770/v1/systemone -H 'Content-Type: application/json' -d @"$here/request.json"
echo
