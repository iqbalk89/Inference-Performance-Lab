#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --output=results/e2e-inference \
  python benchmark.py --device cuda --warmups 2 --runs 1 --output results/profile-measurement.json "$@"

echo "Created results/e2e-inference.nsys-rep"

