#!/usr/bin/env bash

set -euo pipefail

readonly NSYS_VERSION="${NSYS_VERSION:-2026.4.1.191}"
readonly NVIDIA_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub"
readonly NVIDIA_REPO="https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
readonly KEYRING_PATH="/usr/share/keyrings/nvidia-devtools-keyring.gpg"
readonly REPO_PATH="/etc/apt/sources.list.d/nvidia-devtools.list"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This bootstrap currently supports x86_64 GPU instances only." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; launch a Lambda GPU image first." >&2
  exit 1
fi

sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates \
  gnupg2 \
  nsight-compute \
  wget

if [[ ! -f "${KEYRING_PATH}" ]]; then
  wget -qO- "${NVIDIA_KEY_URL}" \
    | gpg --dearmor \
    | sudo tee "${KEYRING_PATH}" >/dev/null
fi

echo "deb [signed-by=${KEYRING_PATH}] ${NVIDIA_REPO} /" \
  | sudo tee "${REPO_PATH}" >/dev/null

sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  "nsight-systems-cli=${NSYS_VERSION}"

echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "Nsight Systems:"
/usr/local/bin/nsys --version

echo
echo "Nsight Compute:"
ncu --version

echo
echo "Use sudo with nsys when collecting CPU samples or scheduling events."
echo "The Lambda image sets kernel.perf_event_paranoid=4 for ordinary users."
