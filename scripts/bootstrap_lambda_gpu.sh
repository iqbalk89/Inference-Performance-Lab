#!/usr/bin/env bash

set -euo pipefail

readonly NSYS_VERSION="${NSYS_VERSION:-2026.4.1.191}"
readonly NVIDIA_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub"
readonly NVIDIA_REPO="https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/"
readonly KEYRING_PATH="/usr/share/keyrings/nvidia-devtools-keyring.gpg"
readonly REPO_PATH="/etc/apt/sources.list.d/nvidia-devtools.list"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VALIDATE=false

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap_lambda_gpu.sh [--validate]

Installs and verifies the Phase 1 GPU profiling tools on a Lambda Cloud
Ubuntu 22.04 x86_64 GPU instance. Re-running the script is safe.

Options:
  --validate  Also run short PyTorch, Nsight Systems, and Nsight Compute tests.
  -h, --help  Show this help text.
EOF
}

while (($#)); do
  case "$1" in
    --validate) VALIDATE=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This bootstrap currently supports x86_64 GPU instances only." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify the operating system." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "This bootstrap is pinned to Ubuntu 22.04; found ${PRETTY_NAME:-unknown}." >&2
  exit 1
fi

for command_name in nvidia-smi python3 sudo; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command is unavailable: ${command_name}" >&2
    exit 1
  fi
done

if ! sudo -n true; then
  echo "This account does not have noninteractive sudo access." >&2
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
echo "PyTorch CUDA:"
python3 - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")

print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
PY

echo
echo "Nsight Systems:"
/usr/local/bin/nsys --version

echo
echo "Nsight Compute:"
ncu --version

if [[ "${VALIDATE}" == false ]]; then
  echo
  echo "Bootstrap complete. Run again with --validate for profiler acceptance tests."
  echo "Use sudo with nsys when collecting CPU samples or scheduling events."
  exit 0
fi

readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_DIR="${REPO_ROOT}/profile-results/bootstrap-validation-${RUN_ID}"
readonly SMOKE_TEST="${REPO_ROOT}/scripts/gpu_smoke_test.py"

if [[ ! -f "${SMOKE_TEST}" ]]; then
  echo "Missing validation workload: ${SMOKE_TEST}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo
echo "Running CUDA smoke test..."
python3 "${SMOKE_TEST}" --iterations 5 \
  | tee "${RESULT_DIR}/pytorch-smoke-test.txt"

echo
echo "Checking elevated CPU profiling support..."
sudo /usr/local/bin/nsys status --environment \
  | tee "${RESULT_DIR}/nsys-environment.txt"

echo
echo "Collecting combined CPU and GPU Nsight Systems trace..."
sudo /usr/local/bin/nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --backtrace=dwarf \
  --cpuctxsw=process-tree \
  --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=1000 \
  --output="${RESULT_DIR}/nsys-combined" \
  python3 "${SMOKE_TEST}" --iterations 20
sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/nsys-combined.nsys-rep"
/usr/local/bin/nsys stats "${RESULT_DIR}/nsys-combined.nsys-rep" \
  > "${RESULT_DIR}/nsys-stats.txt"

echo
echo "Collecting one-kernel Nsight Compute report..."
sudo ncu \
  --target-processes all \
  --set basic \
  --launch-count 1 \
  --force-overwrite \
  --export "${RESULT_DIR}/ncu-basic" \
  python3 "${SMOKE_TEST}" --size 1024 --warmup 0 --iterations 1
sudo chown "$(id -u):$(id -g)" "${RESULT_DIR}/ncu-basic.ncu-rep"

for artifact in \
  "${RESULT_DIR}/nsys-combined.nsys-rep" \
  "${RESULT_DIR}/nsys-stats.txt" \
  "${RESULT_DIR}/ncu-basic.ncu-rep"; do
  if [[ ! -s "${artifact}" ]]; then
    echo "Validation artifact is missing or empty: ${artifact}" >&2
    exit 1
  fi
done

echo
echo "Validation passed. Artifacts: ${RESULT_DIR}"
echo "The profile-results directory is intentionally excluded from Git."
