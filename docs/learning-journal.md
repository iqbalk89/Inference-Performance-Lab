# Learning Journal

This journal records progress, questions, useful commands, and ideas throughout
the Inference Performance Lab roadmap.

## Phase 0 — Development Environment and AI Workstation Setup

### Session 1 — AI Development Environment

**Date:** July 29, 2026

#### Goals

- Understand why the core tools in an AI inference workflow exist.
- Verify the local development environment.
- Configure VS Code for Python, notebooks, Docker, and containers.
- Create a professional repository for future inference work.
- Identify which work can run locally and which work requires remote hardware.

#### Things I Learned

- **Git** records changes to files, supports safe experimentation, and makes
  collaboration possible through branches and shared history.
- **GitHub** hosts Git repositories and adds collaboration tools such as pull
  requests, issue tracking, code review, and automation.
- **Python virtual environments** isolate project dependencies so projects can
  use different package versions without interfering with one another.
- **Docker** packages an application with its runtime and system dependencies,
  making execution more consistent across machines.
- **Docker Compose** describes and operates multiple related containers as one
  application.
- **CUDA** is NVIDIA's software platform for programming NVIDIA GPUs. It cannot
  run on this Mac because the machine has Intel and AMD graphics, not NVIDIA
  graphics.
- **PyTorch** provides tensor operations, automatic differentiation, neural
  network components, and hardware-aware execution for machine-learning
  workloads.
- **VS Code** combines editing, terminals, debugging, source control, notebooks,
  containers, and remote development in one workspace.
- **NVIDIA drivers** allow the operating system and compute software to
  communicate with NVIDIA hardware.
- **Hugging Face** provides model repositories, datasets, documentation, and
  libraries used to download and work with pretrained models.
- This Mac is suitable for development, Docker, CPU testing, and small CPU
  inference tasks. CUDA and NVIDIA GPU work must run on a remote machine.

#### Questions

- Which remote NVIDIA GPU offers the best price-to-performance ratio for the
  early phases of this curriculum?
- Which Python version should become the project's standard when library support
  differs between Python 3.12 and 3.13?
- How does a local Python client communicate with an inference server running in
  a container or on a remote machine?
- What model size and quantization level will run comfortably within this
  machine's CPU and memory limits?

#### Useful Commands

```bash
# Check installed tools
python3 --version
git --version
docker --version
docker compose version
code --version

# Inspect the machine
sw_vers
uname -m
system_profiler SPHardwareDataType SPDisplaysDataType

# List VS Code extensions
code --list-extensions

# Inspect and test Docker
docker info
docker run --rm hello-world

# Inspect the repository
git status

# Create and use the project virtual environment
python3 -m venv .venv
source .venv/bin/activate
python --version
deactivate
```

#### Ideas

- Add a small script that reports the active Python, PyTorch, and accelerator
  environment.
- Use the same benchmark interface for local CPU tests and remote GPU tests.
- Save machine-readable environment metadata with every benchmark result.
- Compare direct Python inference with inference through a local HTTP server.

#### Next Steps

- Review and commit the Session 1 repository.
- Use the verified `.venv` environment for project-specific Python packages.
- Select a project Python version based on inference-library compatibility.
- Run a minimal Python program and automated test.
- Try a small CPU-compatible language model locally.
- Plan access to remote Linux hardware with an NVIDIA GPU before the CUDA
  session.

#### Personal Reflection

Add a few sentences after reviewing this entry:

- What surprised me:
- What remains unclear:
- What I want to remember:

## Phase 1 — GPU Foundation and Inference Fundamentals

### Checkpoint — Remote GPU Acceptance

**Date:** August 1, 2026

#### Goals

- Select a remote NVIDIA environment suitable for Phase 1.
- Verify CUDA-enabled PyTorch and NVIDIA profiling access.
- Make future disposable instances reproducible.

#### Things I Learned

- The Intel Mac remains the development and analysis machine; NVIDIA-specific
  execution must occur on remote Linux hardware.
- A Lambda A10 exposes approximately 23 GB of usable VRAM and supports the
  counters needed for this phase.
- GPU Metrics and Nsight Compute counters are different from CPU call-stack
  sampling. The latter relies on Linux `perf_event_open` permissions.
- Lambda's `perf_event_paranoid=4` blocks ordinary-user CPU sampling, but
  elevated Nsight Systems collection works without permanently weakening that
  setting.
- A profiler can collect data but still fail to generate a usable report when
  its importer libraries are broken. Installing NVIDIA's official current CLI
  resolved the observed `LIBSSH_4_9_0` mismatch.
- Disposable GPU instances need two forms of reproducibility: versioned setup
  automation and a disciplined procedure for preserving results before
  termination.

#### Useful Commands

```bash
# Install and verify the base remote GPU tooling
./scripts/bootstrap_lambda_gpu.sh

# Run the complete automated profiler acceptance check
./scripts/bootstrap_lambda_gpu.sh --validate

# Confirm GPU processes, utilization, and memory
nvidia-smi
```

#### Questions

- Which small decoder-only model should be the first controlled workload?
- Which Transformers version should be pinned alongside the existing CUDA
  PyTorch installation?
- What is the cleanest way to isolate Python dependencies without replacing
  Lambda Stack's working CUDA build?
- Which PyTorch Profiler views best expose prefill and decode behavior?

#### Next Steps

- Learn the foundational GPU concepts listed in the Phase 1 checkpoint.
- Select and pin the first model and dependency environment.
- Run model inference and record generation, utilization, and VRAM metadata.
- Capture and inspect the first model-level PyTorch Profiler trace.
- Use the validated Nsight workflow on the same model workload.
