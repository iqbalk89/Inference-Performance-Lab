# Inference Performance Lab

Inference Performance Lab is a long-term engineering workspace for learning how
modern language-model inference systems are built, measured, optimized, and
operated.

The repository is intended to grow into a portfolio of reproducible experiments
and working systems rather than a collection of disconnected tutorials.

## Long-Term Goals

- Build a strong foundation in machine learning and transformer inference.
- Run and evaluate language models locally and on remote NVIDIA hardware.
- Understand inference engines, model serving, batching, caching, quantization,
  and distributed inference.
- Measure latency, throughput, memory use, and hardware utilization.
- Learn profiling and performance-debugging techniques.
- Develop production-oriented Python, Docker, testing, and documentation
  practices.
- Prepare for technical interviews and engineering roles focused on AI
  inference performance.

## Areas of Study

- Python and PyTorch
- Transformer architecture and inference
- CPU and GPU execution
- CUDA and NVIDIA GPU fundamentals
- Model formats and quantization
- Inference servers and APIs
- Docker and reproducible environments
- Benchmarking and profiling
- Latency, throughput, memory, and cost analysis
- Reliability, testing, and observability

## Development Environment

Local development currently uses an Intel-based MacBook Pro without an NVIDIA
GPU. It supports source development, Docker workflows, CPU testing, and small
CPU inference workloads.

CUDA development, NVIDIA profiling, vLLM, and GPU-intensive experiments will be
performed later on remote Linux hardware with an NVIDIA GPU. See
[Development Environment Notes](docs/development-environment-notes.md) for the
verified environment.

## Repository Organization

| Path | Purpose |
| --- | --- |
| `benchmarks/` | Repeatable performance tests and benchmark definitions |
| `docker/` | Dockerfiles, Compose files, and container configuration |
| `docs/` | Environment, architecture, GPU, reading, and learning notes |
| `experiments/` | Focused investigations with documented results |
| `inference-servers/` | Model-serving implementations and configurations |
| `interview-notes/` | Concepts, questions, and interview preparation |
| `notebooks/` | Exploratory analysis and visualizations |
| `profiling/` | Profiling scripts, configurations, and analysis |
| `scripts/` | Development, automation, and utility scripts |
| `src/` | Reusable Python source code |
| `tests/` | Automated tests |

## Experiment Standard

Every substantive experiment should document:

1. Objective
2. Hypothesis
3. Methodology
4. Environment and dependencies
5. Results
6. Analysis
7. Lessons learned

Measurements should be reproducible, units should be explicit, and raw results
should be kept separate from interpretation.

## Current Status

Phase 0 is complete: the local development tools, Docker workflow, virtual
environment, and GitHub repository have been initialized and verified.

Phase 1 is beginning with GPU foundations and inference fundamentals. CUDA and
GPU profiling will run on remote NVIDIA hardware. See the
[Phase 1 Plan](docs/phase-1-plan.md) for the hardware-adjusted roadmap.
