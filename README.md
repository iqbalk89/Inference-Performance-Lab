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
| `curriculum/` | Canonical phase → module → lesson/exercises/lab learning path |
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
GPU profiling will run on remote NVIDIA hardware. Start at the
[Curriculum Index](curriculum/README.md). The primary job-oriented route is now
the [Inference Performance Engineering track](curriculum/inference-performance-engineering/README.md),
which separates the end-to-end pipeline, prefill, decode, profiling, capacity,
and optimization. The numbered
[Phase 1 modules](curriculum/phase-01-gpu-inference-foundations/README.md) remain
available as the foundational reference library.

Remote GPU sessions follow the
[Lambda GPU Instance Runbook](docs/gpu-notes/lambda-instance-runbook.md), which
covers launch, bootstrap validation, artifact preservation, and shutdown.
The [Phase 1 Checkpoint](docs/phase-1-checkpoint.md) records completed work and
the exact resumption point. The
[GPU Architecture module](curriculum/phase-01-gpu-inference-foundations/modules/01-gpu-architecture/README.md)
is the required concept-only preparation before the first model lab; its
directory contains the corresponding exercises and concept-map lab.
Deferred topics—including the on-device inference and quantization tracks—are
kept in the [Learning and Engineering Backlog](docs/backlog.md).
