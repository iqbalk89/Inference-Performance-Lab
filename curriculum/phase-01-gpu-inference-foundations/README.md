# Phase 1 — GPU Foundations and Inference Fundamentals

**Timeline:** Weeks 1–5

**Status:** In progress

## Objective

Understand how a decoder-only language model generates tokens while becoming
comfortable executing, observing, and measuring inference on an NVIDIA GPU.

## Environment Split

- **Intel Mac:** study, source development, tests, Docker, documentation, trace
  review, and orchestration
- **Lambda A10:** CUDA-enabled PyTorch, model execution, GPU measurements, and
  NVIDIA profiling

Use the [Lambda runbook](../../docs/gpu-notes/lambda-instance-runbook.md) for
every paid GPU session. Never install CUDA or NVIDIA drivers on the Intel Mac.

## Module Sequence

| Module | Learn | Build | Status |
| --- | --- | --- | --- |
| [01 — GPU Architecture](modules/01-gpu-architecture/README.md) | Execution, memory, CUDA stack, observation | Concept map only | In progress |
| [02 — First Model and Profiling](modules/02-first-model-and-profiling/README.md) | Model execution and profiling workflow | First GPU model and traces | Not started |
| [03 — Inference Mechanics](modules/03-inference-mechanics/README.md) | Tokenization, attention, KV cache, prefill/decode, sampling | Instrumented generation experiment | Not started |
| [04 — Minimal Inference Service](modules/04-minimal-inference-service/README.md) | Serving boundary, API behavior, timing | HTTP generation service | Not started |
| [05 — Benchmark and Report](modules/05-benchmark-and-report/README.md) | Experimental design and metric interpretation | Benchmark matrix and report | Not started |

Complete modules in order. Module 01 is concept-only and does not require a
cloud instance.

## Phase Deliverables

- [x] Accepted remote NVIDIA environment
- [x] CUDA-enabled PyTorch verification
- [x] Initial synthetic Nsight Systems and Nsight Compute traces
- [ ] GPU architecture knowledge check
- [ ] First decoder-only model on the A10
- [ ] Model-level PyTorch Profiler trace
- [ ] Model-level Nsight Systems trace
- [ ] Instrumented prefill/decode and KV-cache experiment
- [ ] Minimal inference HTTP service with tests
- [ ] Repeatable latency, throughput, and memory benchmarks
- [ ] Initial engineering report

## Phase Completion Gate

Phase 1 is complete only when the learner can explain and demonstrate:

- The path from input text to generated tokens
- Why prefill and decode have different performance behavior
- What the KV cache stores and how its memory grows
- TTFT, TPOT, end-to-end latency, throughput, and peak memory
- How weights, activations, temporary tensors, and KV cache use GPU memory
- How `nvidia-smi`, PyTorch Profiler, Nsight Systems, and Nsight Compute answer
  different questions
- How the benchmark and report can be reproduced on a fresh instance

## Supporting Records

- [Current checkpoint](../../docs/phase-1-checkpoint.md)
- [A10 acceptance evidence](../../docs/gpu-notes/lambda-a10-acceptance.md)
- [Learning journal](../../docs/learning-journal.md)
- [Deferred specializations](../../docs/backlog.md)
