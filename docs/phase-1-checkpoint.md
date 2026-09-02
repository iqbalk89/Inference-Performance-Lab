# Phase 1 Checkpoint

**Checkpoint date:** September 1, 2026

## Current Position

Phase 1 Part A is in progress. The remote GPU platform and profiling access are
accepted, and the first model-level inference experiment is scaffolded. The next
remote session should run the Week 1 direct PyTorch baseline—not repeat provider
research or environment troubleshooting.

The canonical resumption path is:

1. [Module 01 — GPU Architecture](../curriculum/phase-01-gpu-inference-foundations/modules/01-gpu-architecture/README.md)
2. [Module 01 exercises](../curriculum/phase-01-gpu-inference-foundations/modules/01-gpu-architecture/exercises.md)
3. [Module 01 concept-map lab](../curriculum/phase-01-gpu-inference-foundations/modules/01-gpu-architecture/lab.md)
4. [Module 02 — First Model and Profiling](../curriculum/phase-01-gpu-inference-foundations/modules/02-first-model-and-profiling/README.md)
5. [Agent workload performance roadmap](agent-prefix-performance-roadmap.md)
6. [Week 1 experiment](../experiments/agent-prefix-performance/README.md)

## Completed

- Selected Lambda On-Demand Cloud for initial GPU work.
- Provisioned and connected to a one-GPU NVIDIA A10 instance.
- Verified an NVIDIA A10 with 23,028 MiB usable VRAM.
- Verified CUDA-enabled PyTorch 2.7.0 using CUDA 12.8.
- Ran a synthetic CUDA matrix-multiplication workload.
- Verified Nsight Systems CUDA, NVTX, OS runtime, GPU Metrics, CPU sampling,
  call stacks, and scheduling-event collection.
- Verified Nsight Compute hardware-counter access without
  `ERR_NVGPUCTRPERM`.
- Resolved Lambda's broken Nsight Systems report importer by installing the
  official NVIDIA `nsight-systems-cli` package.
- Added and tested an idempotent GPU bootstrap with automated validation.
- Documented the launch, connection, artifact, and termination workflow.

Evidence is recorded in
[Lambda A10 Environment Acceptance](gpu-notes/lambda-a10-acceptance.md). The
repeatable operating procedure is in the
[Lambda GPU Instance Runbook](gpu-notes/lambda-instance-runbook.md).

## Resume Here

### Next Learning Block

Before the next GPU session, complete the
[GPU Architecture module](../curriculum/phase-01-gpu-inference-foundations/modules/01-gpu-architecture/README.md). It
provides the minimum reading set, time budgets, required notes, and completion
questions for these introductory concepts:

1. GPU versus CPU parallelism
2. Streaming Multiprocessors, CUDA cores, and Tensor Cores
3. GPU memory hierarchy and memory bandwidth
4. VRAM use by weights, activations, and temporary tensors
5. CUDA runtime versus the NVIDIA driver
6. How to read utilization and memory fields in `nvidia-smi`

The goal is to explain their roles, not to write CUDA kernels.

### Next Build Block — First Model

1. Follow the Week 1 experiment instructions and preserve Lambda Stack's
   CUDA-enabled PyTorch in a `--system-site-packages` virtual environment.
2. Run the discovery measurement for `Qwen/Qwen2.5-1.5B-Instruct`, record its
   resolved commit hash, and pin that hash for subsequent measurements.
3. Run the 512/2,048 prompt-token by 32/128 output-token matrix.
4. Observe `nvidia-smi` during execution and compare its memory view with
   PyTorch allocated and reserved memory.
5. Summarize prefill, cached decode, full recomputation, tensor/cache shapes,
   and peak memory in `experiments/agent-prefix-performance/week1-findings.md`.
6. Use the results to state focused profiling questions for Week 2.

### After the First Model

- Trace one request from text through tokenization, embeddings, transformer
  blocks, logits, sampling, and decoded output.
- Separate prefill timing from decode timing.
- Observe KV-cache creation and memory growth as prompt and output lengths
  change.
- Only then begin the minimal HTTP inference service and benchmark harness.

## Decisions Still to Make

- Exact first model and pinned revision
- Transformers/PyTorch dependency versions
- Environment strategy for Python dependencies
- Initial numerical precision (`float16` or `bfloat16`, subject to A10 support
  and the selected model)
- Initial prompt/output-length matrix for controlled measurements
- Trace-viewing workflow on the Intel Mac

These should be decided deliberately in the next session and recorded before
installing a large dependency set or downloading model weights.

## Immediate Operational Reminder

Lambda instances continue billing while Running even when disconnected. Before
ending a GPU session, push wanted source and notes, copy wanted ignored
artifacts, terminate the instance in the dashboard, and confirm it is no longer
Running or Booting.

## Phase 1 Completion Gate

Do not move to Phase 2 until the repository contains:

- A reproducible model inference script
- A PyTorch Profiler trace and a model-level Nsight Systems trace
- A minimal inference service with automated tests
- Repeatable TTFT, TPOT, latency, throughput, and peak-memory measurements
- Experiments covering prompt length, output length, warm versus cold requests,
  and KV-cache memory growth
- The first engineering report with methodology, results, limitations, and
  lessons learned
