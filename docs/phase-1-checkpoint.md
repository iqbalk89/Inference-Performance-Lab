# Phase 1 Checkpoint

**Checkpoint date:** August 1, 2026

## Current Position

Phase 1 Part A is in progress. The remote GPU platform and profiling access are
accepted, but model-level inference work has not started yet. The next session
should begin with the first small decoder-only model—not more provider research
or environment troubleshooting.

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

Before or alongside the next GPU session, study these concepts at an
introductory level:

1. GPU versus CPU parallelism
2. Streaming Multiprocessors, CUDA cores, and Tensor Cores
3. GPU memory hierarchy and memory bandwidth
4. VRAM use by weights, activations, and temporary tensors
5. CUDA runtime versus the NVIDIA driver
6. How to read utilization and memory fields in `nvidia-smi`

The goal is to explain their roles, not to write CUDA kernels.

### Next Build Block — First Model

1. Select one small decoder-only Hugging Face model that fits comfortably in
   24 GB VRAM. Prefer direct PyTorch/Transformers execution for this learning
   step so tokenization, model loading, generation, and profiling remain
   visible. Do not begin with an opaque desktop wrapper.
2. Define and pin the remote Python dependencies. Decide whether to use Lambda
   Stack's system PyTorch directly or a project environment that deliberately
   preserves CUDA support.
3. Add a reproducible model-loading and generation script under `scripts/` or
   `experiments/`.
4. Record model ID, revision, dtype, device, prompt, generation settings, input
   tokens, output tokens, elapsed time, and peak allocated GPU memory.
5. Watch `nvidia-smi` during model loading and generation and record observed
   VRAM and utilization.
6. Capture the first PyTorch Profiler trace, preserving a summary in Git and
   keeping the raw trace under ignored `profile-results/`.
7. Capture a model-level Nsight Systems trace using the already validated
   workflow and compare it with the PyTorch trace.

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
