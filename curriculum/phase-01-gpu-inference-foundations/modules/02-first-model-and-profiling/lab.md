# Module 02 Lab — First Model and Profiler Trace

## Objective

Run one pinned decoder-only model on the accepted Lambda A10 and connect model
execution to three levels of GPU observation.

## Hypothesis

Write a prediction for model-load time, weight memory, generation latency, and
whether the measured workload will fully utilize the A10. Explain the basis for
each prediction.

## Build Requirements

Create a reproducible script under `experiments/first-model/` that records:

- Model ID and revision
- Dependency versions
- GPU, driver, PyTorch, and PyTorch CUDA versions
- Dtype and device
- Prompt and generation parameters
- Input and output token counts
- Cold load time
- Warm end-to-end generation time
- Output tokens per second
- Peak allocated and reserved CUDA memory

Use deterministic generation for the baseline unless the experiment explicitly
studies sampling.

## Procedure

1. Launch an A10 and follow the Lambda runbook.
2. Run the bootstrap without full profiler acceptance unless the image changed.
3. Install only pinned project dependencies.
4. Download and run the pinned model.
5. Observe device memory and utilization with a one-second `nvidia-smi` loop.
6. Capture a short PyTorch Profiler trace after warmup.
7. Capture a short Nsight Systems trace with model-level NVTX ranges.
8. Copy valuable raw traces locally before termination.
9. Commit scripts, environment metadata, and summarized observations—not model
   weights or raw trace binaries.
10. Terminate the instance and verify billing stopped.

## Output Locations

- Workload: `experiments/first-model/`
- Reusable utilities: `src/` or `scripts/`
- Profiling procedure and summary: `profiling/first-model/`
- Raw ignored artifacts: `profile-results/`
- Reflection: `docs/learning-journal.md`

## Required Analysis

1. Did observed model memory approximately agree with the weight-size estimate?
2. How did allocated memory differ from reserved memory and `nvidia-smi` usage?
3. Which PyTorch operators dominated measured CUDA time?
4. Did Nsight show large CPU gaps, transfers, or many short kernels?
5. Was the workload plausibly compute-, memory-, or latency-limited? State the
   evidence and uncertainty.

## Pass Criteria

- A fresh command can reproduce generation.
- Measurements are labeled with units and warm/cold state.
- CUDA timing is synchronized correctly.
- Raw evidence is preserved outside Git and summarized evidence is committed.
