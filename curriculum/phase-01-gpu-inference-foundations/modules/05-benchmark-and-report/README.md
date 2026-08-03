# Module 05 — Benchmark and Engineering Report

**Type:** Measure, analyze, and report

**Estimated time:** 6–10 hours

## Learning Objectives

- Define TTFT, TPOT, end-to-end latency, throughput, and peak memory precisely.
- Design controlled comparisons with warmup and repeated trials.
- Separate cold-start, warm-request, prefill-heavy, and decode-heavy behavior.
- Report distributions rather than a single convenient number.
- Identify confounders, limitations, and unsupported conclusions.
- Establish a half-precision baseline for later quantization work.

## Subsections

1. **Metric definitions and boundaries**
2. **Experimental controls and reproducibility**
3. **Prompt/output length matrix**
4. **Warm versus cold execution**
5. **Batch size and throughput introduction**
6. **KV-cache memory growth**
7. **Interpretation and engineering writing**

## Minimum Resources

1. [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html): read the introduction, timer behavior, warmups, thread considerations, and CUDA synchronization caveat. Do not reproduce every example. **25 minutes.**
2. [PyTorch CUDA asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution): review accurate timing with synchronization and CUDA events. **10 minutes.**
3. Re-read Section 4 of [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html): focus on the limits of arithmetic-intensity reasoning and the need for profiler evidence. **15 minutes.**
4. Re-read this repository's [Experiment Standard](../../../../README.md#experiment-standard) and apply it to the report outline. **5 minutes.**

Do not introduce a third-party benchmarking framework until the v1 harness has
explicit, tested metric definitions.

## Required Work

1. Complete [exercises.md](exercises.md).
2. Complete [lab.md](lab.md).
3. Publish the Phase 1 engineering report under `docs/reports/`.

## Completion Gate

- Raw machine-readable data and analysis are separate.
- Every metric has units and a documented measurement boundary.
- Repeated trials and summary statistics are present.
- Environment, model, revision, dtype, and generation settings are pinned.
- The report explains limitations and avoids causal claims unsupported by the
  experiment.
