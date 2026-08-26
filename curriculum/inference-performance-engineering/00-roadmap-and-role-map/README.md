# Module 00 — Roadmap and Role Map

## Why This Track Fits Your Background

ASIC development provides a strong base for reasoning about data movement,
bandwidth, pipelines, parallelism, and resource constraints. Agent
orchestration provides experience with queues, scheduling, concurrency,
reliability, and end-to-end systems. The bridge to inference performance is:

- practical Python and PyTorch fluency;
- transformer workload shapes and dataflow;
- quantitative latency, memory, capacity, and cost models;
- disciplined GPU measurement and profiling;
- evidence-based optimization experiments.

## Capability Ladder

| Level | You can… | Evidence |
| --- | --- | --- |
| 1. Describe | trace a request and name the major resources | labeled pipeline diagram |
| 2. Estimate | calculate FLOPs, bytes, memory, and idealized latency | executable model with units |
| 3. Measure | collect synchronized, warmed-up, repeatable results | benchmark dataset |
| 4. Diagnose | locate the responsible phase, operator, and resource | profiler-backed report |
| 5. Optimize | change one mechanism and explain the metric movement | controlled before/after study |
| 6. Design | predict fleet capacity and architectural tradeoffs | validated system model |

The curriculum should move you up this ladder repeatedly. It should not require
memorizing every GPU component before you have a workload to reason about.

The capability ladder is implemented through the
[Inference System Performance Workbench](../../../docs/architecture/inference-system-performance-workbench.md).
Its plan maps the visual hardware model, inference phases, profiling evidence,
serving dynamics, correctness, and distributed extensions to the target roles.

## Role Boundaries

- **Performance modeling:** predicts latency, memory, throughput, capacity, and
  cost from workload and hardware properties.
- **Performance profiling:** gathers evidence to explain where actual execution
  differs from the model.
- **Performance optimization:** changes algorithms, kernels, precision,
  scheduling, or system design to improve an explicit metric.
- **Kernel engineering:** implements and tunes low-level CUDA or Triton kernels.
  It becomes valuable after the first three capabilities are established.
