# Inference Performance Engineering

This is the primary, job-oriented curriculum for learning to **model, measure,
explain, and improve** LLM inference performance. The earlier
[Phase 1 curriculum](../phase-01-gpu-inference-foundations/README.md) remains a
reference library; it is no longer the only linear path through the material.

## Target Role

The first target is an inference performance modeling and profiling engineer:
someone who can turn a model workload into a latency, memory, capacity, and cost
prediction; collect evidence on NVIDIA GPUs; explain prediction errors; and
recommend an optimization. Low-level CUDA kernel expertise is a later
specialization, not an entry prerequisite.

## The Engineering Loop

Every module uses the same loop:

```text
understand the workload
        ↓
make a quantitative prediction
        ↓
measure on real hardware
        ↓
profile the prediction error
        ↓
identify the bottleneck
        ↓
change one variable and remeasure
```

A profiler screenshot without a prediction is observation, not yet performance
engineering. A formula without measurement is a hypothesis, not yet a model.

## Learning Sequence

| Module | Central question | Required artifact | Status |
| --- | --- | --- | --- |
| [00 — Roadmap and Role Map](00-roadmap-and-role-map/README.md) | What capabilities do the target jobs require? | Personal capability map | Ready |
| [01 — Measurement and Modeling Foundations](01-measurement-and-modeling-foundations/README.md) | Can I estimate work and measure time correctly? | Hardware calibration report | Planned |
| [02 — E2E Inference Pipeline](02-e2e-inference-pipeline/README.md) | Where does request latency come from? | Predicted-versus-measured phase report | **Start here** |
| [03 — Prefill Performance](03-prefill-performance/README.md) | How does prompt processing scale? | Prefill latency model | Planned |
| [04 — Decode and KV Performance](04-decode-and-kv-performance/README.md) | What limits each generated-token step? | Decode and KV model | Planned |
| [05 — Profiling and Root Cause Analysis](05-profiling-and-root-cause-analysis/README.md) | Which evidence proves the bottleneck? | Root-cause report | Planned |
| [06 — Serving Capacity and Cost](06-serving-capacity-and-cost/README.md) | How do requests become fleet cost and SLA risk? | Capacity simulator | Planned |
| [07 — Optimization Experiments](07-optimization-experiments/README.md) | Which change improves the target metric, and why? | Optimization case studies | Planned |
| [08 — Distributed Inference](08-distributed-inference/README.md) | When do communication and synchronization dominate? | Multi-GPU performance model | Planned |

Module 02 is intentionally available before Module 01 is fully authored. It
provides the map of the whole system. Complete its conceptual lesson now; then
return to Module 01 for the measurement prerequisites before running the GPU
lab.

## Standard for Every Lesson

Each completed lesson will contain:

1. One clearly stated performance question
2. A labeled dataflow visual
3. A tensor-shape or state ledger where applicable
4. A performance equation with units
5. One worked numerical example
6. A prediction made before measurement
7. Profiler evidence
8. Reconciliation of prediction and observation
9. A short knowledge check with answers

## Portfolio Outcomes

The track culminates in four interview-ready projects:

1. A calibrated single-GPU inference performance model
2. A profiler-guided bottleneck and root-cause report
3. A before/after optimization case study
4. A serving capacity and cost model with SLA tradeoffs

## Coding Practice

Use the [Inference-Adjacent LeetCode Practice](practice/leetcode-inference-adjacent/README.md)
as a supplement. Its minimum sequence connects matrix indexing, sparse
representations, sampling, and caching to the relevant performance questions.
It is intentionally smaller than a general-purpose interview grind.
