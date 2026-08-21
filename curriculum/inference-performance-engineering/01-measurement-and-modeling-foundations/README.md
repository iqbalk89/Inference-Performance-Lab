# Module 01 — Measurement and Modeling Foundations

**Status:** Design complete; detailed lessons pending

## Objective

Build the minimum mathematical and experimental foundation needed to make a
prediction that can be compared fairly with a GPU measurement.

## Planned Lessons

1. Units, dimensional analysis, and order-of-magnitude estimates
2. Shapes, dot products, matrix multiplication, FLOPs, and MACs
3. Bytes moved, memory bandwidth, and arithmetic intensity
4. Latency distributions, percentiles, warm-up, and synchronization
5. Python and PyTorch needed for performance experiments
6. Hardware calibration: launch overhead, bandwidth, and GEMM sweeps

## Deliverable

A hardware calibration report containing measured kernel-launch overhead,
effective memory bandwidth, representative GEMM throughput, experimental
method, variability, and limitations.

Until this module is authored, use GPU Architecture Lessons 5–7 as reference:

- [Compute Units and Tensor Cores](../../phase-01-gpu-inference-foundations/modules/01-gpu-architecture/lessons/05-compute-units-and-tensor-cores/README.md)
- [GPU Memory Hierarchy](../../phase-01-gpu-inference-foundations/modules/01-gpu-architecture/lessons/06-gpu-memory-hierarchy/README.md)
- [Performance Limiters](../../phase-01-gpu-inference-foundations/modules/01-gpu-architecture/lessons/07-performance-limiters/README.md)

Complete problems 566, 867, 1570, and 311 from the
[Inference-Adjacent LeetCode Practice](../practice/leetcode-inference-adjacent/README.md)
while studying shapes, layout, multiply-accumulate work, and sparse versus dense
representations.
