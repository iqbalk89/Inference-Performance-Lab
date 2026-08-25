# Inference Performance Modeling Problems

## Purpose

This problem set develops the habit required for inference performance roles:
translate a model operation into **shapes, parameter counts, bytes, FLOPs, and
eventually predicted time**, then state every assumption and unit.

These problems complement:

- the [Hands-On Learning Path](../hands-on-learning-path/README.md), which uses
  implementation and measurement exercises;
- the [Inference-Adjacent LeetCode Practice](../leetcode-inference-adjacent/README.md),
  which develops general coding and data-structure fluency.

## Required Problem-Solving Method

For every problem, use this sequence:

1. Write every tensor shape.
2. Identify what each dimension means.
3. Check that matrix inner dimensions are compatible.
4. Write the symbolic formula before substituting numbers.
5. Carry units through every calculation.
6. Give exact values first, then a useful approximation.
7. Distinguish decimal units (`MB`) from binary units (`MiB`).
8. State what the simplified model includes and excludes.
9. Perform a sanity check using a second line of reasoning.
10. Open the answer key only after recording an attempt.

## Problems

| # | Problem | Primary skills | Status |
| ---: | --- | --- | --- |
| [01](problems/01-transformer-projection-cost/problem.md) | What does one Transformer projection cost? | Parameters, weight bytes, matrix-multiplication FLOPs | Ready |
| [02](problems/02-decode-vs-prefill-roofline/problem.md) | Decode vs. prefill: memory-bound or compute-bound? | Arithmetic intensity, ridge point, latency bounds, bottleneck classification | **Do next** |

## Animated Solutions

The [Manim Visual Solution Set](visual-solutions/README.md) animates the
mechanics and derivations. Attempt each written worksheet first, then render its
solution with one command. Problem 01 is available now.

Additional problems will progressively cover combined QKV projections, MLP
projections, KV-cache capacity, batching, profiler reconciliation, and
end-to-end latency.

## Recommended Order

1. Complete Problem 01 without the answer key.
2. Watch its Manim solution and explain the model boundary aloud.
3. Complete Problem 02's prediction section before calculating.
4. Solve Problem 02 with units, then use its hint ladder if needed.
5. Repeat Problem 02 using [Interview Mode](problems/02-decode-vs-prefill-roofline/interview-mode.md)
   on another day without notes.

## Units Reference

```text
1 KB  = 1,000 bytes             1 KiB = 1,024 bytes
1 MB  = 1,000,000 bytes         1 MiB = 1,048,576 bytes
1 GB  = 1,000,000,000 bytes     1 GiB = 1,073,741,824 bytes

1 MFLOP = 1,000,000 FLOPs
1 GFLOP = 1,000,000,000 FLOPs
1 TFLOP = 1,000,000,000,000 FLOPs
```

Hardware vendors commonly report bandwidth and throughput using decimal SI
units. Memory-capacity discussions sometimes use decimal labels for binary
quantities, so always state the convention used.
