# Module 02 — End-to-End Inference Pipeline

## Performance Question

When a user sends a prompt, where does the time go before and between the
returned tokens?

## Outcome

You will be able to trace one request, assign every important interval to a
phase, write an additive latency model, distinguish TTFT from inter-token
latency, and collect a first predicted-versus-measured phase breakdown.

## Sequence

1. Read [Lesson 1 — Request to Performance Equation](lessons/01-request-to-performance-equation/README.md).
2. Complete [Hands-On Exercise 01 — Batches and Tensor Shapes](../practice/hands-on-learning-path/exercises/01-batches-and-tensor-shapes/README.md)
   after the lesson's batch subsection.
3. Complete the lesson's knowledge check before opening the answers.
4. On the Mac, inspect and dry-run the [lab](lab/README.md) with a small model.
5. On the remote NVIDIA GPU, benchmark and capture the annotated timeline.
6. Complete the [report template](lab/report-template.md).

## Completion Gate

You can proceed when you can:

- identify what exists before and after tokenization;
- explain exactly where prefill ends and iterative decode begins;
- write TTFT, TPOT, generation latency, and E2E latency with units;
- predict which phase changes with prompt length and which changes with output
  length;
- show profiler evidence for each measured GPU phase;
- explain at least two reasons the additive model differs from wall-clock time.
