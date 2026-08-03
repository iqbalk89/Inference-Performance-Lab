# GPU Architecture Lessons

These lessons form the required reading for Phase 1, Module 01. They begin with
basic computing concepts and build toward interpreting GPU inference behavior.

## Reading Order

1. [Computing and Parallelism Foundations](01-computing-and-parallelism-foundations.md)
2. [The GPU Execution Model](02-gpu-execution-model.md)
3. [Compute Units and Tensor Cores](03-compute-units-and-tensor-cores.md)
4. [GPU Memory Hierarchy](04-gpu-memory-hierarchy.md)
5. [Why GPU Workloads Become Slow](05-performance-limiters.md)
6. [CUDA Software Stack and GPU Observability](06-cuda-software-stack-and-observability.md)

## Study Method

For each lesson:

1. Read once without trying to memorize every term.
2. Redraw its main diagrams without looking.
3. Explain the lesson aloud in plain language.
4. Complete its knowledge check without notes.
5. Review only the sections behind missed answers.
6. Record remaining questions in `docs/learning-journal.md`.

After Lesson 06, return to the [module overview](../README.md), complete the
[integration exercises](../exercises.md), and then complete the
[concept-map lab](../lab.md).

## Scope

These lessons build a conceptual and diagnostic foundation. They do not require
a cloud GPU and intentionally defer CUDA programming, occupancy calculations,
PTX/SASS, and advanced kernel optimization.
