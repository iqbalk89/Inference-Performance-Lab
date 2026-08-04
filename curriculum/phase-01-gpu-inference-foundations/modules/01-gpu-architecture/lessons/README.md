# GPU Architecture Lessons

These lessons form the required reading for Phase 1, Module 01. They begin with
basic computing concepts and build toward interpreting GPU inference behavior.

## Reading Order

1. [Computing and Parallelism Foundations](01-computing-and-parallelism-foundations/) — includes a staged visual lab
2. [Transformer Inference Foundations](02-transformer-inference-foundations/)
3. [How Transformer Workloads Map to GPUs](03-transformer-workloads-on-gpus/)
4. [The GPU Execution Model](04-gpu-execution-model/)
5. [Compute Units and Tensor Cores](05-compute-units-and-tensor-cores/)
6. [GPU Memory Hierarchy](06-gpu-memory-hierarchy/)
7. [Why GPU Workloads Become Slow](07-performance-limiters/)
8. [CUDA Software Stack and GPU Observability](08-cuda-software-stack-and-observability/)

## Study Method

For each lesson:

1. Read once without trying to memorize every term.
2. Redraw its main diagrams without looking.
3. Explain the lesson aloud in plain language.
4. Complete its knowledge check without notes.
5. Complete its lesson-local lab when present.
6. Review only the sections behind missed answers.
7. Record remaining questions in `docs/learning-journal.md`.

After Lesson 08, return to the [module overview](../README.md), complete the
[integration exercises](../exercises.md), and then complete the
[concept-map lab](../lab.md).

## Scope

These lessons build a conceptual and diagnostic foundation. They do not require
a cloud GPU and intentionally defer CUDA programming, occupancy calculations,
PTX/SASS, and advanced kernel optimization.
