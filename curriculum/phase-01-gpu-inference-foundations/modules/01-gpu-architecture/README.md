# Module 01 — GPU Architecture and Performance

**Type:** Concept-only

**Time:** 16–20 hours, including exercises and concept-map lab

**Cloud GPU required:** No

## Learning Objectives

Explain:

- CPU versus GPU execution
- Kernels, grids, blocks, threads, warps, and Streaming Multiprocessors
- CUDA execution units versus Tensor Cores
- Registers, shared memory/L1, L2, and VRAM
- Capacity, bandwidth, latency, and arithmetic intensity
- Latency-, compute-, and memory-limited workloads
- NVIDIA driver, CUDA runtime, PyTorch, and GPU responsibilities
- The distinct roles of `nvidia-smi` and the three profilers

## Required Lesson Sequence

Read these repository-authored lessons in order. Each assumes only the material
introduced by earlier lessons.

| Lesson | Subject | Expected study time |
| --- | --- | ---: |
| [01](lessons/01-computing-and-parallelism-foundations.md) | Computing, execution order, CPU/GPU tradeoffs, parallelism, matrices, tensors, and transformer motivation | 4–5 hr |
| [02](lessons/02-gpu-execution-model.md) | Host/device execution, kernels, grids, blocks, threads, warps, SMs, divergence, and latency hiding | 2–3 hr |
| [03](lessons/03-compute-units-and-tensor-cores.md) | Arithmetic pipelines, CUDA-core terminology, matrix multiplication, Tensor Cores, precision, and FLOPS | 90–120 min |
| [04](lessons/04-gpu-memory-hierarchy.md) | Bits and bytes, registers, caches, VRAM, bandwidth, transfers, allocation, and inference memory estimates | 90–120 min |
| [05](lessons/05-performance-limiters.md) | Latency, compute and memory limits, arithmetic intensity, prefill, decode, and batching | 75–90 min |
| [06](lessons/06-cuda-software-stack-and-observability.md) | PyTorch, CUDA, drivers, asynchronous execution, streams, `nvidia-smi`, and profilers | 75–90 min |

Use the [lesson index](lessons/README.md) to resume. Do not read all six in one
sitting. Complete the knowledge check at the end of each lesson before moving
on.

## Supplemental Primary References

The lessons are the required material. Consult these official sources when a
lesson instructs you to verify a definition or when you want the authoritative
version:

1. [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html), Sections 1–4
2. [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
3. [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/index.html)
4. [`nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
5. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
6. [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
7. [Nsight Systems CUDA trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace)

## Required Work

1. Read all six lessons and complete their embedded knowledge checks.
2. Complete [exercises.md](exercises.md) as a closed-notes integration check.
3. Complete the concept-map [lab.md](lab.md).
4. Record unresolved questions in `docs/learning-journal.md`.

## Completion Gate

- Score at least 8/10 on the recall check without notes.
- Draw the execution hierarchy and software stack from memory.
- Correctly explain capacity versus bandwidth and prefill versus decode at a
  high level.
- No paid GPU instance is required.
