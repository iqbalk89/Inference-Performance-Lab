# Module 01 — GPU Architecture and Performance

**Type:** Concept-only

**Time:** 2–3 hours

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

## Minimum Resources

Read only the assigned sections.

1. [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html): Sections 1–4. **45–60 minutes.**
2. [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html): kernels, threads/blocks, warps/SIMT, memory hierarchy, and heterogeneous programming. Skip clusters and kernel implementation. **25–35 minutes.**
3. [CUDA Compatibility introduction](https://docs.nvidia.com/deploy/cuda-compatibility/latest/index.html) and the explanatory text above NVIDIA's [driver/toolkit matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html). Do not memorize version tables. **10–15 minutes.**
4. [`nvidia-smi` documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html): search for `FB Memory Usage`, `Utilization`, `Processes`, and `--loop`. **10–15 minutes.**
5. [PyTorch CUDA asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution) and the introduction/options in the [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). Do not run the example yet. **20–25 minutes.**
6. [Nsight Systems basic CUDA trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace): read only the introduction. **10–15 minutes.**

## Lessons

### 1. CPU and GPU Roles

CPUs emphasize flexible, latency-sensitive execution. GPUs expose much more
parallel arithmetic throughput but need sufficient parallel work. Transformer
linear layers and attention create large matrix operations, while a single
decode step offers less parallel work than prompt processing or batching.

### 2. Execution Hierarchy

```text
GPU
└── Streaming Multiprocessors
    ├── Warp schedulers
    ├── Arithmetic execution pipelines
    ├── Tensor Cores
    ├── Registers
    └── Shared memory / L1
```

A kernel launch creates a grid of thread blocks. Blocks are scheduled onto SMs,
and threads execute in groups of 32 called warps. Divergent branches within a
warp can require multiple paths to execute.

### 3. CUDA Execution Units and Tensor Cores

General CUDA arithmetic pipelines execute ordinary arithmetic instructions.
Tensor Cores accelerate supported matrix multiply-accumulate operations for
specific types and shapes. A workload running on CUDA does not automatically
use Tensor Cores efficiently.

### 4. Memory Hierarchy

Registers and shared memory are small and close to execution. L2 is shared
across the GPU. VRAM is much larger but farther from execution. Model weights,
activations, KV cache, input/output tensors, and temporary workspaces consume
VRAM.

### 5. Performance Limits

- Too little parallel work can be latency-limited.
- Enough work with high operations per byte can be compute-limited.
- Enough work with low operations per byte can be memory-bandwidth-limited.

Prefill typically creates large parallel operations. Single-request decoding
repeatedly reads weights and KV-cache data for one token and is often sensitive
to memory bandwidth.

### 6. Software and Observation Stack

```text
Model / inference engine
        ↓
PyTorch and CUDA libraries
        ↓
CUDA runtime and driver APIs
        ↓
NVIDIA driver
        ↓
GPU
```

`nvidia-smi` reports the maximum CUDA version supported by the driver, not
necessarily the CUDA version used to build PyTorch.

| Tool | Primary question |
| --- | --- |
| `nvidia-smi` | Is the device active, what memory is allocated, and which processes exist? |
| PyTorch Profiler | Which framework operations consume time and memory? |
| Nsight Systems | How do CPU activity, CUDA calls, transfers, kernels, and idle gaps interact? |
| Nsight Compute | Why does a specific GPU kernel perform as it does? |

## Required Work

1. Complete [exercises.md](exercises.md).
2. Complete the concept-map [lab.md](lab.md).
3. Record unresolved questions in `docs/learning-journal.md`.

## Completion Gate

- Score at least 8/10 on the recall check without notes.
- Draw the execution hierarchy and software stack from memory.
- Correctly explain capacity versus bandwidth and prefill versus decode at a
  high level.
- No paid GPU instance is required.
