# Phase 1 GPU Foundations Learning Guide

## Purpose

Complete this concept-only guide before launching another paid GPU instance or
starting the first model lab. The goal is a usable mental model of GPU
execution—not CUDA kernel-programming proficiency.

**Expected time:** 2–3 hours across two study sessions

## Minimum Required Resources

The required set is intentionally small and uses primary documentation. Read
only the assigned sections; the full CUDA manuals are much larger than this
phase requires.

### Resource A — NVIDIA GPU Performance Background

[GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)

Read:

- Section 1, Overview
- Section 2, GPU Architecture Fundamentals
- Section 3, GPU Execution Model
- Section 4, Understanding Performance

This is the main resource. It covers SMs, execution pipelines, the GPU memory
hierarchy, parallel execution, arithmetic intensity, and latency-, math-, and
memory-limited workloads.

**Budget:** 45–60 minutes

### Resource B — CUDA Programming Model

[CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

Read only:

- Kernels, threads, and thread blocks
- Warps and SIMT
- Memory hierarchy
- Heterogeneous programming

Stop before advanced clusters, distributed shared memory, asynchronous SIMT,
and kernel-writing details. For this phase, understand how work is organized;
do not try to write CUDA.

**Budget:** 25–35 minutes

### Resource C — CUDA Software Compatibility

Read:

- [CUDA Compatibility introduction](https://docs.nvidia.com/deploy/cuda-compatibility/latest/index.html)
- The short explanatory text above the architecture table in the
  [CUDA Toolkit, Driver, and Architecture Matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)

Focus on the distinction among the GPU, NVIDIA driver, CUDA toolkit/runtime,
and the CUDA version shown by `nvidia-smi`. Do not memorize compatibility
tables or driver version numbers.

**Budget:** 10–15 minutes

### Resource D — NVIDIA System Management Interface

[NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

Use the page search function and read only:

- `FB Memory Usage`
- `Utilization`
- `Processes`
- `--loop`

Learn what the displayed GPU utilization and memory-utilization percentages
actually measure. They are sampled activity indicators, not direct statements
that an application is optimally using every CUDA or Tensor Core.

**Budget:** 10–15 minutes

### Resource E — PyTorch CUDA Execution and Profiler

Read:

- `Asynchronous execution` in
  [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)
- The introduction and profiler-options explanation in the
  [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

Stop before reproducing the sample program. Understand why naive wall-clock
timing can be wrong, and what CPU activity, CUDA activity, shapes, stacks, and
memory recording mean.

**Budget:** 20–25 minutes

### Resource F — Nsight Systems Timeline

[Nsight Systems User Guide: CUDA Trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace)

Read only the `Basic CUDA trace` introduction. Learn the difference among:

- A CPU-side CUDA API call
- A GPU kernel
- A host-to-device or device-to-host memory transfer
- A CUDA stream

Do not read the complete Nsight manual yet. Nsight Compute and detailed kernel
metrics belong after the first model trace.

**Budget:** 10–15 minutes

## Session 1 — Architecture and Performance

**Duration:** 90–120 minutes

### 1. CPU and GPU Roles

Use Resource A to understand:

- CPUs optimize flexible, latency-sensitive execution.
- GPUs expose far more parallel arithmetic throughput.
- A workload needs enough parallel work to use a GPU effectively.
- Data movement and launch overhead can erase the benefit for small workloads.

Connect this to transformers: linear layers and attention contain large matrix
operations, but individual token-by-token decode steps expose less parallelism
than processing many prompt tokens or requests.

### 2. GPU Execution Hierarchy

Use Resources A and B to build this model:

```text
GPU
└── Streaming Multiprocessors (SMs)
    ├── Warp schedulers
    ├── Arithmetic execution pipelines
    ├── Tensor Cores
    ├── Registers
    └── Shared memory / L1
```

Be able to relate a kernel, grid, block, thread, warp, and SM. A warp contains
32 threads in the CUDA programming model. Understand branch divergence at a
high level, but do not study occupancy tuning yet.

### 3. CUDA Cores and Tensor Cores

Use Resource A. Learn the functional difference:

- CUDA execution units handle general arithmetic instructions used by CUDA
  programs.
- Tensor Cores accelerate supported matrix multiply-accumulate operations for
  particular data types and shapes.
- Using PyTorch on a GPU does not guarantee that every operation uses Tensor
  Cores.

Do not memorize core counts or theoretical peak FLOPS.

### 4. Memory Hierarchy

Use Resources A and B to understand registers, shared memory/L1, L2, and device
DRAM/VRAM. Distinguish:

- **Capacity:** how many bytes fit.
- **Bandwidth:** how many bytes can move per unit time.
- **Latency:** how long one operation or transfer takes to begin and complete.

Relate VRAM consumption to weights, activations, KV cache, input/output tensors,
and temporary kernel workspaces.

### 5. Performance Limiters

Use Resource A, Section 4. Understand—not calculate in detail—arithmetic
intensity as useful math operations per byte moved.

- Insufficient parallelism can make a workload latency-limited.
- High arithmetic intensity can make it math/compute-limited.
- Low arithmetic intensity can make it memory-bandwidth-limited.

Connect this to inference: prefill generally creates larger parallel operations,
while single-request decode repeatedly reads weights and KV-cache data for one
new token and is often memory-bandwidth-sensitive.

## Session 2 — Software Stack and Observation

**Duration:** 60–75 minutes

### 1. Software Stack

Use Resource C to understand:

```text
Model code / inference engine
            ↓
PyTorch and CUDA libraries
            ↓
CUDA runtime and driver APIs
            ↓
NVIDIA driver
            ↓
GPU hardware
```

Know that `nvidia-smi` reports the maximum CUDA version supported by the
installed driver; that number is not necessarily the CUDA version against which
PyTorch was built.

### 2. Operational Observation

Use Resource D. Learn what `nvidia-smi` can tell you about device identity,
VRAM allocation, sampled GPU activity, sampled memory activity, and active
processes. Also learn what it cannot prove: a high-level utilization percentage
does not identify the responsible operators, kernels, stalls, or bottleneck.

### 3. Profiling Layers

Use Resources E and F to distinguish:

| Tool | Primary question |
| --- | --- |
| `nvidia-smi` | Is the GPU active, how much memory is allocated, and which processes are present? |
| PyTorch Profiler | Which framework operations consume time and memory? |
| Nsight Systems | How do CPU work, CUDA calls, transfers, kernels, and idle gaps interact over time? |
| Nsight Compute | Why does an individual GPU kernel perform as it does? |

Understand CUDA's asynchronous execution before interpreting Python timers or
profiler timelines.

## Required Notes

Add your own answers under a new Phase 1 entry in `docs/learning-journal.md`:

1. Why are GPUs effective for transformer inference?
2. What does an SM do?
3. How do CUDA execution units and Tensor Cores differ?
4. What is a warp, and why can divergent branches reduce efficiency?
5. What occupies VRAM during inference?
6. How do capacity, bandwidth, and latency differ?
7. What makes work latency-, compute-, or memory-limited?
8. Why is prefill generally more parallel than single-request decode?
9. How do PyTorch, the CUDA runtime, the NVIDIA driver, and the GPU differ?
10. When would you use `nvidia-smi`, PyTorch Profiler, Nsight Systems, or Nsight
    Compute?

Also record at least three questions or points that remain unclear. Do not copy
documentation definitions verbatim; explain each idea in your own words.

## Completion Gate

Proceed to the first-model lab when you can answer at least 8 of the 10
questions accurately without reading your notes, and can draw both the GPU
execution hierarchy and software stack from memory. Imperfect terminology is
acceptable; unresolved conceptual contradictions are not.

## Not Required Yet

- Writing CUDA kernels
- Calculating occupancy
- Reading SASS or PTX
- Roofline calculations
- Nsight Compute metric analysis
- Multi-GPU communication
- Kernel fusion implementation
- Memorizing GPU specifications

These topics are intentionally deferred so the first learning block remains
small enough to finish.
