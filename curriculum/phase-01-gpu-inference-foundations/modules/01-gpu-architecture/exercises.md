# Module 01 Exercises

Answer in your own words. Short answers are preferable to copied definitions.

## CPU and GPU Roles

1. Why can a GPU outperform a CPU on transformer matrix multiplication?
2. Give one workload for which a CPU may be preferable and explain why.
3. Why does merely moving code to CUDA not guarantee a speedup?

## Execution Hierarchy

4. Put these in a coherent execution relationship: block, grid, kernel, SM,
   thread, warp.
5. What does an SM provide that an individual arithmetic execution unit does
   not?
6. A warp's threads take two different conditional branches. What performance
   issue may occur, and why?

## CUDA Execution Units and Tensor Cores

7. How are general arithmetic pipelines and Tensor Cores functionally
   different?
8. Why might a matrix operation fail to achieve expected Tensor Core
   performance?

## Memory

9. List four categories of data that consume VRAM during inference.
10. A model fits in 24 GB VRAM but generates tokens slowly. Explain why this
    does not contradict the fact that it fits.
11. Define capacity, bandwidth, and latency using three different sentences.
12. Why is keeping frequently reused data closer to execution beneficial?

## Performance Limits

13. Describe latency-limited, compute-limited, and memory-limited execution.
14. What does arithmetic intensity compare?
15. Why is single-request decode often more memory-bandwidth-sensitive than
    prefill?
16. How can batching change GPU utilization and request latency?

## Software Stack and Tools

17. Explain the separate responsibilities of PyTorch, the CUDA runtime, the
    NVIDIA driver, and the GPU.
18. Why can `nvidia-smi` display CUDA 13.0 while PyTorch reports CUDA 12.8?
19. What can high GPU utilization in `nvidia-smi` tell you, and what can it not
    prove?
20. Choose the first tool for each question and justify it:
    - Which PyTorch operator allocated the most tensor memory?
    - Where is the CPU waiting between GPU kernel launches?
    - Which process currently owns 8 GB of VRAM?
    - Why is one matrix-multiplication kernel underperforming?

## Recall Check — Closed Notes

Answer these after a break without reading resources:

1. Why are GPUs effective for transformer inference?
2. What does an SM do?
3. How do CUDA execution units and Tensor Cores differ?
4. What is a warp?
5. What occupies VRAM during inference?
6. How do capacity, bandwidth, and latency differ?
7. What makes work latency-, compute-, or memory-limited?
8. Why is prefill generally more parallel than single-request decode?
9. How do PyTorch, CUDA, the driver, and the GPU differ?
10. When would you use each GPU observation/profiling tool?
