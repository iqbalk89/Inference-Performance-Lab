# Module 02 Exercises

## Model and Environment

1. What files or metadata are required to reproduce a model load?
2. Why should a model revision be pinned instead of relying only on its name?
3. What could happen if an arbitrary `pip install torch` replaces Lambda
   Stack's working CUDA build?
4. Why should downloaded weights and profiler traces stay out of ordinary Git?

## Execution

5. What does `model.eval()` change, and how is it different from
   `torch.inference_mode()`?
6. Why must input tensors and model weights reside on compatible devices?
7. What determines the memory needed just for model weights?
8. Why might FP16 and BF16 behave differently across hardware?

## Measurement

9. Why can a Python timer under-report asynchronous CUDA execution time?
10. Compare CUDA events with `torch.cuda.synchronize()` plus a wall-clock timer.
11. Why should warmup iterations be separated from measured iterations?
12. Distinguish cold model-load time from warm generation latency.

## Profiling

13. What would you expect PyTorch Profiler to name that `nvidia-smi` cannot?
14. What relationship should exist between a CPU CUDA launch call and a GPU
    kernel on an Nsight Systems timeline?
15. What does an idle GPU gap suggest? Give at least three possible causes.
16. Why should profiling results not automatically be treated as benchmark
    results?
