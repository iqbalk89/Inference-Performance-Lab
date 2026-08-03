# Module 02 — First Model and Profiling

**Type:** Learn and build

**Estimated time:** 3–5 hours

**Cloud GPU required:** Yes, only for the lab

## Learning Objectives

- Explain model weights, configuration, tokenizer files, and model revisions.
- Load a decoder-only model at an explicit dtype and device.
- Measure CUDA work correctly despite asynchronous execution.
- Distinguish warmup, cold load, and steady-state generation.
- Interpret `nvidia-smi`, PyTorch Profiler, and Nsight Systems at an introductory
  level.
- Preserve reproducibility without committing model weights or raw traces.

## Learn Before Launching

Study:

- Hugging Face model repositories, model cards, revisions, and tokenizers
- PyTorch `inference_mode`, CUDA events/synchronization, and CUDA memory APIs
- Warmup and profiler overhead
- Trace layers: framework operator → CUDA API → GPU kernel

The exact model, revision, dependency versions, and dtype must be decided and
committed before starting the paid instance.

## Minimum Resources

1. [Hugging Face model documentation](https://huggingface.co/docs/transformers/main_classes/model): read `from_pretrained` parameters for revision, dtype, and device placement. **15 minutes.**
2. [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer): read the `from_pretrained`, encode/call, and decode concepts. Skip tokenizer training. **15 minutes.**
3. [PyTorch CUDA asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution): review synchronization and CUDA-event timing. **10 minutes.**
4. [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): read through timing, memory, and trace export. **20 minutes.**
5. [Nsight Systems basic CUDA trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace): review API calls, transfers, kernels, and streams. **10 minutes.**

Stop after these sections. Distributed loading, compilation, quantization, and
advanced profiler schedules are not required for the first model.

## Required Work

1. Complete [exercises.md](exercises.md).
2. Prepare the implementation and configuration locally where possible.
3. Follow [lab.md](lab.md) on an accepted A10 instance.
4. Update the learning journal and module status.

## Completion Gate

- Reproducible generation succeeds from a fresh checkout.
- Environment and model metadata are recorded.
- Timing explicitly handles CUDA asynchrony.
- `nvidia-smi`, PyTorch Profiler, and Nsight Systems evidence exists.
- At least one observation connects model code to profiler evidence.
