# Module 01 Lab — Concept Map

This is a paper or Markdown reasoning lab. Do not launch a GPU instance.

## Objective

Produce a compact mental model that can be used to predict and interpret the
first model profile.

## Tasks

1. Draw the path from a Python `torch` operation to GPU execution. Include
   PyTorch, CUDA libraries/runtime, driver, kernel launch, block, SM, warp, and
   execution pipelines.
2. Add the memory hierarchy: registers, shared memory/L1, L2, and VRAM.
3. Annotate where model weights and KV-cache data usually reside between
   operations.
4. Add one likely performance limiter to each scenario:
   - Tiny isolated GPU operation
   - Large prompt prefill
   - Batch-one token decoding
5. Add the observation tool that could test each prediction.

## Evidence

Record the diagram and a one-paragraph explanation in the Phase 1 section of
`docs/learning-journal.md`. List at least three uncertainties for review.

## Pass Criteria

- Relationships are directional and internally consistent.
- Memory capacity is not confused with bandwidth.
- `nvidia-smi` is not treated as a kernel-level profiler.
- The diagram distinguishes framework, software platform, driver, and hardware.
