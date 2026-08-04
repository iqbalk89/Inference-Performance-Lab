# Lesson 01 Lab — Visual Foundations of Inference

**Time:** 90–150 minutes  
**Hardware:** Any modern browser; no NVIDIA GPU required  
**Deliverable:** Completed [`worksheet.md`](worksheet.md) plus three screenshots

## Purpose

This lab turns Lesson 01's abstractions into objects you can manipulate. It is
a teaching simulation, not a GPU profiler or performance benchmark. Its timing,
worker, and matrix dimensions are deliberately small so dependencies and shapes
remain visible.

By the end, you should be able to explain visually:

1. Why dependencies constrain execution order.
2. Why independent work creates scheduling opportunities but does not imply
   unlimited simultaneous execution.
3. How tensor rank, axes, shape, indices, value count, dtype, and storage relate.
4. How one matrix-output cell is calculated from an input row and weight column.
5. How a large logical output is divided over finite worker slots and waves.
6. How Q/K scores create weights and how those weights mix V rows.
7. Why a causal mask prevents future information from influencing earlier rows.
8. What prefill saves and how the KV cache grows during decode.
9. Why transfer and launch overhead can make a tiny GPU task slower overall.

## Start the Lab

From this directory on macOS:

```bash
open index.html
```

Or open [`index.html`](index.html) from VS Code and choose **Open in Default
Browser**. The file runs locally and makes no network requests.

Keep [`worksheet.md`](worksheet.md) open beside it. Do not merely click through;
predict each result before revealing or changing it.

## Required Route

Complete the seven stations in order:

1. **Dependency scheduler** — predict which operations are ready.
2. **Tensor explorer** — construct and index a `[batch, tokens, features]` tensor.
3. **Matrix microscope** — select output cells and verify dot products.
4. **Finite GPU waves** — separate logical parallel work from physical capacity.
5. **Attention and causal mask** — compare Q/K, then mix V.
6. **Prefill/decode timeline** — watch the known boundary and KV cache grow.
7. **End-to-end cost model** — determine when offloading does and does not help.

## Evidence to Capture

Save screenshots in a local `evidence/` directory (create it if needed):

- `tensor-index.png`: tensor explorer with a nonzero batch, token, and feature
  index selected.
- `masked-attention.png`: query position 1 with future positions visibly masked.
- `kv-cache-growth.png`: after at least two decode steps.

Screenshots are evidence of interaction, not proof of understanding. Your
written explanations are the primary deliverable.

## Completion Standard

The lab is complete when:

- Every worksheet prediction and explanation is filled in.
- Matrix arithmetic is correct without relying on the displayed answer.
- You can distinguish logical workers from physical execution resources.
- You can explain `known to the server` versus `visible to an attention row`.
- You can explain the KV cache without calling it a cache of words or answers.
- You can identify the complete measured boundary in the offload-cost station.

Return to the [Lesson 01 reading](../README.md) for any missed concept. Do not
move to Lesson 02 until you can complete the final teach-back without notes.

