# Module 03 Lab — Instrumented Generation

## Objective

Expose prefill, decode, token selection, and KV-cache behavior instead of using
only a single opaque `generate()` latency.

## Method

Build an experiment under `experiments/inference-mechanics/` that:

1. Tokenizes controlled prompts and records token counts.
2. Runs an explicit prefill forward pass with caching enabled.
3. Records time to initial logits and initial KV-cache shape/memory.
4. Executes a visible token-by-token decode loop.
5. Records per-token latency and cumulative sequence length.
6. Compares caching enabled versus disabled for a small safe case.
7. Compares greedy decoding with one stochastic configuration while keeping
   performance claims separate from output-quality observations.

## Controlled Cases

- Short prompt, short output
- Long prompt, short output
- Short prompt, long output

Use fixed token lengths where practical. Warm up before measured runs.

## Measurements

- Tokenization time
- Input and output token counts
- Prefill latency
- Per-token decode latency
- TTFT approximation and end-to-end latency
- Peak CUDA memory
- KV-cache tensor shapes or a derived byte estimate

## Analysis Questions

1. Which case had the highest TTFT, and why?
2. Which case spent the most total time decoding?
3. How did KV-cache memory change with total sequence length?
4. How much redundant work appeared when caching was disabled?
5. Did later decode tokens become slower? If so, what grew?

## Pass Criteria

- Prefill and decode are measured separately.
- The experiment exposes rather than merely names the KV cache.
- Prompt and output effects are not conflated.
- Conclusions distinguish observation from inference.
