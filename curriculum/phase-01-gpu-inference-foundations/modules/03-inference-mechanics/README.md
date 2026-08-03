# Module 03 — Inference Mechanics

**Type:** Learn and experiment

**Estimated time:** 4–6 hours

## Learning Objectives

Explain and observe:

- Tokenization and token IDs
- Decoder-only transformer data flow
- Causal self-attention
- Autoregressive generation
- KV-cache purpose, contents, and memory growth
- Prefill versus decode
- Greedy, temperature, top-k, and top-p sampling
- How prompt length and output length affect time and memory

## Lesson Sequence

1. **Tokenization:** text becomes token IDs; token counts, not character counts,
   drive sequence computation.
2. **Decoder-only flow:** token embedding → repeated transformer blocks → final
   normalization/output projection → logits.
3. **Causal attention:** each position may attend only to allowed current and
   previous positions.
4. **Autoregression:** select one token, append it, and repeat until a stop
   condition.
5. **KV cache:** retain prior-layer keys and values so decode does not recompute
   them for every earlier token.
6. **Prefill/decode:** process all prompt tokens to initialize state, then
   generate one or more new tokens iteratively.
7. **Sampling:** transform logits into a token-selection policy.

## Minimum Resources

1. [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer): focus on encode, decode, special tokens, truncation, and attention masks. **20 minutes.**
2. [Hugging Face generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies): read greedy decoding and sampling controls; skip beam search and custom generation methods. **25 minutes.**
3. [Hugging Face cache strategies](https://huggingface.co/docs/transformers/kv_cache): read the introduction, default dynamic cache, and cache-disabled example. Skip offloading, compilation, quantized caches, and model-specific caches. **25 minutes.**
4. The selected model's official model card: read architecture, context length,
   tokenizer, generation recommendations, license, and known limitations.
   **15–20 minutes.**

These resources explain the library surface. The explicit loop in the lab is
required to connect that surface to prefill and decode mechanics.

## Required Work

1. Complete [exercises.md](exercises.md).
2. Complete [lab.md](lab.md).
3. Add diagrams and conclusions to the learning journal.

## Completion Gate

- Trace one request from text to returned text.
- Explain why KV caching reduces computation but consumes memory.
- Predict how prompt and output length separately affect TTFT and decode time.
- Explain when deterministic versus stochastic generation is appropriate.
