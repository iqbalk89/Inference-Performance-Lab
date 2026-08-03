# Module 03 Exercises

## Tokenization

1. Why can two strings with equal character counts have different token counts?
2. What responsibilities belong to a tokenizer rather than the model?
3. Why must benchmark reports record token counts instead of only text length?

## Decoder-Only Transformers

4. Describe the path from token IDs to next-token logits.
5. What information does a token embedding represent at the model input?
6. Why is the output a distribution over the vocabulary rather than final text?

## Attention

7. What are queries, keys, and values used for conceptually?
8. What does the causal mask prevent?
9. Why does prompt attention become more expensive as sequence length grows?

## Autoregression and KV Cache

10. List the steps in one decode iteration.
11. What would be recomputed without a KV cache?
12. Why does KV-cache memory grow with sequence length, layer count, batch size,
    and representation width?
13. Does the KV cache remove all decode computation? Explain.

## Prefill and Decode

14. Which phase most directly affects TTFT?
15. Which phase most directly affects TPOT?
16. Compare a 2,000-token prompt with 10 output tokens against a 10-token prompt
    with 2,000 output tokens.

## Sampling

17. Compare greedy decoding, temperature, top-k, and top-p.
18. Why is deterministic generation useful for performance baselines?
19. Can two sampling configurations have similar speed but different output
    quality? Explain.

## Memory Growth

20. Categorize weights, activations, KV cache, and temporary workspaces as
    mostly fixed or request-dependent for one loaded model.
21. Why can a request fail with out-of-memory even though model loading worked?
