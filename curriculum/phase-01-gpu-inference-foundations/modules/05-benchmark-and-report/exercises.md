# Module 05 Exercises

## Metrics

1. Define TTFT, TPOT, output tokens per second, end-to-end latency, and peak
   memory. State each unit.
2. Why can a system improve throughput while worsening per-request latency?
3. Why should TTFT and decode throughput not be collapsed into one number?

## Experimental Design

4. What variables must remain fixed when comparing prompt lengths?
5. Why are warmup and repeated trials necessary?
6. What summary statistics would you report for latency, and why?
7. What is wrong with selecting the fastest observed run as the result?
8. List five environmental details required for reproduction.

## Workload Matrix

9. What does a long-prompt/short-output case isolate?
10. What does a short-prompt/long-output case isolate?
11. How should random sampling be controlled during a performance comparison?

## Memory and Batching

12. Predict how batch size affects throughput, latency, and memory.
13. Predict how total sequence length affects KV-cache memory.
14. Why might `nvidia-smi` memory differ from PyTorch peak allocated memory?

## Interpretation

15. Distinguish a measurement, an observation, an interpretation, and a causal
    conclusion.
16. Give three confounders that could invalidate a model-performance comparison.
17. What must a later quantization experiment hold constant?
18. What quality evidence is needed before calling a quantized model better?
