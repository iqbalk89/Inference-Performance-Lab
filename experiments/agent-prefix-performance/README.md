# Agent Workload Inference Performance

This directory contains the experiments for the [four-week inference
performance roadmap](../../docs/agent-prefix-performance-roadmap.md). Each
numbered directory is a self-contained experiment with its own README,
visuals, code, and findings.

## Experiments

1. [01 — Direct PyTorch baseline](01-pytorch-baseline/README.md)

   Establishes the single-request prefill/decode baseline, explains tensor
   shapes and KV caching, and records latency, throughput, and memory.

2. [02 — Batch-size scaling](02-batch-size-scaling/README.md)

   Measures how processing multiple independent requests together changes
   throughput, per-request latency, and GPU memory use.

Shared raw benchmark outputs belong under the ignored `benchmark-results/`
directory. Do not commit model weights or raw benchmark JSON; commit scripts,
commands, visualizations, and summarized findings.
