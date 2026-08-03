# Module 05 Lab — Phase 1 Benchmark and Report

## Objective

Produce the first controlled server-side inference benchmark and a professional
engineering report.

## Benchmark Matrix

At minimum compare:

- Short prompt / short output
- Long prompt / short output
- Short prompt / long output
- Cold model start / warm request
- Batch size 1 / one larger supported batch

Use a fixed model, revision, dtype, hardware type, software environment,
generation policy, and measurement implementation.

## Required Metrics

- Input and output token counts
- Model-load time for cold-start cases
- TTFT
- TPOT
- End-to-end latency
- Output tokens per second
- Request or batch throughput
- Peak PyTorch allocated and reserved CUDA memory
- Device memory observation

Store machine-readable results under `benchmarks/results/`, following the
repository's ignore policy for bulky raw data. Commit a small representative
dataset or summary when appropriate.

## Report

Create `docs/reports/phase-1-initial-inference-report.md` with:

1. Objective
2. Hypotheses
3. Environment and dependencies
4. Model and workload definitions
5. Methodology and timing boundaries
6. Results
7. Analysis
8. Limitations and confounders
9. Lessons learned
10. Reproduction instructions

## Required Analysis

- Explain the prompt-length effect on TTFT.
- Explain the output-length effect on total latency.
- Explain batch-size tradeoffs.
- Compare predicted and observed KV-cache memory growth.
- Identify evidence of compute, memory-bandwidth, or latency limitation without
  overstating profiler evidence.
- Define the stable FP16/BF16 baseline for a future quantization comparison.

## Pass Criteria

- Benchmark commands are repeatable.
- Raw data can be traced to reported values.
- Units, warm/cold state, and sample counts are explicit.
- Failed or anomalous runs are disclosed rather than silently removed.
- The report answers the hypotheses and states what remains unknown.
