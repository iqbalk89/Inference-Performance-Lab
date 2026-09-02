# Agent Workload Inference Performance Project

## Objective

Build and publish an evidence-backed optimization study for an agent-style LLM
workload with a long, mostly stable prefix and short per-turn additions. The
project is successful when it can explain *why* a configuration changes TTFT,
inter-token latency, throughput, memory use, and tail latency—not merely report
that one setting is faster.

The initial performance hypothesis is:

> Prefix-aware serving and workload-specific scheduling can reduce P99 time to
> first token by at least 50% relative to the direct PyTorch baseline while
> preserving at least 95% of output-token throughput.

This is a hypothesis to test, not a promised result. A well-supported negative
result is still valuable.

## Fixed Workload

- One decoder-only model throughout the first study.
- Development model: `Qwen/Qwen2.5-1.5B-Instruct` in FP16 on one NVIDIA A10.
- Agent request: 4K–8K tokens of stable instructions/context, 100–500 tokens of
  changing turn content, and a controlled output-token budget.
- Primary metric: P99 TTFT under concurrent load.
- Guardrail: output-token throughput must remain within 5% of the baseline.
- Supporting metrics: median/P95 TTFT, TPOT/ITL, end-to-end latency, request and
  token throughput, peak VRAM, cache hit rate, and failure rate.

The 1.5B model keeps early experiments inexpensive. After the methodology is
stable, repeat the strongest experiment on a 7B-class model if the A10 memory
budget permits it.

## Week-by-Week Plan

### Week 1 — Direct PyTorch Baseline

**Question:** Where do time and memory go in a single, visible inference loop?

- Establish a reproducible GPU environment and record the exact model revision,
  software versions, GPU, driver, dtype, and generation settings.
- Load one Transformers model with direct PyTorch execution.
- Record model parameter size, input tensor shapes, logits shape, and KV-cache
  shapes and bytes.
- Separate prefill from autoregressive decode.
- Compare decode with KV caching against full-prefix recomputation.
- Use CUDA events and synchronization correctly; distinguish warmup from
  measured runs.
- Sweep at least two prompt lengths and two output lengths.

**Deliverable:** runnable baseline, JSON results, and a short findings note with
one table and at least three evidence-backed observations.

### Week 2 — Explain the Baseline with Profilers

**Question:** Which framework operations, CUDA launches, and GPU kernels explain
the Week 1 result?

- Add NVTX ranges around tokenization, prefill, and decode.
- Capture a PyTorch Profiler trace and summarize dominant operators.
- Capture an Nsight Systems trace and identify CPU gaps, transfers, launch
  patterns, synchronization, and prefill/decode differences.
- Relate GEMMs, attention, normalization, sampling, and memory operations back
  to the model loop.
- Use Nsight Compute only for a focused kernel question that the timeline cannot
  answer; custom-kernel development is out of scope.

**Deliverable:** annotated traces and a concise bottleneck analysis that names
the next optimization target.

### Week 3 — Move the Same Workload to vLLM

**Question:** What does a serving engine change compared with the direct
PyTorch loop?

- Serve the same model and use an asynchronous streaming client.
- Reproduce the Week 1 prompt/output matrix.
- Measure TTFT, TPOT/ITL, throughput, and VRAM from engine and client views.
- Study continuous batching, paged KV-cache allocation, scheduler behavior, and
  concurrency scaling.
- Compare PyTorch and vLLM only under documented equivalent conditions.

**Deliverable:** apples-to-apples comparison, concurrency curves, and an
explanation of where the engine wins and where it does not.

### Week 4 — Optimize the Agent Workload

**Question:** Which engine and workload choices improve long-prefix multi-turn
traffic without sacrificing throughput or reliability?

- Generate a reproducible workload with stable prefixes and changing suffixes.
- Test automatic prefix caching, chunked prefill, concurrency, token budgets,
  and eager versus optimized execution where applicable.
- Run one-factor ablations first, then test the best combined configuration.
- Include repeated trials, percentiles, cache warm/cold states, errors, and an
  explicit cost or GPU-efficiency view.
- Attempt to falsify the 50% P99 TTFT / 95% throughput hypothesis.

**Deliverable:** employer-facing engineering report with methodology, charts,
tradeoffs, limitations, reproducible commands, and recommended configuration.

## Engineering Standard

Every reported result must include hardware, software revisions, exact command,
workload distribution, warmup, sample count, and metric definition. Preserve
raw outputs outside Git under `benchmark-results/` or `profile-results/`, and
commit scripts plus summarized findings. Do not tune against a single request
and present it as a serving result.

## Current Status

- [x] Lambda A10 environment and profiler access accepted.
- [x] Four-week project and success hypothesis defined.
- [x] Week 1 baseline scaffold added.
- [ ] Week 1 remote dependencies installed and exact model commit recorded.
- [ ] Week 1 prompt/output sweep run on the A10.
- [ ] Week 1 findings summarized.
- [ ] Week 2 profiler analysis.
- [ ] Week 3 vLLM comparison.
- [ ] Week 4 workload optimization report.

Start with the commands in
[`experiments/agent-prefix-performance/README.md`](../experiments/agent-prefix-performance/README.md).
