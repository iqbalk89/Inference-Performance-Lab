# Agent Prefix Performance Experiment

This directory contains the executable work for the
[four-week project roadmap](../../docs/agent-prefix-performance-roadmap.md).

For the shared math behind Week 1, read
[the formula page](week1-formulas.md). It collects the tensor-shape, KV-cache,
throughput, and storage equations used throughout this experiment.

## Week 1: Direct PyTorch Baseline

Run this on the accepted Lambda A10, not on the local Intel Mac.

Before or after the first run, read the
[Week 1 PyTorch code walkthrough](pytorch-code-walkthrough.md). It explains how
Transformers constructs Qwen, how PyTorch moves its tensors to the A10, how the
prefill and decode loops use the KV cache, and why vLLM is not involved yet.

For the shortest commented example, see
[`week1_pytorch_minimal.py`](week1_pytorch_minimal.py). It contains the same
prefill and cached-decode ideas as the benchmark, without timing or measurement
bookkeeping.

### 1. Prepare the instance

Follow the [Lambda runbook](../../docs/gpu-notes/lambda-instance-runbook.md),
then create an environment that retains Lambda Stack's CUDA-enabled PyTorch:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experiments/agent-prefix-performance/requirements-gpu.txt
```

Confirm that the environment did not replace CUDA PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 2. Run the first measurement

```bash
python experiments/agent-prefix-performance/week1_pytorch_baseline.py \
  --prompt-tokens 512 \
  --new-tokens 32 \
  --warmups 1 \
  --repeats 3
```

The command downloads the model on its first run and writes ignored raw JSON to
`benchmark-results/agent-prefix-performance/week1/`. It measures:

- synchronized prefill and decode time;
- cached decode versus full-prefix recomputation;
- input, logits, and KV-cache shapes;
- estimated model and observed cache bytes;
- peak allocated and reserved CUDA memory; and
- model, dependency, GPU, driver, dtype, and workload metadata.

The default revision is `main` only for the discovery run. Copy the resolved
commit hash from the JSON and use it for every formal run:

```bash
python experiments/agent-prefix-performance/week1_pytorch_baseline.py \
  --revision <RESOLVED_COMMIT_HASH> \
  --prompt-tokens 512 \
  --new-tokens 32
```

### 3. Run the Week 1 matrix

Start with an inexpensive correctness pass, then run the four combinations of:

- prompt tokens: 512 and 2,048
- new tokens: 32 and 128
- warmups: 1
- measured repetitions: 5

Use the command above for each combination, changing `--prompt-tokens` and
`--new-tokens`. Full-prefix recomputation becomes deliberately expensive at the
largest case. Use `--skip-no-cache` after at least one valid cached/uncached
comparison if the remaining no-cache runs add cost without new information.

### 4. Interpret the result

Create `week1-findings.md` in this directory after the remote run. Include:

1. A table of median prefill, decode, total latency, tokens/s, and peak VRAM.
2. The shapes of inputs, logits, and one layer's key/value cache.
3. Why prompt length primarily changes prefill and cache memory.
4. Why output length primarily changes decode work.
5. Why no-cache generation repeatedly pays for the growing prefix.
6. At least one surprising result and a proposed profiler question for Week 2.

The completed baseline analysis is available in
[week1-findings.md](week1-findings.md).

## Week 2: Batch-size scaling

Week 2 studies what happens when the model serves several requests at once.
Before running it, make sure the term **batch size** is clear: the batch size
`B` is the number of independent sequences included in one model call. With
`B=1`, the model receives one request. With `B=4`, it receives four requests
at the same time and performs the same layers and matrix operations for four
rows of data in parallel. The requests do not share words, attention scores, or
KV entries; batching only gives the GPU more independent work to process
together.

![Detailed batch-size visualization](assets/week2-batch-size-visual.svg)

### What changes when B increases

For the Week 1 Qwen configuration, one request with a 512-token prompt has
these representative shapes:

```text
input_ids       [1, 512]
hidden states   [1, 512, 1536]
K per layer     [1, 2, 512, 128]
V per layer     [1, 2, 512, 128]
```

With four equal-length requests, the leading batch dimension changes to 4:

```text
input_ids       [4, 512]
hidden states   [4, 512, 1536]
K per layer     [4, 2, 512, 128]
V per layer     [4, 2, 512, 128]
```

The model weights are not duplicated. The same weights process every batch
item, while the request-dependent tensors gain four times as many rows. The
KV-cache equation makes the memory consequence explicit:

```text
KV bytes = 2 × B × L × H_kv × T × D × bytes_per_element
```

At `T=512`, the Week 1 cache was 14 MiB for `B=1`. Holding every other factor
constant gives approximately 28 MiB at `B=2`, 56 MiB at `B=4`, and 112 MiB at
`B=8`. This is only the KV cache; weights, activations, logits, CUDA
workspaces, and allocator reserve also consume memory.

### Why batching can improve throughput

A GPU is most efficient when it has enough independent matrix work to keep its
many streaming multiprocessors busy. A single short decode step can leave much
of the GPU idle. Processing four requests together makes the batch dimension
larger, so matrix multiplications have more rows and the fixed cost of launching
and scheduling work is shared across requests. The result can be higher
**aggregate throughput** (tokens/second across all requests), even though each
individual request may take longer to complete.

### Why batching can hurt latency or capacity

Every additional request adds input/activation storage and a separate K/V
history. Larger batches therefore increase peak memory and can cause queueing:
an arriving request may wait for an existing batch to finish. The goal is not
to choose the largest possible `B`; it is to find the largest useful batch that
meets a latency and memory target.

### Equal and unequal sequence lengths

The first experiment should use equal-length prompts so the effect of `B` is
isolated. Real serving traffic has different prompt and generation lengths. A
framework can handle that with padding (wasted computation on placeholder
tokens), packing/variable-length kernels, or continuous batching (admitting
new sequences as other sequences finish). Those are separate effects to study
after the controlled sweep.

### Prefill versus decode batching

Batching has two different meanings in the two phases:

- **Prefill:** each request may contribute hundreds or thousands of prompt
  tokens. Increasing `B` increases the large matrix operations and the KV cache
  created for each prompt.
- **Decode:** each active request contributes approximately one new token per
  step. Batching active sequences is especially important here because it turns
  many small one-token operations into a larger GPU workload.

### Week 2 objective and measurements

Run the same cached-generation workload at prompt lengths 512 and 2,048 while
sweeping `B=1, 2, 4, 8, ...` until the A10 approaches its memory limit. Record:

- aggregate tokens/second;
- per-request tokens/second;
- prefill, decode, and end-to-end latency;
- peak allocated and reserved CUDA memory;
- measured KV-cache bytes; and
- the first batch size that fails or violates the chosen latency target.

Use these definitions when interpreting the results:

```text
aggregate throughput = total generated tokens / elapsed seconds
per-request throughput = aggregate throughput / B
KV bytes = 2 × B × L × H_kv × T × D × bytes_per_element
```

The deliverable is a plot or table showing where aggregate throughput improves,
where it flattens, and where memory or latency becomes the limiting constraint.

Before you write the findings note, perform these calculations from the
measured run:

1. model parameter bytes;
2. Q/K/V, score, probability, and context tensor shapes;
3. expected KV-cache bytes at each prompt length;
4. cached-versus-uncached speedup; and
5. the reason prompt length changes prefill and cache memory more than decode.

Do not commit model weights or raw benchmark JSON. Commit the exact commands and
the summarized findings.
