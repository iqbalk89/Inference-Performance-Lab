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

Before you write the findings note, perform these calculations from the
measured run:

1. model parameter bytes;
2. Q/K/V, score, probability, and context tensor shapes;
3. expected KV-cache bytes at each prompt length;
4. cached-versus-uncached speedup; and
5. the reason prompt length changes prefill and cache memory more than decode.

Do not commit model weights or raw benchmark JSON. Commit the exact commands and
the summarized findings.
