# Lab — Predict and Measure One Inference Request

## Objective

Make an explicit phase-level latency prediction, measure the same boundaries,
inspect their profiler annotations, and explain the difference. This lab uses a
small model so the workflow is understandable; it is not a production-serving
benchmark.

## What the Code Makes Explicit

The code deliberately avoids `model.generate()` so you can see:

- tokenization;
- one prefill forward call over the prompt;
- first-token selection from prefill logits;
- one decode forward call per later output decision;
- reuse and growth of `past_key_values`;
- synchronization around device timings;
- NVTX labels for Nsight Systems.

The code uses greedy `argmax` selection to reduce sampling variability. Real
servers add scheduling, batching, streaming, stop handling, and optimized
attention/cache implementations.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | explicit prefill/decode execution and phase timers |
| `benchmark.py` | warm-up, repeated trials, metadata, and JSON output |
| `analyze.py` | median, p90, and minimum phase summary |
| `prediction-template.json` | prediction recorded before measurement |
| `profile.sh` | Nsight Systems capture with CUDA and NVTX |
| `expected-observations.md` | interpretation guide, opened after prediction |
| `report-template.md` | required experiment report |

## Part 1 — Define the Experiment Before Running It

Copy the template so the original remains reusable:

```bash
cp prediction-template.json results/prediction.json
```

Fill every `null` and reasoning field. Your first prediction may be inaccurate.
That is expected. The point is to expose the mental model before seeing results.

Use one canonical workload for the first comparison:

```text
model: distilgpt2
prompt: Cats chase mice
requested output: 8 tokens
batch: 1
selection: greedy
```

Record the exact latency boundary: this code measures tokenizer start through
token selection. It does **not** include network, queue, detokenization, or
streaming time in its TTFT field.

## Part 2 — Prepare the Environment

From this lab directory, create an isolated environment. On the Intel Mac, this
is a workflow check only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers
```

On the remote GPU, use the CUDA-enabled PyTorch environment installed by the
repository bootstrap process. Confirm before measuring:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Part 3 — Inspect Before Executing

In `model.py`, find and explain these five points:

1. Device synchronization before and after a timed region
2. The prompt-wide prefill call
3. The first token selected from `logits[:, -1, :]`
4. The next call receiving only `next_token` plus `past_key_values`
5. The cache returned from one step and passed to the next

Write your explanation in the report. If you cannot locate one, do not proceed
until the boundary is clear.

## Part 4 — Mac Dry Run

This validates code and downloads the model; it is not NVIDIA evidence:

```bash
python benchmark.py --device cpu --warmups 1 --runs 2
python analyze.py results/measurements.json
```

An MPS run may be used on Apple Silicon, but the current 2018 Intel Mac should
use CPU. Do not compare its timings with the remote GPU as if the environments
were equivalent.

## Part 5 — GPU Measurement

On the remote NVIDIA instance:

```bash
python benchmark.py \
  --device cuda \
  --warmups 3 \
  --runs 10 \
  --output results/a10-canonical.json

python analyze.py results/a10-canonical.json
```

Why warm up? First execution may include model initialization side effects,
memory allocation, library setup, and kernel-selection work that does not
represent steady state.

Why synchronize? CUDA launches are asynchronous relative to the CPU. Without a
device synchronization, a CPU clock can measure only the launch rather than the
completed GPU work.

## Part 6 — Controlled Sweeps

Change one independent variable at a time.

### Prompt-length sweep

Use three prompts whose measured token counts differ substantially while
keeping `--max-new-tokens 8`. Record actual token counts from the JSON.

### Output-length sweep

Keep the prompt fixed and run:

```bash
python benchmark.py --device cuda --max-new-tokens 4  --output results/output-4.json
python benchmark.py --device cuda --max-new-tokens 16 --output results/output-16.json
python benchmark.py --device cuda --max-new-tokens 64 --output results/output-64.json
```

Before each sweep, write the expected direction of TTFT, decode-step latency,
and total generation time. Do not infer scaling from requested token count when
the model stops early; use the measured output count.

## Part 7 — Capture the Annotated Timeline

On the NVIDIA instance:

```bash
chmod +x profile.sh
./profile.sh
```

Open `results/e2e-inference.nsys-rep` in Nsight Systems. Locate:

- `phase/tokenize`;
- `phase/prefill_forward`;
- `phase/first_token_selection`;
- each `phase/decode_forward_N`;
- each `phase/decode_selection_N`;
- CUDA API activity and kernels nested below the CUDA ranges.

Capture one screenshot containing the named ranges and GPU kernel row. In the
report, distinguish elapsed NVTX range time from summed kernel time.

## Part 8 — Reconcile Prediction and Measurement

For each modeled phase calculate:

```text
signed error = measured - predicted
absolute percentage error = |measured - predicted| / measured × 100
```

Then answer:

1. Which phase had the largest absolute error?
2. Was the prediction missing work, using the wrong rate, or using an incorrect
   phase boundary?
3. Does the Nsight timeline support that explanation?
4. What single next measurement would most reduce uncertainty?
5. What optimization would you test, and which metric might it harm?

Read [expected observations](expected-observations.md) only after recording your
own interpretation.

## Completion Evidence

- [ ] `results/prediction.json` created before measurements
- [ ] Canonical ten-run GPU JSON retained
- [ ] Prompt-length and output-length sweep results retained
- [ ] Nsight Systems report retained or copied off the paid instance
- [ ] One annotated timeline screenshot retained
- [ ] [Report template](report-template.md) completed
- [ ] Prediction errors and next hypothesis documented

Before terminating the paid instance, follow the repository's
[Lambda runbook](../../../../docs/gpu-notes/lambda-instance-runbook.md) and copy
all evidence to durable storage.

