# E2E Inference Pipeline — Experiment Report

## 1. Question

What specific performance question did this experiment answer?

## 2. Environment

| Field | Value |
| --- | --- |
| Date | |
| GPU and VRAM | |
| Driver / CUDA | |
| PyTorch | |
| Transformers | |
| Model and revision | |
| Precision | |
| OS / instance | |

## 3. Workload and Boundaries

| Dimension | Value |
| --- | --- |
| Prompt text and measured tokens | |
| Requested and actual output tokens | |
| Batch | 1 |
| Selection policy | greedy argmax |
| Warm-up runs / measured runs | |
| TTFT start and end boundaries | |
| E2E exclusions | |

## 4. Prediction

Record this section before measuring.

| Phase | Predicted ms | Reasoning |
| --- | ---: | --- |
| Tokenization | | |
| Prefill | | |
| First-token selection | | |
| Decode forward per step | | |
| Model-boundary generation | | |

## 5. Method

Explain synchronization, warm-up, trial count, controlled variables, and the
commands used. Identify where the five important boundaries appear in
`model.py`.

## 6. Results

| Phase | Predicted ms | Measured median ms | p90 ms | Absolute error ms | Absolute % error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tokenization | | | | | |
| Prefill | | | | | |
| First-token selection | | | | | |
| Decode forward per step | | | | | |
| Model-boundary generation | | | | | |

Add the prompt-length and output-length sweep tables here.

## 7. Profiler Evidence

Embed or link the timeline screenshot. State what the NVTX ranges prove, what
the kernel row proves, and what neither proves.

## 8. Reconciliation

Explain the largest prediction error. Classify it as missing work, wrong rate,
wrong scaling assumption, boundary error, measurement error, or another
specific cause.

## 9. Bottleneck Hypothesis

State the limiting resource or overhead, supporting evidence, conflicting
evidence, and confidence level.

## 10. Next Experiment

Change one variable. Predict which metric moves, its direction, approximate
magnitude, and the tradeoff that may worsen.

## 11. Lessons Learned

Summarize only conclusions supported by this experiment.

