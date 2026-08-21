# Hands-On Inference Performance Learning Path

## How This Path Works

This is the primary practice route for a learn-by-doing student. Reading is
limited to what is needed to begin an exercise. Each exercise follows one loop:

```text
SEE        inspect one labeled visual or concrete example
PREDICT    write an expected result before running anything
BUILD      implement or manipulate the mechanism yourself
CHECK      use automated checks or a controlled measurement
EXPLAIN    state why the result occurred in your own words
EXTEND     change one variable and predict again
```

Do not judge progress by pages read. Judge it by what you can build, predict,
measure, and explain without notes.

## Exercise Sequence

| Exercise | Build | Performance habit | Hardware | Status |
| ---: | --- | --- | --- | --- |
| [01 — Batches and Tensor Shapes](exercises/01-batches-and-tensor-shapes/README.md) | Construct and index a 3D batch tensor using Python loops | Shapes, elements, bytes, scaling | Mac | **Ready—start here** |
| 02 — Reshape and Transpose | Implement index mapping and compare traversal orders | Layout and memory access | Mac | Planned |
| 03 — Dot Product and Matrix Multiplication | Implement multiply-accumulate loops and shape validation | FLOPs/MACs and scaling | Mac | Planned |
| 04 — Timing Is an Experiment | Build a timing harness with warm-up and distributions | Measurement correctness | Mac, then GPU | Planned |
| 05 — Hardware Calibration | Measure launch overhead, bandwidth, and GEMM throughput | Empirical hardware ceilings | NVIDIA GPU | Planned |
| 06 — E2E Latency Ledger | Instrument tokenize, prefill, sampling, and decode boundaries | Additive models and boundary discipline | Mac, then GPU | Planned |
| 07 — Prefill Model | Predict FLOPs, bytes, and prompt-length scaling | Roofline-style modeling | NVIDIA GPU | Planned |
| 08 — Decode and KV Ledger | Model cache capacity, history reads, and per-token latency | Memory-bound reasoning | NVIDIA GPU | Planned |
| 09 — Profiler Evidence Hunt | Find annotated phases, gaps, and dominant operators | Tool selection and root cause | NVIDIA GPU | Planned |
| 10 — Optimization Case Study | Change one variable and conduct a before/after experiment | Evidence-based optimization | NVIDIA GPU | Planned |

## Rules for Productive Practice

1. Write predictions before running checks.
2. Use explicit units: values, bytes, milliseconds, tokens, or tokens/second.
3. Implement the mechanical loop before using a library shortcut.
4. Change one independent variable at a time.
5. Keep failed predictions; they are evidence of how the model improved.
6. Explain results aloud or in writing without copying the lesson.
7. Open the solution only after a genuine attempt and the hint ladder.

## Weekly Rhythm

Use three kinds of sessions:

- **Build session (60–90 minutes):** complete or substantially advance one
  exercise.
- **Measurement session (60–90 minutes):** run controlled experiments and
  preserve raw results.
- **Explanation session (30–45 minutes):** reconstruct diagrams, answer checks,
  and update the learning journal without notes.

Early exercises run locally. Do not rent a GPU just to practice Python loops or
tensor shapes.

## Completion Evidence

For every exercise preserve:

- the initial prediction;
- working implementation;
- automated-check output;
- one changed-input extension;
- a short prediction-versus-observation explanation;
- questions that remain unresolved.

