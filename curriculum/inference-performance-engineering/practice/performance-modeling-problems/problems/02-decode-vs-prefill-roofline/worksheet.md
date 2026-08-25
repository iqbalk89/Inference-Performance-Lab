# Problem 02 Worksheet — Decode vs. Prefill Roofline

## Attempt Record

- Date:
- Start time:
- Finish time:
- Completed without hints? Yes / No

## Predict Before Calculating

1. Which workload do you predict will be memory-bound?
2. Which workload do you predict will be compute-bound?
3. Will prefill move 512 times as many bytes as decode?
4. Which workload will have lower idealized time per token row?
5. Confidence from 1–5, and why:

## Shape Ledger

| Workload | `M` | `K` | `N` | Input shape | Weight shape | Output shape |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Decode | | | | | | |
| Prefill | | | | | | |

## Part A — Hardware Ridge Point

```text
ridge point =

=

=                         FLOPs/byte
```

Below the ridge means:

Above the ridge means:

## Part B — Decode

```text
FLOPs =

weight bytes =

input bytes =

output bytes =

total bytes =

arithmetic intensity =

compute time in seconds =
compute time in µs =

memory time in seconds =
memory time in µs =

roofline lower bound =

classification =
```

## Part C — Prefill

```text
FLOPs =

weight bytes =

input bytes =

output bytes =

total bytes =

arithmetic intensity =

compute time in seconds =
compute time in µs =

memory time in seconds =
memory time in µs =

roofline lower bound =

classification =
```

## Comparison Table

| Quantity | Decode, `M=1` | Prefill, `M=512` | Unit |
| --- | ---: | ---: | --- |
| FLOPs | | | FLOPs |
| Weight read | | | bytes |
| Input read | | | bytes |
| Output write | | | bytes |
| Total traffic | | | bytes |
| Arithmetic intensity | | | FLOPs/byte |
| Compute time | | | µs |
| Memory time | | | µs |
| Roofline lower bound | | | µs |
| Bound classification | | | — |
| Lower bound per token row | | | µs/token |

## Part D — Explanation

1. Why does weight traffic not scale with `M` in this model?

2. FLOP-count ratio:

3. Total-byte-count ratio:

4. Why arithmetic intensity rises:

5. Why the bottleneck classification changes:

6. Why lower per-row time is not lower request latency:

## Part E — Crossover

Derive symbolically before substituting `d = 4096`:

```text
AI(M) =

Set AI(M) >= 200:


Smallest integer M =
```

## Prediction Versus Result

| Initial prediction | Correct? | What changed in your mental model? |
| --- | --- | --- |
| Decode classification | | |
| Prefill classification | | |
| Byte-growth expectation | | |
| Per-row expectation | | |

## Two-Minute Explanation Outline

Explain aloud without reading equations:

1. Same weights, different number of token rows
2. Weight reuse within one model call
3. FLOPs versus bytes
4. Arithmetic intensity
5. Hardware ridge point
6. Decode result
7. Prefill result
8. Limitations of the model

