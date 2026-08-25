# Problem 02 — Decode vs. Prefill: Memory-Bound or Compute-Bound?

## Why This Is the Next Problem

Problem 01 counted parameters, weight bytes, and FLOPs for one decode-time
projection. A performance engineer must take the next step:

> Given a workload and hardware ceilings, which resource imposes the larger
> idealized latency—and how does that change between decode and prefill?

This problem introduces a simple roofline-style model. It deliberately analyzes
only one dense Transformer projection so the reasoning stays visible.

## Interview Setup

Suppose a Transformer projection computes:

```text
Y = XW
```

with:

```text
W shape: [4096 × 4096]
weight format: FP16 = 2 bytes/value
activation format: FP16 = 2 bytes/value
```

Compare two model calls:

### Workload A — Decode

One active sequence processes one new token row:

```text
X_decode: [1 × 4096]
W:        [4096 × 4096]
Y_decode: [1 × 4096]
```

### Workload B — Prefill

One prompt processes 512 token rows together:

```text
X_prefill: [512 × 4096]
W:         [4096 × 4096]
Y_prefill: [512 × 4096]
```

## Hypothetical GPU

Use these idealized ceilings—not specifications for a particular product:

```text
Peak HBM bandwidth:       600 GB/s
Peak FP16 compute:        120 TFLOP/s
```

Use decimal performance units:

```text
1 GB     = 10^9 bytes
1 TFLOP  = 10^12 FLOPs
1 second = 10^6 microseconds (µs)
```

## Simplified Traffic Model

For each projection, count only:

```text
read W once + read X once + write Y once
```

Assume:

- `W` is read from HBM once per model call;
- `X` is read from HBM once;
- `Y` is written to HBM once;
- no weight or activation traffic is reused from on-chip cache;
- no intermediate workspace or extra read/write traffic exists;
- bandwidth and compute cannot hide one another beyond the roofline maximum;
- the hardware can attain the stated peak ceilings.

These assumptions create a theoretical lower-bound model, not a measured
latency prediction.

## Given Formulas

For:

```text
[M × K] [K × N] → [M × N]
```

use:

```text
FLOPs = 2MKN

weight bytes = K × N × bytes_per_value
input bytes  = M × K × bytes_per_value
output bytes = M × N × bytes_per_value
total bytes  = weight bytes + input bytes + output bytes

arithmetic intensity = FLOPs / total bytes

compute time = FLOPs / peak compute rate
memory time  = total bytes / peak memory bandwidth

roofline lower bound = max(compute time, memory time)

ridge point = peak compute rate / peak memory bandwidth
```

Arithmetic intensity is measured in `FLOPs/byte`. The ridge point has the same
unit.

## Your Tasks

### Part A — Hardware balance

1. Calculate the GPU's ridge point in `FLOPs/byte`.
2. Explain what it means when an operation lies below or above that value.

### Part B — Decode projection

Calculate:

1. FLOPs
2. Weight bytes
3. Input bytes
4. Output bytes
5. Total bytes
6. Arithmetic intensity
7. Ideal compute time in microseconds
8. Ideal memory time in microseconds
9. Roofline lower bound
10. Classification: memory-bound or compute-bound

### Part C — Prefill projection

Repeat all ten calculations for `M = 512`.

### Part D — Compare and explain

Answer:

1. The prefill call has 512 times as many token rows. Does its weight traffic
   increase by 512 times? Why or why not?
2. Approximately how many times larger is prefill's total FLOP count?
3. Approximately how many times larger is its total byte count?
4. Why does arithmetic intensity rise dramatically?
5. Why can decode and prefill land on different sides of the roofline ridge?
6. Calculate idealized time per processed token row for each workload.
7. Does lower time per row mean lower request latency?

### Part E — Crossover stretch problem

For arbitrary token-row count `M`, derive the arithmetic-intensity expression:

```text
AI(M) = projection FLOPs / projection bytes
```

Then estimate the smallest integer `M` for which this simplified projection
reaches or exceeds the `200 FLOPs/byte` ridge point.

## Required Answer Format

Build one table before writing prose:

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

Use exact values where practical and label every approximation.

## Deliverables

- [ ] Completed [worksheet](worksheet.md)
- [ ] All assumptions restated in your own words
- [ ] Comparison table with units
- [ ] Crossover derivation
- [ ] Two-minute verbal explanation using no notes
- [ ] Interview follow-ups attempted

Use [hints.md](hints.md) one section at a time if stuck. Open
[answer.md](answer.md) only after completing a full attempt.

