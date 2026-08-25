# Problem 02 Answer — Decode vs. Prefill Roofline

Do not read this until the worksheet contains a complete attempt.

## Part A — Hardware Ridge Point

```text
ridge point = peak compute / peak memory bandwidth

            = 120 × 10¹² FLOPs/s
              ──────────────────
               600 × 10⁹ bytes/s

            = 200 FLOPs/byte
```

Under this model:

- an operation below `200 FLOPs/byte` reaches the bandwidth ceiling before the
  compute ceiling and is classified as memory-bound;
- an operation above `200 FLOPs/byte` reaches the compute ceiling before the
  bandwidth ceiling and is classified as compute-bound;
- an operation at the ridge puts equal pressure on the idealized ceilings.

## Part B — Decode Projection, `M=1`

### FLOPs

```text
FLOPs = 2MKN
      = 2 × 1 × 4096 × 4096
      = 33,554,432 FLOPs
```

### Bytes

```text
weight bytes = 4096 × 4096 × 2
             = 33,554,432 bytes

input bytes  = 1 × 4096 × 2
             = 8,192 bytes

output bytes = 1 × 4096 × 2
             = 8,192 bytes

total bytes  = 33,554,432 + 8,192 + 8,192
             = 33,570,816 bytes
```

### Arithmetic intensity

```text
AI = 33,554,432 FLOPs / 33,570,816 bytes
   ≈ 0.9995 FLOPs/byte
   ≈ 1 FLOP/byte
```

### Ideal time bounds

```text
compute time = 33,554,432 / (120 × 10¹²) seconds
             ≈ 0.0000002796203 seconds
             ≈ 0.2796 µs

memory time  = 33,570,816 / (600 × 10⁹) seconds
             ≈ 0.00005595136 seconds
             ≈ 55.9514 µs
```

Therefore:

```text
roofline lower bound = max(0.2796, 55.9514) µs
                     ≈ 55.9514 µs

classification = memory-bound
```

The byte time is approximately 200 times the compute time.

## Part C — Prefill Projection, `M=512`

### FLOPs

```text
FLOPs = 2MKN
      = 2 × 512 × 4096 × 4096
      = 17,179,869,184 FLOPs
```

### Bytes

```text
weight bytes = 4096 × 4096 × 2
             = 33,554,432 bytes

input bytes  = 512 × 4096 × 2
             = 4,194,304 bytes

output bytes = 512 × 4096 × 2
             = 4,194,304 bytes

total bytes  = 33,554,432 + 4,194,304 + 4,194,304
             = 41,943,040 bytes
```

### Arithmetic intensity

```text
AI = 17,179,869,184 FLOPs / 41,943,040 bytes
   = 409.6 FLOPs/byte
```

### Ideal time bounds

```text
compute time = 17,179,869,184 / (120 × 10¹²) seconds
             ≈ 0.0001431656 seconds
             ≈ 143.1656 µs

memory time  = 41,943,040 / (600 × 10⁹) seconds
             ≈ 0.0000699051 seconds
             ≈ 69.9051 µs
```

Therefore:

```text
roofline lower bound = max(143.1656, 69.9051) µs
                     ≈ 143.1656 µs

classification = compute-bound
```

## Comparison Table

| Quantity | Decode, `M=1` | Prefill, `M=512` | Unit |
| --- | ---: | ---: | --- |
| FLOPs | 33,554,432 | 17,179,869,184 | FLOPs |
| Weight read | 33,554,432 | 33,554,432 | bytes |
| Input read | 8,192 | 4,194,304 | bytes |
| Output write | 8,192 | 4,194,304 | bytes |
| Total traffic | 33,570,816 | 41,943,040 | bytes |
| Arithmetic intensity | approximately 0.9995 | 409.6 | FLOPs/byte |
| Compute time | approximately 0.2796 | approximately 143.1656 | µs |
| Memory time | approximately 55.9514 | approximately 69.9051 | µs |
| Roofline lower bound | approximately 55.9514 | approximately 143.1656 | µs |
| Bound classification | Memory | Compute | — |
| Lower bound per token row | approximately 55.9514 | approximately 0.2796 | µs/token |

## Part D — Comparison

### 1. Does weight traffic increase by 512 times?

No. The stated model reads `W` once per model call. The 512 token rows are
processed together against the same weight matrix, so weight traffic remains
`33,554,432 bytes`. Input and output traffic scale with `M`.

### 2. FLOP growth

```text
prefill FLOPs / decode FLOPs = 512
```

FLOPs scale linearly with `M` for this fixed projection.

### 3. Byte growth

```text
41,943,040 / 33,570,816 ≈ 1.2494
```

Total modeled traffic grows by only about `1.25×`, not `512×`, because the
large weight-read term is unchanged.

### 4. Why arithmetic intensity rises

Each weight value supports arithmetic for 512 rows instead of one row during
the call. Compute grows by `512×`, while modeled bytes grow by only about
`1.25×`. More arithmetic is performed per byte fetched from HBM.

### 5. Why the bound changes

Decode's approximately `1 FLOP/byte` is far below the `200 FLOPs/byte` ridge,
so memory time dominates. Prefill's `409.6 FLOPs/byte` is above the ridge, so
compute time dominates.

### 6. Lower-bound time per row

```text
decode  = 55.9514 µs / 1
        ≈ 55.9514 µs/token row

prefill = 143.1656 µs / 512
        ≈ 0.2796 µs/token row
```

This idealized projection processes rows much more efficiently together.

### 7. Per-row time versus request latency

Prefill has lower modeled time per row but higher total call latency:

```text
decode call lower bound  ≈ 55.95 µs
prefill call lower bound ≈ 143.17 µs
```

Efficiency per unit of work, aggregate throughput, and one-call latency are
different metrics.

## Part E — Crossover

Let `d = K = N = 4096`.

```text
FLOPs = 2Md²

bytes = weight + input + output
      = 2d² + 2Md + 2Md
      = 2d² + 4Md
```

Therefore:

```text
AI(M) = 2Md² / (2d² + 4Md)
      = Md / (d + 2M)
```

Set the expression equal to the `200 FLOPs/byte` ridge:

```text
4096M / (4096 + 2M) = 200

4096M = 819,200 + 400M

3696M = 819,200

M ≈ 221.645
```

The smallest integer that reaches or exceeds the ridge is:

```text
M = 222 token rows
```

## What This Model Does Not Prove

It does not prove that a measured decode kernel will take `55.95 µs` or that a
measured prefill kernel will be compute-bound. Real behavior depends on:

- achieved rather than peak bandwidth and compute;
- kernel shape efficiency and Tensor Core eligibility;
- launch and framework overhead;
- memory hierarchy and cache reuse;
- tiling, fusion, precision, and layout;
- additional reads, writes, and workspace;
- concurrent workloads and scheduling;
- the rest of the Transformer layer.

The model creates a falsifiable starting hypothesis. Profiling determines why
the hardware does or does not follow it.

