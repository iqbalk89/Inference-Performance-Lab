# Problem 02 Answer — Decode vs. Prefill Roofline

Do not read this until the worksheet contains a complete attempt.

## Map Every Number to Its Source

| Number | Meaning | Source |
| --- | --- | --- |
| `M = 1` | Decode rows | `X_decode` shape `[1 × 4096]` |
| `M = 512` | Prefill rows | `X_prefill` shape `[512 × 4096]` |
| `K = 4096` | Input features per row | Second dimension of `X`; first dimension of `W` |
| `N = 4096` | Output features per row | Second dimension of `W` and `Y` |
| `2 FLOPs` | One multiply plus one add | The problem's `2MKN` convention |
| `2 bytes/value` | FP16 storage | Specified for weights and activations |
| `600 × 10⁹ bytes/s` | HBM ceiling | `600 GB/s`, with decimal `1 GB = 10⁹ bytes` |
| `120 × 10¹² FLOPs/s` | Compute ceiling | `120 TFLOP/s`, with `1 TFLOP = 10¹² FLOPs` |
| `10⁶ µs/s` | Time conversion | One second contains one million microseconds |

The operation is:

```text
X [M × K] × W [K × N] → Y [M × N]
```

Only `M` changes between the two workloads.

## Part A — Hardware Ridge Point

The ridge is the GPU's compute rate divided by its byte-delivery rate:

```text
                         peak compute rate
ridge point = ───────────────────────────────────
                         peak HBM bandwidth

              120 TFLOP/s × (10¹² FLOPs / TFLOP)
            = ───────────────────────────────────────
                600 GB/s × (10⁹ bytes / GB)

              120 × 10¹² FLOPs/s
            = ──────────────────
               600 × 10⁹ bytes/s

            = (120 / 600) × (10¹² / 10⁹) FLOPs/byte
            = 0.2 × 1,000 FLOPs/byte
            = 200 FLOPs/byte
```

The seconds cancel. Below `200 FLOPs/byte`, byte delivery reaches its ceiling
first: memory-bound. Above it, compute reaches its ceiling first: compute-bound.
At 200, both ideal ceilings meet.

## Part B — Decode Projection, `M = 1`

```text
X [1 × 4096] × W [4096 × 4096] → Y [1 × 4096]

M = 1       one new-token row
K = 4096    input features consumed by each dot product
N = 4096    output values produced per row
```

### B1. FLOPs

There are `M × N` outputs. Each is a length-`K` dot product, with approximately
one multiplication plus one addition—2 FLOPs—for each of its `K` contributions.

```text
FLOPs = 2 × M × K × N
      = 2 FLOPs/contribution
        × 1 row
        × 4096 contributions/output
        × 4096 outputs/row
      = 2 × 1 × 4096 × 4096
      = 33,554,432 FLOPs
```

### B2. Weight bytes

`W` contains `K × N` FP16 values and the model reads it once from HBM:

```text
weight values = 4096 × 4096 = 16,777,216 values
weight bytes  = 16,777,216 values × 2 bytes/value
              = 33,554,432 bytes
```

### B3. Input bytes

`X_decode` contains `M × K` FP16 values:

```text
input values = 1 × 4096 = 4,096 values
input bytes  = 4,096 values × 2 bytes/value
             = 8,192 bytes
```

### B4. Output bytes

`Y_decode` contains `M × N` FP16 values:

```text
output values = 1 × 4096 = 4,096 values
output bytes  = 4,096 values × 2 bytes/value
              = 8,192 bytes
```

### B5. Total modeled HBM traffic

The stated model counts one weight read, one input read, and one output write:

```text
total bytes = weight read + input read + output write
            = 33,554,432 + 8,192 + 8,192
            = 33,570,816 bytes
```

### B6. Arithmetic intensity

Divide B1's work by B5's HBM traffic:

```text
AI = 33,554,432 FLOPs / 33,570,816 bytes
   ≈ 0.9995 FLOPs/byte
   ≈ 1 FLOP/byte
```

It is slightly below 1 because traffic includes the weights plus the input and
output—not just the weights.

### B7. Ideal compute time

Divide B1's work by the given `120 TFLOP/s` compute rate:

```text
T_compute = 33,554,432 FLOPs / (120 × 10¹² FLOPs/s)
          ≈ 0.0000002796203 s
          = 0.0000002796203 s × 10⁶ µs/s
          ≈ 0.2796 µs
```

### B8. Ideal memory time

Divide B5's traffic by the given `600 GB/s` HBM rate:

```text
T_memory = 33,570,816 bytes / (600 × 10⁹ bytes/s)
         ≈ 0.00005595136 s
         = 0.00005595136 s × 10⁶ µs/s
         ≈ 55.9514 µs
```

### B9–B10. Lower bound and classification

The roofline model selects the slower resource time:

```text
lower bound = max(T_compute, T_memory)
            = max(0.2796 µs, 55.9514 µs)
            = 55.9514 µs
```

Memory time is about `55.9514 / 0.2796 ≈ 200×` compute time. Equivalently,
`≈1 FLOP/byte` is below the 200 ridge. The model says **memory-bound**.

## Part C — Prefill Projection, `M = 512`

```text
X [512 × 4096] × W [4096 × 4096] → Y [512 × 4096]

M = 512     prompt-token rows processed together
K = 4096    input features per row
N = 4096    output features per row
```

### C1. FLOPs

There are `512 × 4096` outputs, each a length-4096 dot product:

```text
FLOPs = 2 × M × K × N
      = 2 FLOPs/contribution
        × 512 rows
        × 4096 contributions/output
        × 4096 outputs/row
      = 2 × 512 × 4096 × 4096
      = 17,179,869,184 FLOPs
```

### C2. Weight bytes

The shape and datatype of `W` are unchanged. The assumption is one weight read
per call, not one read per row:

```text
weight bytes = K × N × 2 bytes/value
             = 4096 × 4096 × 2
             = 33,554,432 bytes
```

### C3. Input bytes

`X_prefill` contains `M × K` FP16 values:

```text
input values = 512 × 4096 = 2,097,152 values
input bytes  = 2,097,152 values × 2 bytes/value
             = 4,194,304 bytes
```

### C4. Output bytes

`Y_prefill` contains `M × N` FP16 values:

```text
output values = 512 × 4096 = 2,097,152 values
output bytes  = 2,097,152 values × 2 bytes/value
              = 4,194,304 bytes
```

### C5. Total modeled HBM traffic

```text
total bytes = weight read + input read + output write
            = 33,554,432 + 4,194,304 + 4,194,304
            = 41,943,040 bytes
```

The weight term stays fixed; input and output are 512 times their decode sizes.

### C6. Arithmetic intensity

Divide C1's work by C5's traffic:

```text
AI = 17,179,869,184 FLOPs / 41,943,040 bytes
   = 409.6 FLOPs/byte
```

The weights support work for all 512 rows, so each modeled HBM byte enables far
more work than during decode.

### C7. Ideal compute time

```text
T_compute = 17,179,869,184 FLOPs / (120 × 10¹² FLOPs/s)
          ≈ 0.0001431656 s
          = 0.0001431656 s × 10⁶ µs/s
          ≈ 143.1656 µs
```

### C8. Ideal memory time

```text
T_memory = 41,943,040 bytes / (600 × 10⁹ bytes/s)
         ≈ 0.0000699051 s
         = 0.0000699051 s × 10⁶ µs/s
         ≈ 69.9051 µs
```

### C9–C10. Lower bound and classification

```text
lower bound = max(T_compute, T_memory)
            = max(143.1656 µs, 69.9051 µs)
            = 143.1656 µs
```

Compute time is larger. Equivalently, `409.6 FLOPs/byte` exceeds the 200 ridge.
The model says **compute-bound**.

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

## Part D — Compare and Explain

### D1. Does weight traffic grow 512 times?

No. Both calls use the same `[4096 × 4096]` FP16 matrix, read once:

```text
weight traffic = 4096 × 4096 values × 2 bytes/value
               = 33,554,432 bytes in either call
```

Only `X` and `Y`, whose shapes contain `M`, grow with row count.

### D2. FLOP growth

All factors other than `M` cancel:

```text
prefill FLOPs / decode FLOPs
= (2 × 512 × 4096 × 4096) / (2 × 1 × 4096 × 4096)
= 512×
```

### D3. Byte growth

Use C5's and B5's totals:

```text
prefill bytes / decode bytes
= 41,943,040 / 33,570,816
≈ 1.2494×
```

It is only about `1.25×` because the large weight read is unchanged.

### D4. Arithmetic-intensity growth

Since `AI = FLOPs / bytes`:

```text
AI growth ≈ FLOP growth / byte growth
          ≈ 512 / 1.2494
          ≈ 409.8×
```

The difference from `409.6 / 1` is due to rounding decode's `0.9995` to 1.

### D5. Why the bound changes

```text
decode:  0.9995 < 200 → memory-bound
prefill: 409.6  > 200 → compute-bound
```

Prefill reuses the weights across enough rows to cross the GPU's balance point.

### D6. Lower-bound time per row

First, notice that **each row requires the same amount of mathematical work**.
Whether the row belongs to decode or prefill, it is multiplied by the same
`[4096 × 4096]` weight matrix:

```text
work per row = 2 × 1 × 4096 × 4096
             = 33,554,432 FLOPs
```

Prefill is more efficient per row because it does not execute 512 isolated
decode-sized calls. It forms one large matrix multiplication:

```text
[512 × 4096] × [4096 × 4096] → [512 × 4096]
```

The GPU can work on many output rows and columns in parallel. More importantly
for this model, the same weight tiles can be used for many input rows after
being fetched from HBM. The 512 rows therefore **share one modeled read of the
weight matrix**.

#### Compare the HBM traffic charged to each row

Decode has only one row, so that row bears the entire weight-read cost:

```text
decode traffic per row
= 33,570,816 total bytes / 1 row
= 33,570,816 bytes/row
```

Prefill spreads its one weight read across 512 rows:

```text
prefill traffic per row
= 41,943,040 total bytes / 512 rows
= 81,920 bytes/row
```

The prefill value can also be decomposed:

```text
amortized weight bytes per row = 33,554,432 / 512 = 65,536 bytes
input bytes per row            = 4096 × 2          =  8,192 bytes
output bytes per row           = 4096 × 2          =  8,192 bytes
                                                       ────────────
total modeled bytes per row                           = 81,920 bytes
```

“Amortized” means assigning each row an equal share of a cost paid once by the
whole call. It does not mean the weight matrix physically becomes smaller or
that each row mathematically uses only part of it. Every row still depends on
the full matrix; the implementation reuses weight data while processing many
rows together.

Thus, compared with decode, prefill performs the same `33,554,432 FLOPs` per
row while moving approximately:

```text
33,570,816 / 81,920 ≈ 409.8× fewer HBM bytes per row
```

That reuse is why arithmetic intensity rises and why the GPU can spend more of
its time performing arithmetic instead of waiting for weights from HBM.

#### Convert total-call time into amortized time per row

Now divide each modeled call lower bound by the number of rows it completes:

```text
decode  = 55.9514 µs / 1 row
        ≈ 55.9514 µs/token row

prefill = 143.1656 µs / 512 rows
        ≈ 0.2796 µs/token row
```

The prefill result is about `200×` lower per row:

```text
55.9514 / 0.2796 ≈ 200
```

In this idealized example, prefill crosses into the compute-bound region. Its
amortized time per row therefore approaches the time needed to perform one
row's arithmetic at the peak compute rate:

```text
33,554,432 FLOPs/row / (120 × 10¹² FLOPs/s)
≈ 0.2796 µs/row
```

This is not a universal claim that prefill is always exactly 200 times more
efficient. It follows from this matrix shape, row count, traffic model, and
hypothetical hardware rates.

### D7. Per-row time versus call latency

The phrase `0.2796 µs/token row` can be misleading if interpreted as “each
prefill token finishes after 0.2796 µs.” That is not what division by 512 means.
The 512 rows are processed together in one large operation and the modeled
operation completes after approximately `143.1656 µs`.

```text
decode call lower bound  = 55.9514 µs
prefill call lower bound = 143.1656 µs
```

So two statements are simultaneously true:

1. **The prefill call has higher total latency.** It completes much more work,
   so `143.1656 µs > 55.9514 µs`.
2. **The prefill call has better row throughput.** It completes 512 rows in
   `143.1656 µs`, rather than paying a separate `55.9514 µs` memory-dominated
   call for every row.

If the 512 rows were processed as 512 separate decode-like calls under this
model, their total lower-bound time would be:

```text
512 × 55.9514 µs ≈ 28,647.1 µs ≈ 28.65 ms
```

Processing them together has a modeled lower bound of only `143.1656 µs`
because the large matrix multiplication exposes parallel work and amortizes
the weight traffic. This comparison explains the efficiency gain; it is not a
claim that real prompt tokens should be processed as independent decode calls.

The metric interpretations are:

| Metric | Question it answers | Result here |
| --- | --- | --- |
| Call latency | How long until this entire operation finishes? | Decode is lower |
| Amortized time per row | How much total call time is charged to each completed row? | Prefill is lower |
| Row throughput | How many rows can the operation complete per unit time? | Prefill is higher |

The lesson is that **doing more total work can take longer while still using the
GPU much more efficiently**. Batching work improves reuse and parallelism, but
the individual request or call may wait longer for the larger batch to finish.

## Part E — Crossover

### What is `d`?

`d` is a new **shorthand symbol** for the width of this square projection. It
is not another tensor and it does not introduce another matrix dimension.

The original matrix shapes are:

```text
X [M × K] × W [K × N] → Y [M × N]
```

In this particular problem, both feature dimensions happen to have the same
value:

```text
K = 4096    input width
N = 4096    output width
```

Because `K` and `N` are equal, we can give their common value the shorter name
`d`:

```text
d = 4096
K = d
N = d

therefore:

X [M × d] × W [d × d] → Y [M × d]
```

You can read `d` here as **the model/projection width**. Using it makes the
algebra shorter while we solve for `M`, the unknown number of token rows.

This substitution would not be valid for a non-square projection where
`K ≠ N`. For example, if `W` had shape `[4096 × 11008]`, we would need to keep
`K = 4096` and `N = 11008` as separate symbols.

### E1. Work

Start with the general matrix-multiplication formula:

```text
FLOPs = 2MKN

Replace K with d and N with d because K = N = d:

FLOPs = 2 × M × d × d
      = 2Md²
```

The `d²` appears because `d × d = d²`. Substituting the actual width would
produce the same expression numerically:

```text
FLOPs = 2 × M × 4096 × 4096
```

### E2. Traffic

Every tensor uses 2 bytes per FP16 value:

```text
weight bytes = d × d × 2 = 2d²
input bytes  = M × d × 2 = 2Md
output bytes = M × d × 2 = 2Md

total bytes  = 2d² + 2Md + 2Md
             = 2d² + 4Md
```

### E3. Intensity as a function of row count

```text
AI(M) = FLOPs / bytes
      = 2Md² / (2d² + 4Md)

Divide the numerator and denominator by 2d:

AI(M) = Md / (d + 2M)
```

### E4. Solve for the ridge crossover

Set `AI(M)` equal to the 200 ridge and substitute `d = 4096`:

```text
4096M / (4096 + 2M) = 200
```

Multiply both sides by the denominator:

```text
4096M = 200(4096 + 2M)
      = 819,200 + 400M
```

Collect the `M` terms and divide:

```text
4096M - 400M = 819,200
3696M        = 819,200
M            = 819,200 / 3696
M            ≈ 221.645 rows
```

Rows must be whole numbers. Since 221 is below the crossover, round upward:

```text
smallest integer that reaches or exceeds the ridge = 222 token rows
```

## What This Model Does Not Prove

It does not prove measured latency or binding behavior. Real results depend on
achieved—not peak—rates, kernel shape efficiency, Tensor Core eligibility,
launch overhead, caches, tiling, fusion, precision, layout, extra traffic,
concurrency, and the rest of the Transformer layer. This calculation creates a
falsifiable hypothesis; profiling tests it.
