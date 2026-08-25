# Problem 01 Answer — Transformer Projection Cost

For the animated derivation, run the
[Problem 01 Manim visual solution](../../visual-solutions/README.md).

## 1. Parameters in `W_Q`

Each matrix position contains one parameter:

```text
parameters = rows × columns
           = 4096 × 4096
           = 16,777,216 parameters
```

## 2. Weight Data Read from HBM

FP16 uses 2 bytes per stored value:

```text
weight bytes = parameters × bytes per parameter
             = 16,777,216 × 2 bytes
             = 33,554,432 bytes
```

Decimal megabytes use `1 MB = 1,000,000 bytes`:

```text
33,554,432 / 1,000,000
= 33.554432 MB
≈ 33.55 MB
```

Binary mebibytes use `1 MiB = 1,048,576 bytes`:

```text
33,554,432 / 1,048,576
= 32 MiB
```

## 3. Matrix-Multiplication FLOPs

From:

```text
[M × K] [K × N] → [M × N]
[1 × 4096] [4096 × 4096] → [1 × 4096]
```

we identify:

```text
M = 1
K = 4096
N = 4096
```

Therefore:

```text
FLOPs ≈ 2MKN
      = 2 × 1 × 4096 × 4096
      = 33,554,432 FLOPs
```

Using decimal MFLOPs:

```text
33,554,432 / 1,000,000
= 33.554432 MFLOPs
≈ 33.55 MFLOPs
```

## Final Results

| Quantity | Result |
| --- | ---: |
| Parameters in `W_Q` | 16,777,216 |
| Weight data read | 33,554,432 bytes |
| Weight data read, decimal | approximately 33.55 MB |
| Weight data read, binary | exactly 32 MiB |
| Matrix-multiplication work | approximately 33,554,432 FLOPs |
| Matrix-multiplication work | approximately 33.55 MFLOPs |

## Sanity-Check Explanations

1. `W_Q` maps every one of 4,096 input features to every one of 4,096 output
   features. One output feature needs a column of 4,096 weights, and there are
   4,096 output features.
2. Each output dot product pairs the 4,096 values in the input row with 4,096
   weights in one column of `W_Q`. Mismatched lengths could not be paired
   element by element.
3. Every output value receives 4,096 multiply-accumulate contributions.
4. `4,096 outputs × 4,096 MAC contributions/output = 16,777,216 MACs`.
   Counting a multiply and add as two FLOPs gives `33,554,432 FLOPs`.
5. With `M=1`, each of the `K × N` FP16 weights contributes approximately one
   multiply and one add: two FLOPs per weight and two bytes per weight. The
   numeric totals therefore match. This does not make FLOPs and bytes the same
   physical quantity.

## Model Boundary

This simplified answer includes the full HBM read of `W_Q` and the projection's
multiply-add work. It excludes input reads, output writes, bias, kernel launch,
K/V projections, attention, other layers, and cache reuse. It is an operation
model—not yet a complete latency prediction.
