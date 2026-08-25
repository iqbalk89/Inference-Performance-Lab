# Problem 01 Worksheet — Transformer Projection Cost

## Before Calculating

In your own words:

- What does `X` represent?
- What does `W_Q` represent?
- What does one row of `W_Q` represent mechanically?
- What does `Q` represent?

## Shape Ledger

| Tensor | Shape | Meaning of rows | Meaning of columns |
| --- | --- | --- | --- |
| `X` | | | |
| `W_Q` | | | |
| `Q` | | | |

Inner-dimension compatibility check:

```text

```

## 1. Parameter Count

Symbolic formula:

```text

```

Substitution and exact result:

```text

```

## 2. Weight Data Read

Exact bytes:

```text

```

Decimal MB:

```text

```

Binary MiB:

```text

```

## 3. Compute

Identify dimensions:

```text
M =
K =
N =
```

Exact FLOPs:

```text

```

MFLOPs:

```text

```

## Assumption Ledger

| Component | Included or excluded? | Reason |
| --- | --- | --- |
| Read `W_Q` | | |
| Read `X` | | |
| Write `Q` | | |
| Bias | | |
| Kernel launch | | |
| K and V projections | | |
| Attention calculation | | |
| Weight-cache reuse | | |

## Sanity-Check Answers

1. Why `4096 × 4096` parameters?

2. Why must the inner dimensions match?

3. MAC contributions per output value:

4. Independent total-MAC calculation:

5. Why are FLOPs and weight bytes numerically similar here?

## Final Answer

| Quantity | Exact result | Approximation |
| --- | ---: | ---: |
| Parameters | | |
| Weight bytes | | |
| Decimal MB | | |
| Binary MiB | | |
| FLOPs | | |
| MFLOPs | | |

## Reflection

- Which calculation was least intuitive?
- Which unit conversion was easiest to get wrong?
- What would change if batch size were 8?
- What would change if the weights used one byte per value?

