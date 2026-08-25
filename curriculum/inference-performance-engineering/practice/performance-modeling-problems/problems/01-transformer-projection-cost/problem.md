# Problem 01 — What Does One Transformer Projection Cost?

## Scenario

Suppose we are running a simplified Transformer with:

| Quantity | Value |
| --- | ---: |
| Hidden dimension, `d_model` | 4,096 |
| Weight format | FP16 |
| Bytes per FP16 value | 2 bytes |
| Batch size, `B` | 1 |
| New token positions processed | 1 |
| Inference phase | Decode |

We are processing one new token for one sequence. The hidden state entering the
attention layer is therefore a single row:

```text
X shape: [1 × 4096]
```

To generate the query vector, the model performs the projection:

```text
Q = X W_Q
```

with:

```text
X   shape: [1 × 4096]
W_Q shape: [4096 × 4096]
Q   shape: [1 × 4096]

[1 × 4096] [4096 × 4096] → [1 × 4096]
```

Assume the GPU cannot reuse the weights from an on-chip cache, so all of `W_Q`
must be read from HBM for this operation.

## Given Formula

For matrix multiplication:

```text
[M × K] [K × N] → [M × N]
```

approximate compute is:

```text
FLOPs ≈ 2MKN
```

The factor `2` counts approximately one multiplication and one addition for
each multiply-accumulate contribution.

## Your Tasks

Calculate:

1. How many parameters are in `W_Q`?
2. How many bytes of `W_Q` must be read from HBM?
3. Express the weight read in both decimal MB and binary MiB.
4. Approximately how many FLOPs does the matrix multiplication require?
5. Express the result in both FLOPs and MFLOPs.

## Required Work

Do not provide only final numbers. Show:

```text
parameter formula
→ substituted values
→ exact parameter count

weight-byte formula
→ substituted values with bytes/value
→ exact bytes
→ MB conversion
→ MiB conversion

FLOP formula
→ identify M, K, and N
→ substituted values
→ exact FLOPs
→ MFLOP conversion
```

## Assumptions to State

Identify whether each item is included or excluded:

- Reading `W_Q`
- Reading the input row `X`
- Writing the output row `Q`
- Bias addition
- Kernel-launch overhead
- Other attention projections
- Attention-score computation
- GPU cache reuse

## Sanity Checks

Answer without opening the solution:

1. Why does `W_Q` contain `4096 × 4096` parameters rather than only `4096`?
2. Why does the inner dimension `4096` have to match?
3. For this `M=1` decode projection, how many multiply-accumulate contributions
   are performed per output value?
4. There are `4096` output values. Use your answer to Question 3 to independently
   recover the total multiply-accumulate count.
5. Why are the approximate FLOP count and weight-byte count numerically similar
   in this particular FP16, `M=1` example?

## Record Your Work

Use [worksheet.md](worksheet.md). Do not open
[answer.md](answer.md) until the worksheet contains a complete attempt.

