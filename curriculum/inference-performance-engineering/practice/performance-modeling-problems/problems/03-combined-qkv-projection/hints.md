# Problem 03 Hints

## Hint 1 — output width

Q, K, and V each have width 4096. Concatenating them side by side gives an
output width of `3 × 4096`.

## Hint 2 — fused matrix

If `X` is `[M × K]` and the output is `[M × N]`, then the fused weight is
`[K × N]`. Here `K = 4096` and `N = 12288`.

## Hint 3 — bytes

The fused weight has `4096 × 12288` FP16 values. The output contains three
`[M × 4096]` tensors, so its byte count is three times one projection's output
size.

## Hint 4 — FLOPs

Use one matrix multiplication:

```text
2 × M × 4096 × 12288
```

Do not multiply by three again after using the fused output width.

## Hint 5 — fusion interpretation

Fusion changes how the work is scheduled and stored. It does not eliminate the
three logical projections or the need to produce Q, K, and V.
