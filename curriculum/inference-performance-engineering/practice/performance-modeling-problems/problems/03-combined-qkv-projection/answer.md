# Problem 03 Answer — Combined QKV Projection

## Shapes and parameters

One projection uses:

```text
W_Q, W_K, or W_V: [4096 × 4096]
parameters = 4096 × 4096 = 16,777,216 values
```

The fused matrix concatenates the three output widths:

```text
W_QKV: [4096 × (4096 + 4096 + 4096)]
     = [4096 × 12288]
parameters = 4096 × 12288 = 50,331,648 values
```

That is exactly three times one projection's parameter count.

For either workload:

```text
Q, K, V: [M × 4096]
[Q K V]: [M × 12288]
```

## Decode: M = 1

```text
FLOPs = 2 × 1 × 4096 × 12288
      = 100,663,296 FLOPs

Weight bytes = 50,331,648 values × 2 bytes/value
             = 100,663,296 bytes

Input bytes = 1 × 4096 × 2
            = 8,192 bytes

Q/K/V output bytes = 1 × 12288 × 2
                   = 24,576 bytes

Total bytes = 100,663,296 + 8,192 + 24,576
            = 100,696,064 bytes

Arithmetic intensity = 100,663,296 ÷ 100,696,064
                     ≈ 0.9997 FLOPs/byte

Compute time = 100,663,296 ÷ (120 × 10¹²) × 10⁶
             ≈ 0.8389 µs

Memory time = 100,696,064 ÷ (600 × 10⁹) × 10⁶
            ≈ 167.8268 µs

Roofline lower bound = max(0.8389, 167.8268)
                    ≈ 167.8268 µs
```

Decode is HBM-bandwidth-bound under this model.

## Prefill: M = 512

```text
FLOPs = 2 × 512 × 4096 × 12288
      = 51,539,607,552 FLOPs

Weight bytes = 100,663,296 bytes
Input bytes = 512 × 4096 × 2
            = 4,194,304 bytes
Q/K/V output bytes = 512 × 12288 × 2
                   = 12,582,912 bytes

Total bytes = 100,663,296 + 4,194,304 + 12,582,912
            = 117,440,512 bytes

Arithmetic intensity = 51,539,607,552 ÷ 117,440,512
                     ≈ 438.8571 FLOPs/byte

Compute time = 51,539,607,552 ÷ (120 × 10¹²) × 10⁶
             ≈ 429.4967 µs

Memory time = 117,440,512 ÷ (600 × 10⁹) × 10⁶
            ≈ 195.7342 µs

Roofline lower bound = max(429.4967, 195.7342)
                    ≈ 429.4967 µs
```

Prefill is FP16-compute-bound under this model.

## Interpretation

Fusion makes one wider matrix multiplication. It allows an implementation to
load or stage `X` once for the combined operation and produce a contiguous
QKV result. The logical results are still three different tensors with three
different roles in attention.

Fusion does not remove the weight values: the fused matrix still contains all
three projection matrices. It also does not remove the Q/K/V output values.

This is not yet attention. No `QKᵀ`, softmax, causal mask, or multiplication by
`V` has been modeled. It only creates the inputs that attention consumes.

With GQA or MQA, Q usually retains more heads than K and V. Therefore the
fused output width becomes:

```text
Q width + K width + V width
```

rather than `3 × d_model`, reducing K/V parameters and output traffic.
