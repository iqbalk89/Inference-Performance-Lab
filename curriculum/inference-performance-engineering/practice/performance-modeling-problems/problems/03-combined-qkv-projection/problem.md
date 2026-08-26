# Problem 03 — What does a combined QKV projection cost?

## Why this problem matters

The previous problem modeled one dense projection, `Y = XW`. A Transformer
attention layer normally creates three related projections:

```text
Q = XW_Q     K = XW_K     V = XW_V
```

Implementations commonly fuse them into one larger matrix multiplication:

```text
[Q K V] = XW_QKV
```

This problem models the arithmetic and HBM traffic for that operator before
you model attention scores or the KV cache.

## Model setup

Use:

```text
d_model = 4096
number of attention heads = 32
head dimension = 128
FP16 = 2 bytes/value
```

### Short primer: what is an attention head?

An attention head is a **logical slice of the model's feature vector**. It is
not a physical GPU component and it is not a separate processor. The model
starts with one hidden vector containing `d_model` values and divides that
width into several smaller vectors so several attention calculations can be
performed in parallel.

Here, the relationship is:

```text
number of heads × values per head = total feature width
32 heads × 128 values/head = 4096 values = d_model
```

So one row of `Q`, `K`, or `V` has 4,096 values, which can be viewed as 32
smaller rows of 128 values each:

```text
one Q row: [4096 values]
           └─ head 0: [128] ─┬─ head 1: [128] ─┬─ ... ─┬─ head 31: [128]
```

For this problem, you only need the dimensional fact above. You do **not** yet
need to understand how a head computes attention scores. That comes in the
attention lesson. The heads simply explain why `4096` can also be written as
`32 × 128` and why a Q/K/V row may later be reshaped to `[32, 128]`.

Check the head relationship:

```text
32 heads × 128 values/head = 4096 values
```

Compare:

```text
Decode:  X = [1 × 4096]
Prefill: X = [512 × 4096]
```

Each of `Q`, `K`, and `V` has shape `[M × 4096]`. The fused output therefore
has shape `[M × 12288]`, and the fused weight matrix has shape
`[4096 × 12288]`.

Use the simplified traffic model:

```text
read W_QKV once + read X once + write Q, K, and V once
```

Ignore cache hits, workspace, launch overhead, and extra framework traffic.

## Tasks

For both `M = 1` and `M = 512`, calculate:

1. The shape and parameter count of one Q, K, or V weight matrix.
2. The shape and parameter count of the fused `W_QKV` matrix.
3. The shapes of Q, K, V, and the concatenated output.
4. QKV FLOPs using `2MKN`.
5. Weight-read bytes, input-read bytes, and output-write bytes.
6. Total modeled HBM traffic.
7. Arithmetic intensity.
8. Ideal compute and memory time using the Problem 02 hardware assumptions:
   `120 TFLOP/s` and `600 GB/s`.
9. The roofline lower bound and bottleneck.

Then answer:

- Why does fusing Q, K, and V change the matrix shape but not the mathematical
  meaning of the three projections?
- Why does the fused operation reuse the input `X` conceptually?
- What traffic is reduced by fusion in this simplified model, and what traffic
  is not reduced?
- Why is this still not an attention model?
- In a model using GQA or MQA, which output widths would change?

## Required table

| Quantity | Decode, `M=1` | Prefill, `M=512` | Unit |
| --- | ---: | ---: | --- |
| One projection parameters | | | values |
| Fused QKV parameters | | | values |
| QKV FLOPs | | | FLOPs |
| Weight read | | | bytes |
| Input read | | | bytes |
| Q/K/V output write | | | bytes |
| Total traffic | | | bytes |
| Arithmetic intensity | | | FLOPs/byte |
| Compute time | | | µs |
| Memory time | | | µs |
| Roofline lower bound | | | µs |
| Bottleneck | | | — |

State every assumption and distinguish the fused operator from the attention
score/value operations that come afterward.
