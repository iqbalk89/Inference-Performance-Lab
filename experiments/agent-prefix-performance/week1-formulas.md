# Week 1 Formula Page

This page collects the calculations used by the Week 1 baseline so the
experiment, the walkthrough, and the findings note all point at the same math.
It is the page to read before interpreting the raw benchmark output.

## What To Calculate Before Running

Before you trust a result, calculate these quantities for the chosen model and
workload:

1. model parameter bytes;
2. hidden size, head counts, and head dimension;
3. the logical shapes of Q, K, V, score tensor `S`, probability tensor `P`,
   and attention output/context `C`;
4. expected KV-cache bytes for the prompt length; and
5. output-token throughput and cached-versus-uncached speedup from the measured
   times.

## Core Equations

### 1. Model storage

```text
parameter_bytes = parameter_count × bytes_per_parameter
```

For FP16:

```text
bytes_per_parameter = 2
```

### 2. Attention projection shapes

For one batch item with prompt length `T`, query heads `H_q`, KV heads `H_kv`,
and head dimension `D`:

```text
Q = [B, H_q, T, D]
K = [B, H_kv, T, D]
V = [B, H_kv, T, D]
```

The grouped-query attention score tensor is:

```text
S = Q × K^T = [B, H_q, T, T]
```

The probability tensor after softmax has the same logical shape as `S`:

```text
P = softmax(S) = [B, H_q, T, T]
```

The output/context tensor is:

```text
C = P × V = [B, H_q, T, D]
```

### 3. KV-cache bytes

For equal sequence length across the batch:

```text
KV bytes = 2 × B × L × H_kv × T × D × bytes_per_element
```

Where:

```text
2 = key tensor + value tensor
B = batch size
L = transformer layers
H_kv = key/value heads
T = cached token positions
D = head dimension
```

### 4. Throughput and ratios

```text
tokens/s = output_tokens / seconds
speedup   = no-cache time / cached time
```

For the cached decode loop in Week 1, the model emits one token from prefill
and then produces the rest autoregressively. That is why the cached decode
throughput in the benchmark divides the decode-only time by `new_tokens - 1`
when a separate decode rate is reported.

### 5. Prompt scaling

If batch size, layer count, head counts, and dtype stay fixed, then KV-cache
bytes grow linearly with prompt length:

```text
KV bytes ∝ T
```

That linear growth is the reason a 2,048-token prompt uses about 4× the cache
space of a 512-token prompt.

## Qwen2.5-1.5B Week 1 Constants

For the model used in the baseline:

```text
B = 1
L = 28
H_q = 12
H_kv = 2
D = 128
bytes_per_element = 2  # FP16
hidden_size = H_q × D = 1536
group size = H_q / H_kv = 6
```

So the logical shapes during 512-token prefill are:

```text
Q = [1, 12, 512, 128]
K = [1, 2, 512, 128]
V = [1, 2, 512, 128]
S = [1, 12, 512, 512]
P = [1, 12, 512, 512]
C = [1, 12, 512, 128]
```

And the expected KV-cache bytes are:

```text
KV bytes = 2 × 1 × 28 × 2 × 512 × 128 × 2
         = 14,680,064 bytes
         = 14 MiB
```

For a 2,048-token prompt:

```text
KV bytes = 2 × 1 × 28 × 2 × 2048 × 128 × 2
         = 58,720,256 bytes
         = 56 MiB
```

## What To Compare Against The Run

When the JSON result comes back, compare:

1. `parameter_bytes` against the weight-size estimate;
2. `initial_kv_cache.total_bytes` against the KV-cache formula;
3. `logits_shape` against `[B, T, vocab_size]`;
4. the cached/uncached time ratio against `no-cache time / cached time`; and
5. the prompt-length sweep against linear cache growth and slower prefill.

