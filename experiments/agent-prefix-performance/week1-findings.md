# Week 1 Findings — Direct PyTorch Baseline

**Run date:** September 2, 2026 UTC  
**Hardware:** NVIDIA A10, 23,028 MiB usable VRAM  
**Software:** PyTorch 2.7.0 / CUDA 12.8 / Transformers 4.52.4  
**Model:** `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP16  
**Workload:** batch 1, greedy decoding, one synthetic stable agent-context
prompt, 1 warmup, 5 measured repetitions per matrix case

Raw JSON results and the `nvidia-smi` sample are preserved locally under the
ignored `benchmark-results/agent-prefix-performance/week1/` directory.

## Results

Times are median milliseconds. Cached output-token throughput includes prefill;
the uncached path has no meaningful separate prefill/decode split because every
step recomputes the full sequence.

| Prompt | New tokens | Cached prefill | Cached total | Cached tok/s | No-cache total | No-cache tok/s | No-cache / cached |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 30.2 | 621.6 | 51.5 | 1,301.1 | 24.6 | 2.09× |
| 512 | 128 | 30.5 | 2,427.6 | 52.7 | 5,385.9 | 23.8 | 2.22× |
| 2,048 | 32 | 134.1 | 719.9 | 44.5 | 4,588.5 | 7.0 | 6.37× |
| 2,048 | 128 | 134.7 | 2,519.3 | 50.8 | 19,652.5 | 6.5 | 7.80× |

The first discovery run was repeated once at 512/32; it measured 590.3 ms
cached total and 1,295.6 ms without cache. It is retained as a separate raw
file but is not used in the table above.

## Calculations Used

The findings above are backed by these explicit calculations:

```text
parameter_bytes = parameter_count × bytes_per_parameter
```

For the chosen FP16 model:

```text
bytes_per_parameter = 2
```

The logical attention shapes for batch 1 and prompt length `T` are:

```text
Q = [1, 12, T, 128]
K = [1, 2, T, 128]
V = [1, 2, T, 128]
S = [1, 12, T, T]
P = [1, 12, T, T]
C = [1, 12, T, 128]
```

The KV-cache size is:

```text
KV bytes = 2 × B × L × H_kv × T × D_head × bytes_per_element
```

With the Week 1 constants:

```text
B = 1
L = 28
H_kv = 2
D_head = 128
bytes_per_element = 2
```

So:

```text
T = 512  → 14,680,064 bytes = 14 MiB
T = 2048 → 58,720,256 bytes = 56 MiB
```

Throughput and comparison metrics use:

```text
tokens/s = output_tokens / seconds
speedup   = no-cache time / cached time
```

For cached decode, the first token is produced by prefill, so the separate
decode-rate estimate uses the remaining `new_tokens - 1` decode steps.

## Theoretical memory versus actual memory

The table below puts the formula result beside the value observed by the
benchmark. The KV-cache rows are directly comparable: the benchmark reports
the bytes allocated for the initial cache, before decode adds new positions.

| Quantity | Theoretical calculation | Actual result | How to interpret the difference |
|---|---:|---:|---|
| KV cache, `T=512` | `2 × 1 × 28 × 2 × 512 × 128 × 2` = **14,680,064 bytes (14 MiB)** | **14,680,064 bytes (14 MiB)** | Exact match; the cache contains one K and one V tensor for each of 28 layers. |
| KV cache, `T=2,048` | `2 × 1 × 28 × 2 × 2048 × 128 × 2` = **58,720,256 bytes (56 MiB)** | **58,720,256 bytes (56 MiB)** | Exact match; increasing `T` by 4× increases cache storage by 4×. |
| Peak cached allocation, `T=512` | **≈3.09 GB weights + 14 MiB KV = ≈3.10 GB**, before temporary tensors | **~3.34 GB** PyTorch allocator peak | The remaining ~0.24 GB is activations, logits, inputs, CUDA workspaces, and allocator overhead. |
| Peak cached allocation, `T=2,048` | **≈3.09 GB weights + 56 MiB KV = ≈3.15 GB**, before temporary tensors | **~4.14–4.27 GB** PyTorch allocator peak | The additional peak is primarily longer-sequence activations and attention work; it should not be attributed to KV alone. |

The first two rows are a validation of the KV formula. The last two rows are
end-to-end allocator measurements, so their theoretical value is best treated
as a decomposition (what should contribute) rather than as an exact prediction
of the peak. In particular, `nvidia-smi` process memory, PyTorch allocated
memory, and PyTorch reserved memory measure different things.

## Tensor and memory evidence

For a 512-token prompt, the first layer's K and V tensors were each:

```text
[1, 2, 512, 128]   # batch, KV heads, positions, head dimension
```

There were 56 tensors total: one K and one V for each of 28 layers. The initial
cache was 14,680,064 bytes (14 MiB). For a 2,048-token prompt it was
58,720,256 bytes (56 MiB), exactly 4× larger as expected from linear growth in
the position dimension.

Peak cached allocator readings were approximately 3.34 GB for the 512-token
cases and 4.14–4.27 GB for the 2,048-token cases. PyTorch allocated and reserved
memory are both recorded in every JSON result; they are not interchangeable with
the `nvidia-smi` process view.

## Observations

1. **Prefill scales with prompt length.** Increasing the prompt from 512 to
   2,048 tokens increased median prefill from about 30 ms to 134 ms (4.4×),
   while the cached decode portion for 128 outputs stayed near 2.4 seconds.
2. **Cached decode scales mainly with output length.** At a fixed prompt length,
   128 outputs took about four times the decode time of 32 outputs. Each step
   processes one new token and reuses previous K/V tensors.
3. **The KV cache makes long-prefix generation viable.** At 2,048/128, caching
   was 7.8× faster than recomputing the entire sequence. At 512/32 the advantage
   was only 2.1×, because the repeated prefix is shorter.
4. **No-cache work compounds with sequence length.** Uncached throughput fell
   from 24.6 tok/s at 512/32 to 6.5 tok/s at 2,048/128. This is the expected
   cost of repeatedly processing an increasingly long prefix.
5. **GPU utilization is bursty for this single request.** In a representative
   512/32 cached run, `nvidia-smi` sampled 3.6–3.7 GB used and 57–59% GPU
   utilization during the measured interval, with idle samples before and after
   the short workload. This is not a serving-throughput measurement.

## Limitations and Week 2 questions

- This is a single-request PyTorch baseline, not a concurrent serving test, so
  it says nothing about P95/P99 under load.
- The prompt is synthetic and exactly length-controlled; it does not yet model
  realistic changing agent turns or prefix-cache reuse across requests.
- The run used Transformers' direct model call and CUDA events. It did not yet
  include NVTX ranges, PyTorch Profiler, or Nsight Systems traces.
- The no-cache comparison is intentionally pedagogical and is not a production
  configuration.

Week 2 should answer:

1. Which operators dominate the 134 ms long-prompt prefill?
2. Why are decode steps bursty at this batch size, and where are CPU gaps or
   synchronization points?
3. How much of the peak allocation is weights, activations, allocator reserve,
   and KV cache?
4. Which kernels differ between prefill and one-token decode?
