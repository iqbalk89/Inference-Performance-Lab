# Lesson 03 — How Transformer Workloads Map to GPUs

**Prerequisites:** Lessons 01 and 02  
**Purpose:** Connect an understood transformer algorithm to GPU-oriented
workload shapes  
**Expected study time:** 90–150 minutes

## Why This Is a Separate Lesson

Lesson 02 explained what transformer inference does. This lesson asks a
different question:

> Which parts expose structured parallel work, which parts remain sequential,
> and what data must move or remain stored?

Separating these questions prevents phrases such as “attention runs in
parallel” from substituting for an understanding of attention itself.

## 1. Translate Algorithm Steps Into Operations

For a simplified decoder-only block:

```text
hidden states X
   ├── linear projections ──▶ Q, K, V
   ├── Q × Kᵀ ──────────────▶ attention scores
   ├── mask + softmax ──────▶ attention weights
   ├── weights × V ─────────▶ contextual states
   └── feed-forward matrices and activations
```

These steps contain different computational patterns:

| Operation | Available structure | Important dependency |
| --- | --- | --- |
| Linear projection | Many output cells and tiles | Each output cell must combine its shared-dimension products |
| Q × Kᵀ | Many query-key score cells | Q and K must exist first |
| Causal mask | Independent score-cell updates | Score matrix must exist conceptually |
| Softmax | Work across each row | Normalization needs row statistics and a row sum |
| Attention weights × V | Many output cells | Weights and V must exist |
| Activation | Similar operation across many elements | Input elements must exist |

The word **parallel** never means “dependency-free.” It means enough independent
or regularly structured work exists within the dependency boundaries.

## 2. Why Matrix Multiplication Is GPU-Friendly

Suppose:

```text
X [256 token rows, 4096 input features]
× W [4096 input features, 4096 output features]
→ Y [256 token rows, 4096 output features]
```

`Y` contains:

```text
256 × 4096 = 1,048,576 output cells
```

Each output cell is a dot product of length 4,096. Output regions can be divided
into tiles. A GPU kernel describes logical workers for those regions, and finite
hardware schedules the work over SMs in waves.

```text
large logical output grid
          │ divide into tiles
          ▼
[tile 0][tile 1][tile 2][tile 3] ...
          │ finite GPU resources
          ▼
SMs execute resident thread groups and later waves
```

This creates opportunities for:

- High aggregate arithmetic throughput
- Reuse of input values within on-chip storage
- Many ready thread groups for latency hiding
- Specialized matrix multiply-accumulate hardware

It does not imply that every logical output cell executes at the same physical
instant.

## 3. Prefill Workload Shape

During prefill, many prompt-token rows are available:

```text
Xprefill [prompt positions, hidden size]
```

For a 128-token prompt at batch size 1:

```text
Xprefill [128, 4096] × W [4096, 4096] → Y [128, 4096]
```

The same large weight matrix contributes to 128 token rows in that operation.
The token-position dimension gives kernels a relatively large grid of work.

Attention also creates score matrices across prompt positions. A causal mask
restricts which cells influence each row, but all known prompt rows can still be
formed using large tensor operations.

Prefill therefore often offers:

- More token-position parallelism per operation
- Better reuse of weights across token rows
- Larger matrix shapes that can use GPU resources effectively

This does not mean prefill has little work. Long prompts can require substantial
total computation, especially in attention.

## 4. Batch-One Decode Workload Shape

During one batch-one decode iteration, only one newest token row is processed:

```text
Xdecode [1, 4096] × W [4096, 4096] → Y [1, 4096]
```

The weight matrix remains large, but there is only one token row in this
simplified operation. The model then selects one token before the next decode
iteration can have its complete input.

```text
decode step 1: parallel tensor work → choose token 1
                                           │ dependency
decode step 2:                             parallel tensor work → choose token 2
```

Decode is therefore:

- **Parallel inside an iteration:** matrix cells, tiles, heads, and other
  operation regions expose parallel work.
- **Sequential across generated tokens:** the next selected token is an input to
  the following iteration.

Small-batch decode often has less work available per weight read and per kernel
launch. It is commonly sensitive to memory bandwidth and launch overhead, but
that is a hypothesis to verify for a particular model, batch, and GPU.

## 5. How Batching Changes Decode

If four independent requests decode together:

```text
Xdecode [4 newest-token rows, 4096 features]
× W [4096, 4096]
→ Y [4 rows, 4096 output features]
```

Each row belongs to a different request. The requests do not attend to one
another; batching packages compatible independent work into larger operations.

Potential effect:

```text
larger batch → more work per weight use → higher aggregate throughput
```

Possible tradeoff:

```text
waiting to form or serve a batch → greater per-request latency
```

Throughput and latency must therefore be measured separately.

## 6. The KV Cache as a GPU-Memory Workload

The KV cache saves recomputation, but the cache must reside somewhere and be
read during decode.

Conceptually, cache capacity grows with:

```text
layers × batch × KV heads × sequence positions × head dimension × bytes/value
```

As the sequence grows:

- More K/V rows occupy memory.
- A new query attends over a longer history.
- More cached data may need to be read.
- Available capacity for larger batches or more requests decreases.

The KV cache therefore links an algorithmic optimization to hardware concerns:
capacity, bandwidth, layout, and reuse.

## 7. Offload Overhead and the Measured Boundary

GPU arithmetic time is only one part of an end-to-end path:

```text
CPU preparation → transfer if required → launch/queue → GPU execution
→ synchronization → return required data → CPU post-processing
```

Suppose a tiny operation takes `20 μs` entirely on the CPU. A hypothetical GPU
path might be:

```text
host-to-device preparation/transfer: 100 μs
kernel submission:                    15 μs
GPU arithmetic:                       10 μs
required result returned:            100 μs
complete GPU path:                    225 μs
```

Comparing `20 μs` of CPU work only with `10 μs` of GPU arithmetic would produce
the wrong conclusion. The relevant comparison is `20 μs` versus the complete
`225 μs` path.

Large transformer operations can amortize fixed overhead and often keep weights
and intermediate tensors in device memory across many operations. Efficient
inference systems avoid unnecessary CPU/GPU transfers and synchronization. The
principle is:

> Measure the complete boundary relevant to the user or system objective, then
> explain the GPU portion within that boundary.

## 8. CPU and GPU Responsibilities

A simplified request path is:

```text
CPU/service side                       GPU/device side
----------------                       ---------------
receive request
validate and tokenize
construct input state       ────────▶  execute tensor kernels
manage scheduling                       read weights and cache
interpret required result   ◀────────  produce logits/selected data
stream response
```

Implementations differ. Token selection and more generation logic may remain on
the GPU to reduce synchronization. The key performance principle is to avoid
unnecessary movement and waiting across the host-device boundary.

## 9. Questions to Carry Into Profiling

When you later capture a real trace, ask:

1. Is this interval prefill or decode?
2. What are the batch, prompt-length, and generated-length shapes?
3. Are GPU kernels separated by CPU or synchronization gaps?
4. Are matrix kernels large enough to use the device effectively?
5. Is time dominated by computation, memory movement, or launch overhead?
6. How much VRAM is occupied by weights and the KV cache?
7. Does batching improve throughput at an acceptable latency cost?

## 10. Common Mapping Mistakes

### “Transformers use matrices, so every operation is compute-bound.”

Elementwise operations, reductions, cache reads, launches, and synchronization
have different performance characteristics. Workload shape and measured
evidence decide the limiter.

### “Causal means prefill must process prompt tokens one complete pass at a time.”

Causality restricts information flow. A causal mask allows kernels to process
known prompt rows in large tensor operations while blocking forbidden score
cells.

### “Decode is sequential, so the GPU cannot help.”

Generated tokens are sequential across iterations, but each iteration still
contains large tensor operations with internal parallelism.

### “The KV cache makes decode constant-cost.”

It avoids recreating old K/V projections, but cache capacity and the history
read by attention grow with sequence length.

### “High GPU utilization proves efficient inference.”

A coarse utilization sample does not show whether the right resources are busy,
whether kernels are efficient, or whether latency and throughput objectives are
met.

### “If the GPU arithmetic is faster, offloading is faster.”

The complete path includes preparation, transfer, launch, synchronization, and
post-processing. Compare the same end-to-end boundary.

## Lesson 03 Completion Gate

Explain without notes:

1. Why large matrix outputs expose parallel work.
2. Why logical work is scheduled over finite hardware in waves.
3. Why prefill and batch-one decode present different operation shapes.
4. Why decode can be parallel inside but sequential outside.
5. How batching changes throughput opportunity and latency.
6. Why the KV cache affects both capacity and memory traffic.

Next: [Lesson 04 — The GPU Execution Model](../04-gpu-execution-model/).
