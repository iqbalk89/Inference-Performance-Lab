# Lesson 01 Visual Lab Worksheet

Name:  
Date:  

Write predictions before manipulating the station. Use complete sentences for
explanations.

## Station 1 — Dependency Scheduler

1. Before running it, which operations should be ready first, and why?
2. After both loads finish, which operations become independent?
3. What must finish before the final store can begin?
4. Explain why “independent” does not necessarily mean “executing at exactly the
   same instant.”

## Station 2 — Tensor Explorer

Set `batch = 2`, `tokens = 4`, `features = 3`, and `dtype = FP16`.

1. What are the tensor's rank and shape?
2. How many token vectors exist?
3. How many scalar values exist?
4. How many bytes of raw storage are required?
5. Select index `[1, 2, 1]`. Explain what each index chooses.
6. Is dtype another axis? Explain.

## Station 3 — Matrix Microscope

Before selecting the highlighted answer, calculate `Y[1,2]` from the displayed
`X` and `W` matrices.

1. Selected row from `X`:
2. Selected column from `W`:
3. Products:
4. Sum and final value:
5. Why must the inner dimensions match?
6. Which other output cells could be calculated independently of `Y[1,2]`?

## Station 4 — Finite GPU Waves

Set `logical tiles = 13` and `available slots = 4`.

1. Predict the number of required waves.
2. Record the displayed result.
3. Does the simulation imply that one tile equals one CUDA core? Why not?
4. Explain the difference between described parallel work and physical
   simultaneous capacity.

## Station 5 — Attention and the Causal Mask

Select query position 1 (`sky`).

1. Which source positions are permitted?
2. Which are future positions relative to query position 1?
3. What happens to a masked score before softmax, conceptually?
4. What weight does that source receive after softmax?
5. In one sentence each, state the mechanical role of Q/K and of V.
6. Explain how all prompt tokens can be known to the server while some remain
   invisible to an earlier query row.

Then select query position 3 (`blue`). Why are all four positions permitted?

## Station 6 — Prefill, Decode, and KV-Cache Growth

1. Before prefill, which token IDs are known and which position is unknown?
2. What two kinds of numerical rows does prefill save?
3. After selecting `because`, which old cache rows are reused?
4. Which cache rows must be newly calculated?
5. After two decode steps, how many positions does the teaching cache cover?
6. State one benefit and one cost of the KV cache.

## Station 7 — End-to-End Offload Cost

Create one setting where the GPU arithmetic time is lower than CPU compute time,
but the complete GPU path is slower.

1. CPU compute time:
2. Host-to-device time:
3. Launch time:
4. GPU arithmetic time:
5. Device-to-host time:
6. Complete GPU-path time:
7. Why would comparing only CPU compute with GPU arithmetic be misleading?

Now create a setting where offloading is beneficial. What changed?

## Final Teach-Back

Without using the lesson, explain this complete path:

```text
prompt text → token IDs → tensor rows → matrix operations → attention weights
→ causal restriction → prefill output and KV cache → one-token decode steps
```

Then answer:

1. Where is parallelism available?
2. Where are sequential dependencies unavoidable?
3. Which statements in the simulation describe logical work rather than exact
   physical NVIDIA hardware behavior?
4. List three questions you would ask a profiler when repeating these ideas on
   a real GPU.

