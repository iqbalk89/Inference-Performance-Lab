# Current Study Plan — Foundation Reset

**Current position:** Lesson 01  
**Do not study attention yet.** Attention begins only after the foundation gate
below is comfortable.

## Why This Reset Exists

The target GPU-performance roles eventually require attention, FlashAttention,
KV-cache optimization, CUDA/Triton kernels, profiling, memory-bandwidth
analysis, and distributed systems. Those topics depend on a smaller vocabulary:
dependencies, vectors, dot products, matrices, tensor shapes, and finite
parallel hardware.

If those foundations are still effortful, Q/K/V notation hides the mechanism
instead of explaining it. This is a sequencing problem, not evidence that the
student cannot learn attention.

## Resume Here

1. Read [Lesson 01](lessons/01-computing-and-parallelism-foundations/), one
   section at a time. Write an answer to each section checkpoint before reading
   the supplied explanation again.
2. Open the [Lesson 01 visual lab](lessons/01-computing-and-parallelism-foundations/lab/)
   and complete **Stations 1–4 only**.
3. Complete the exercises below without copying an answer from the lesson.
4. Use the readiness gate. Review only the weak item and try again the next day.
5. After passing the gate, begin
   [Lesson 02](lessons/02-transformer-inference-foundations/). Then return to
   lab Stations 5–6.

## Minimum Foundation Exercises

### 1. Dependencies

Given:

    load A ──→ multiply A ──┐
                            ├──→ add ──→ store
    load B ──→ multiply B ──┘

Identify which operations can begin together and which must wait. Explain why
the two multiplication results may be produced in either order but `add` must
wait for both.

<details>
<summary><strong>Worked answer</strong></summary>

At the beginning, `load A` and `load B` are both ready because neither depends
on another operation. They may start together if execution resources are
available, or they may run in either order.

After each load finishes, its corresponding multiplication becomes ready. The
two multiplications are independent: each uses a different input and neither
needs the other multiplication's result.

`add` has two dependencies. It needs both multiplication results, so finishing
only one is insufficient. `store` then waits for `add` because the sum is the
value being stored.

One valid schedule is:

    wave 1: load A, load B
    wave 2: multiply A, multiply B
    wave 3: add
    wave 4: store

This is one valid schedule, not the only possible timing. Limited hardware could
serialize independent operations without changing their dependency relationships.

</details>

### 2. Dot Product

Calculate every step:

    [2, 3, 4] · [5, 6, 7]

Expected final value: `56`. Do not move on until you can explain where every
multiplication and addition came from.

<details>
<summary><strong>Worked answer</strong></summary>

A dot product pairs values at matching coordinates, multiplies each pair, and
adds the products:

    [2, 3, 4] · [5, 6, 7]
    = (2 × 5) + (3 × 6) + (4 × 7)
    = 10 + 18 + 28
    = 56

There are three multiplications because each vector has three coordinates. The
three products are partial results, and the additions reduce them to one scalar.

</details>

### 3. One Matrix Output Cell

Given:

    A = [1  2]       B = [5  6]
        [3  4]           [7  8]

Calculate `C = AB` one cell at a time. For every cell, name the selected row of
A and column of B. Then explain why `C[0,0]` does not need the completed value
of `C[1,1]`.

Expected result:

    C = [19  22]
        [43  50]

<details>
<summary><strong>Worked answer</strong></summary>

Each output cell selects one row from A and one column from B:

    C[0,0] = [1,2] · [5,7] = (1×5) + (2×7) = 19
    C[0,1] = [1,2] · [6,8] = (1×6) + (2×8) = 22
    C[1,0] = [3,4] · [5,7] = (3×5) + (4×7) = 43
    C[1,1] = [3,4] · [6,8] = (3×6) + (4×8) = 50

Therefore:

    C = [19  22]
        [43  50]

`C[0,0]` and `C[1,1]` read A and B, but neither reads the other output
cell. They can therefore be calculated independently once A and B are
available. Within each cell, its products must still exist before their sum.

</details>

### 4. Shapes

Interpret this operation aloud:

    X [128, 4096] × W [4096, 4096] → Y [128, 4096]

Explain:

- what one row of X represents;
- why the two inner dimensions must match;
- how many output rows and columns Y contains;
- why separate Y cells expose parallel work.

<details>
<summary><strong>Worked answer</strong></summary>

X contains 128 rows with 4,096 features per row. In an inference example, one
row could represent the hidden state of one token position. W maps 4,096 input
features to 4,096 output features.

The inner dimensions match:

    X [128, 4096] × W [4096, 4096]
             └──────────┘
             dot-product length

One Y cell uses a length-4,096 row from X and a length-4,096 column from W.
The outer dimensions determine the result:

    Y shape = [128, 4096]
    Y values = 128 × 4096 = 524,288

Every Y cell is a distinct row-column dot product. Cells share input data but
do not require one another's completed output values. This exposes parallel
work, although finite hardware may schedule the cells in tiles and waves.

</details>

### 5. Tensor Axes

For `H [2, 128, 4096]`, explain the meaning of each axis and the meaning of
`H[1, 17, 250]`. Calculate the total number of scalar values.

Expected scalar count: `1,048,576`.

<details>
<summary><strong>Worked answer</strong></summary>

Interpret the axes as:

    H [batch, token position, hidden feature]
      [  2,          128,           4096]

- Axis 0 contains two sequences or batch items.
- Axis 1 contains 128 token positions per sequence.
- Axis 2 contains 4,096 features per token position.

`H[1,17,250]` selects one scalar: feature 250 for token position 17 in batch
item 1. With zero-based indexing, these are the second batch item, eighteenth
token position, and 251st feature.

    2 × 128 × 4096 = 1,048,576 values

Shape does not state the data type, device, or storage bytes; those are
additional tensor properties.

</details>

### 6. Finite Hardware

If a computation contains 13 independent logical tiles but the teaching model
has four execution slots, draw the scheduling waves. Explain why 13 independent
tiles do not mean 13 tiles execute simultaneously.

<details>
<summary><strong>Worked answer</strong></summary>

At most four tiles fit in one teaching wave:

    wave 1: tiles  0,  1,  2,  3
    wave 2: tiles  4,  5,  6,  7
    wave 3: tiles  8,  9, 10, 11
    wave 4: tile  12, idle, idle, idle

    ceiling(13 ÷ 4) = ceiling(3.25) = 4 waves

Independence makes all tiles eligible for scheduling without a result-order
dependency. It does not create physical capacity. Only four fit at once in
this teaching model, so the remaining ready tiles wait for later waves.

A tile and slot are conceptual; they are not literal CUDA cores or a complete
model of an NVIDIA Streaming Multiprocessor.

</details>

## Readiness Gate for Attention

Begin Lesson 02 only when you can answer these without notes:

1. What is a dependency, and how does it constrain execution order?
2. What is a vector?
3. How is a dot product calculated?
4. How is one matrix-multiplication output cell calculated?
5. Why must the inner matrix dimensions match?
6. What does `[batch, tokens, features]` mean?
7. Which matrix output cells can be calculated independently?
8. Why can available parallel work exceed simultaneously executing work?
9. What is the difference between arithmetic work and moving its input data?

This is not a memorization test. Draw a small example while explaining. If the
explanation is mechanically correct and you can recover the details yourself,
the foundation is adequate.

## What Comes After the Gate

Lesson 02 will reuse familiar operations:

    X × WQ → Q
    X × WK → K
    X × WV → V
    Q × Kᵀ → scores
    softmax(masked scores) × V → attention output

At that point attention should look like an arrangement of known operations,
not an unrelated collection of letters.
