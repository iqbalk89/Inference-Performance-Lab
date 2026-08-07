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

### 2. Dot Product

Calculate every step:

    [2, 3, 4] · [5, 6, 7]

Expected final value: `56`. Do not move on until you can explain where every
multiplication and addition came from.

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

### 4. Shapes

Interpret this operation aloud:

    X [128, 4096] × W [4096, 4096] → Y [128, 4096]

Explain:

- what one row of X represents;
- why the two inner dimensions must match;
- how many output rows and columns Y contains;
- why separate Y cells expose parallel work.

### 5. Tensor Axes

For `H [2, 128, 4096]`, explain the meaning of each axis and the meaning of
`H[1, 17, 250]`. Calculate the total number of scalar values.

Expected scalar count: `1,048,576`.

### 6. Finite Hardware

If a computation contains 13 independent logical tiles but the teaching model
has four execution slots, draw the scheduling waves. Explain why 13 independent
tiles do not mean 13 tiles execute simultaneously.

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
