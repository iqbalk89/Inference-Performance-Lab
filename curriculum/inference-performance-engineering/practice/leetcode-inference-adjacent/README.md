# Inference-Adjacent LeetCode Practice

## Purpose

These problems supplement the inference-performance curriculum with focused
coding practice. They develop matrix-indexing fluency, representation choices,
caching, sampling, scheduling, and concurrency.

They are **not** substitutes for performance modeling, GPU profiling, or
inference labs. An accepted submission is only the beginning of each exercise.

## Minimum Sequence

Complete these in order. Problems marked Premium may require a LeetCode
subscription; the repository will eventually provide equivalent local
exercises.

| Order | Problem | Difficulty/access | Curriculum connection | Status |
| ---: | --- | --- | --- | --- |
| 1 | [566 — Reshape the Matrix](https://leetcode.com/problems/reshape-the-matrix/) | Easy | Tensor shapes, flat storage, index mapping | Not started |
| 2 | [867 — Transpose Matrix](https://leetcode.com/problems/transpose-matrix/) | Easy | Layout transformations and memory-access order | Not started |
| 3 | [1570 — Dot Product of Two Sparse Vectors](https://leetcode.com/problems/dot-product-of-two-sparse-vectors/) | Medium, Premium | Multiply-accumulate work and sparse representations | Not started |
| 4 | [311 — Sparse Matrix Multiplication](https://leetcode.com/problems/sparse-matrix-multiplication/) | Medium, Premium | Matrix mechanics, sparsity, and avoided work | Not started |
| 5 | [528 — Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/) | Medium | Weighted selection and sampling foundations | Not started |
| 6 | [146 — LRU Cache](https://leetcode.com/problems/lru-cache/) | Medium | Capacity, lookup, reuse, and eviction policy | Not started |

## Systems Extensions

Complete these after the minimum sequence or alongside the serving modules.

| Problem | Difficulty/access | Curriculum connection | Status |
| --- | --- | --- | --- |
| [1188 — Design Bounded Blocking Queue](https://leetcode.com/problems/design-bounded-blocking-queue/) | Medium, Premium | Backpressure, concurrency, request admission | Not started |
| [621 — Task Scheduler](https://leetcode.com/problems/task-scheduler/) | Medium | Scheduling, idle intervals, and utilization | Not started |

## Required Workflow for Every Problem

### 1. Solve mechanically first

Do not begin with a Python shortcut that hides the operation being studied.
Write explicit loops and index calculations. After the mechanical version is
correct, a second idiomatic Python version may be added.

### 2. Record shape and representation

Answer:

- What is the shape of every input and output?
- How is the data represented in memory?
- What does each index mean?
- Which dimensions must be compatible?

### 3. Count work and data

Answer:

- How many loop iterations execute?
- How many comparisons, multiplications, and additions occur?
- How many input values are read?
- How many output values are written?
- Which additional data structures are allocated?

Exact byte counts are not required initially. State assumptions such as
`4 bytes per integer` when making a byte estimate.

### 4. Predict scaling

Before testing larger inputs, predict:

- What happens if each relevant dimension doubles?
- What happens if it becomes ten times larger?
- Is runtime likely dominated by arithmetic, memory access, allocation, or
  synchronization?

### 5. Measure

Use several input sizes, multiple trials, and a warm-up. Record median rather
than only one run. Compare the observed curve with the predicted complexity.

### 6. Translate to inference

Explain what transfers to inference engineering and what does not. For example,
an LRU cache teaches reuse and eviction, but a transformer KV cache normally
stores per-layer attention tensors and is not simply an LRU dictionary.

## Problem-Specific Assignments

### 566 — Reshape the Matrix

Implement the index conversion without flattening shortcuts:

```text
flat_index = old_row × old_column_count + old_column
new_row    = flat_index // new_column_count
new_column = flat_index % new_column_count
```

Then answer:

1. Why must the old and new matrices contain the same number of values?
2. Does a reshape necessarily move data in a tensor framework?
3. Under what layout conditions could reshape be a metadata-only operation?
4. How would a non-contiguous input complicate the operation?

### 867 — Transpose Matrix

After the basic solution, compare these traversal orders:

```text
read input rows, write output columns
read input columns, write output rows
```

Then answer:

1. Which version reads contiguous input locations in row-major storage?
2. Which version writes contiguous output locations?
3. Why can two `O(MN)` implementations have different measured performance?
4. Why do matrix kernels care about layout even when shapes are compatible?

### 1570 — Dot Product of Two Sparse Vectors

Implement three versions when possible:

1. Dense loop over every position
2. Dictionary of nonzero index → value
3. Sorted `(index, value)` pairs with two pointers

Then answer:

1. How many multiply-accumulate operations does each representation perform?
2. At what sparsity might representation overhead erase the benefit?
3. How does irregular lookup affect locality?
4. Why is sparse-vector performance not representative of ordinary dense LLM
   GEMMs?

### 311 — Sparse Matrix Multiplication

First implement the complete mechanical triple loop:

```text
for output_row
    for output_column
        accumulator = 0
        for shared_dimension
            accumulator += left[row][k] × right[k][column]
```

Then implement a version that skips work involving zeros.

Answer:

1. What compatibility rule must the matrix shapes satisfy?
2. What are the dense FLOP and MAC counts?
3. Which zero patterns does the optimized representation exploit?
4. What metadata and irregular-access costs does sparsity introduce?
5. Why does fewer mathematical operations not guarantee proportionally lower
   execution time?

### 528 — Random Pick with Weight

Treat the weights as unnormalized scores and build cumulative probability
intervals.

Answer:

1. How does a prefix sum turn weights into searchable intervals?
2. What are initialization and per-selection complexities?
3. How is this similar to selecting from token probabilities?
4. What important steps are missing compared with LLM sampling—such as logits,
   softmax, temperature, top-k, or top-p processing?

### 146 — LRU Cache

Implement the required constant-average-time lookup and update behavior.

Answer:

1. Which two data structures cooperate to provide lookup and eviction order?
2. What metadata overhead exists per cached entry?
3. How does capacity affect hit rate and memory?
4. Why is an LRU cache not the same thing as a transformer KV cache?
5. Where might eviction policies appear in a real inference service?

### 1188 — Design Bounded Blocking Queue

Complete this while studying serving capacity and concurrency.

Answer:

1. What happens when producers arrive faster than consumers complete work?
2. What condition wakes a blocked producer? A blocked consumer?
3. How does bounded capacity implement backpressure?
4. Which latency metric grows while requests wait?

### 621 — Task Scheduler

Use it as a scheduling analogy, not a model of a CUDA scheduler.

Answer:

1. Why do idle intervals occur?
2. What change improves utilization?
3. What aspect resembles bubbles in a processing schedule?
4. Which assumptions make the problem unlike continuous batching or GPU
   execution?

## Exercise Record Template

Create one file per completed problem under your experiment or journal records:

```markdown
# Problem number and title

## Mechanical solution
## Idiomatic Python solution
## Input/output shapes
## Representation and index meanings
## Operation count
## Estimated bytes
## Complexity and scaling prediction
## Measurements
## Prediction versus observation
## Connection to inference
## What does not transfer to inference
## Mistakes and lessons learned
```

Do not commit copied LeetCode problem statements or editorial solutions. Link to
the problem, write the implementation yourself, and record your own analysis.

