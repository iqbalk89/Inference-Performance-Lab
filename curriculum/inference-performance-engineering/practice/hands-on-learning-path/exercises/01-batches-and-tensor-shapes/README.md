# Exercise 01 — Build a Batch and Reason About Its Shape

**Time:** 60–90 minutes  
**Hardware:** Intel Mac; no GPU required  
**Language:** Python, using explicit loops and lists

## Performance Question

How do batch size, prompt-token count, hidden width, element count, and storage
relate to one another?

## What You Will Build

You will represent this tensor without NumPy or PyTorch:

```text
X shape = [B, T, D] = [2, 3, 4]

batch item 0: [Cats]  [chase] [mice]
batch item 1: [Dogs]  [guard] [homes]
```

Each token position owns a four-number feature row. Python represents the
tensor as nested lists:

```text
X[batch_item][token_position][feature]
```

Read the accompanying
[batch visual](../../../../02-e2e-inference-pipeline/lessons/01-request-to-performance-equation/assets/prefill-03-what-is-a-batch.svg)
before starting.

## Part 1 — Predict on Paper

Write answers in [worksheet.md](worksheet.md) before running code.

For `X` with shape `[2, 3, 4]`, predict:

1. Number of independent prompts
2. Token positions per prompt
3. Features per token position
4. Total stored numerical values
5. Storage bytes if every value uses 4 bytes
6. New shape, value count, and bytes if batch size doubles
7. New shape, value count, and bytes if hidden width doubles

## Part 2 — Orient Yourself in Python

Open [exercise.py](exercise.py). The example tensor uses three levels of lists:

```python
X[0]        # complete matrix for batch item 0
X[0][1]     # feature row for token position 1 in item 0
X[0][1][2]  # feature 2 of token position 1 in item 0
```

The bracket order matches `[B, T, D]`.

Trace these accesses manually:

```text
X[0][0][0]
X[0][2][3]
X[1][0][2]
X[1][2][1]
```

## Part 3 — Implement Six Functions

Replace the `TODO` sections in `exercise.py`:

1. `shape_3d(tensor)`
2. `count_values(tensor)`
3. `read_value(tensor, batch_index, token_index, feature_index)`
4. `flatten_index_3d(batch_index, token_index, feature_index, tokens, width)`
5. `unflatten_index_3d(flat_index, tokens, width)`
6. `estimate_storage_bytes(tensor, bytes_per_value)`

Constraints:

- Use explicit loops.
- Do not import NumPy, PyTorch, or another tensor library.
- Do not use a hard-coded `[2, 3, 4]` return value.
- `shape_3d` must reject ragged or empty tensors with `ValueError`.
- Write out intermediate arithmetic while learning; compact code is not the
  objective.

## Part 4 — Run the Checks

From this directory:

```bash
python3 check.py
```

The checker reports one concept at a time. When a check fails, explain the
failure before changing code.

If stuck, use [hints.md](hints.md) one hint at a time. Only then compare with
[solution.py](solution.py).

## Part 5 — Connect Logical Indexing to Flat Memory

Nested lists help you see dimensions, but dense tensor storage is commonly
reasoned about as a linear sequence of values. For row-major `[B, T, D]`:

```text
flat_index = (batch_index × T × D)
           + (token_index × D)
           + feature_index
```

Equivalent grouping:

```text
flat_index = (batch_index × T + token_index) × D + feature_index
```

Explain why changing the feature index by one changes the flat index by one,
while changing the batch index by one skips an entire `T × D` matrix.

## Part 6 — Extend the Experiment

Add a third batch item with three token rows of width four. Before running the
checker or printing anything, predict:

- the new shape;
- total value count;
- storage bytes at 4 bytes/value;
- the flat index of `[batch=2, token=1, feature=3]`.

Then verify your prediction using your own functions.

## Part 7 — Explain Without Notes

Record short answers in the worksheet:

1. What is a batch?
2. Why is `[2, 3, 4]` not a two-dimensional matrix?
3. What does `X[1][2][3]` select?
4. If `B` doubles, which work and storage quantities should approximately
   double under this simplified model?
5. Why do batch items share execution but not attention context?
6. What does this nested-list representation fail to model about real GPU
   tensor storage?

## Completion Gate

- [ ] All paper predictions recorded before execution
- [ ] All automated checks pass
- [ ] Third batch item extension completed
- [ ] Prediction-versus-observation comparison recorded
- [ ] Six explanation questions answered without notes
- [ ] You can reconstruct `[B, T, D]` from memory

Do not move to reshape and transpose until you can point to any element in
`[B, T, D]`, name what each index means, and calculate its row-major flat index.

