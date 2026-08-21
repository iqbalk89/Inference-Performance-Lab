# Exercise 01 Worksheet

## Name and Date

- Date:
- Start time:
- End time:

## Initial Predictions — Complete Before Running Code

For `X` with shape `[2, 3, 4]`:

| Question | Prediction | Reasoning |
| --- | --- | --- |
| Independent prompts | | |
| Token positions per prompt | | |
| Features per token | | |
| Total numerical values | | |
| Bytes at 4 bytes/value | | |
| Shape if batch doubles | | |
| Values and bytes if batch doubles | | |
| Shape if hidden width doubles | | |
| Values and bytes if hidden width doubles | | |

## Manual Index Trace

| Expression | Batch item | Token position | Feature | Value |
| --- | ---: | ---: | ---: | ---: |
| `X[0][0][0]` | | | | |
| `X[0][2][3]` | | | | |
| `X[1][0][2]` | | | | |
| `X[1][2][1]` | | | | |

## Flat-Index Trace

Show all arithmetic.

```text
flat index of X[0][2][3] =

flat index of X[1][0][0] =

flat index of X[1][1][2] =
```

## Check Results

Paste or summarize the checker output:

```text

```

Failures encountered and what caused them:

## Extension Prediction

Before adding the third batch item:

| Quantity | Prediction | Measured result |
| --- | ---: | ---: |
| New shape | | |
| Value count | | |
| Bytes at 4 bytes/value | | |
| Flat index `[2, 1, 3]` | | |

## Explain Without Notes

1. What is a batch?

2. Why is `[2, 3, 4]` not a two-dimensional matrix?

3. What does `X[1][2][3]` select?

4. If `B` doubles, which work and storage quantities should approximately
   double in this simplified model?

5. Why do batch items share execution but not attention context?

6. What does this nested-list representation fail to model about real GPU
   tensor storage?

## Prediction Versus Observation

- Which predictions were correct?
- Which were wrong?
- What did the incorrect predictions assume?
- What can you now explain that you could not explain before?
- What question remains?

