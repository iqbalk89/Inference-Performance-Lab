# Problem 03 Worksheet — Combined QKV Projection

Complete this worksheet before opening the answer key.

## 1. Shapes

```text
X:
W_Q:
W_K:
W_V:
W_QKV:
Q:
K:
V:
[Q K V]:
```

## 2. Symbolic equations

Write formulas before numbers:

```text
one projection parameters =
fused QKV parameters =
QKV FLOPs =
total HBM bytes =
arithmetic intensity =
```

## 3. Decode table

Fill in the values and units from the problem statement.

## 4. Prefill table

Fill in the values and units from the problem statement.

## 5. Interpretation

Explain in your own words:

- what fusion changes;
- what fusion does not change;
- why QKV projection is still separate from attention;
- why prefill and decode have different arithmetic intensity.

## 6. Sanity checks

- Does fused QKV have exactly three times as many weight parameters as one
  projection?
- Does prefill have 512 times the QKV FLOPs of decode?
- Does the fused weight read stay constant as `M` changes?
- Do the inner matrix dimensions match?
