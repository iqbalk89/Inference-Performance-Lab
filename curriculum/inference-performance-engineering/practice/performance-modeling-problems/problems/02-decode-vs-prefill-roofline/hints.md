# Problem 02 Hint Ladder

Open only the section where you are stuck.

## Ridge Point

<details>
<summary>Hint 1</summary>

Divide `120 × 10^12 FLOPs/s` by `600 × 10^9 bytes/s`. Seconds cancel.

</details>

## Decode Traffic

<details>
<summary>Hint 1</summary>

`W` has the same byte count as Problem 01. `X` and `Y` each contain only 4,096
FP16 values.

</details>

<details>
<summary>Hint 2</summary>

```text
total bytes = 4096² × 2 + 4096 × 2 + 4096 × 2
```

</details>

## Prefill Traffic

<details>
<summary>Hint 1</summary>

The model call still reads one `W` matrix. Both `X` and `Y` now contain
`512 × 4096` values.

</details>

## Time Conversion

<details>
<summary>Hint</summary>

First calculate seconds. Multiply seconds by `10^6` to obtain microseconds.

</details>

## Bound Classification

<details>
<summary>Hint</summary>

The larger of compute time and memory time is the roofline lower bound. The
resource associated with that larger time is the modeled bottleneck.

</details>

## Crossover

<details>
<summary>Hint 1</summary>

Let `d = K = N`, with FP16 input, output, and weights:

```text
FLOPs = 2Md²
bytes = 2d² + 2Md + 2Md
```

</details>

<details>
<summary>Hint 2</summary>

Simplify:

```text
AI(M) = Md / (d + 2M)
```

Set this equal to `200`, substitute `d = 4096`, and solve for `M`.

</details>

