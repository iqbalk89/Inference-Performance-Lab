# Hint Ladder — Exercise 01

Open only the hint needed for the function you are currently solving.

## `shape_3d`

<details>
<summary>Hint 1</summary>

The outer list length is batch size. The first batch item's length is token
count. The first token row's length is hidden width.

</details>

<details>
<summary>Hint 2</summary>

After obtaining the candidate dimensions, visit every batch item. Confirm its
token count equals the first item. Then visit every token row and confirm its
width equals the first row.

</details>

## `count_values`

<details>
<summary>Hint</summary>

Start `total = 0`. Nest three loops: batch items, token rows, then scalar
features. Add one for each scalar visited.

</details>

## `read_value`

<details>
<summary>Hint</summary>

Make each selection a separate line: select the batch item, then the token row,
then the feature.

</details>

## `flatten_index_3d`

<details>
<summary>Hint 1</summary>

One complete batch item contains `tokens × width` scalar values.

</details>

<details>
<summary>Hint 2</summary>

The flat offset is:

```text
complete batch items before this one
+ complete token rows before this one
+ features before this feature
```

</details>

## `unflatten_index_3d`

<details>
<summary>Hint 1</summary>

`values_per_batch = tokens × width`. Integer division by that amount gives the
batch index. Remainder gives the position inside that batch item.

</details>

<details>
<summary>Hint 2</summary>

Within one batch item, integer division by `width` gives token index and
remainder by `width` gives feature index.

</details>

## `estimate_storage_bytes`

<details>
<summary>Hint</summary>

Logical storage is `number_of_values × bytes_per_value`. Reuse the function that
already counts values.

</details>

