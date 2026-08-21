"""Friendly automated checks for Exercise 01."""

import exercise


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, received {actual!r}")
    print(f"PASS  {label}")


def expect_value_error(label, function):
    try:
        function()
    except ValueError:
        print(f"PASS  {label}")
        return
    raise AssertionError(f"{label}: expected ValueError")


def main():
    print("Exercise 01 — concept checks\n")
    check("shape identifies B, T, and D", exercise.shape_3d(exercise.X), (2, 3, 4))
    check("three loops visit 24 values", exercise.count_values(exercise.X), 24)
    check("read batch 0 / token 2 / feature 3", exercise.read_value(exercise.X, 0, 2, 3), -0.3)
    check("read batch 1 / token 0 / feature 2", exercise.read_value(exercise.X, 1, 0, 2), -0.3)
    check("first flat index", exercise.flatten_index_3d(0, 0, 0, 3, 4), 0)
    check("last value in batch item 0", exercise.flatten_index_3d(0, 2, 3, 3, 4), 11)
    check("first value in batch item 1", exercise.flatten_index_3d(1, 0, 0, 3, 4), 12)
    check("middle value in batch item 1", exercise.flatten_index_3d(1, 1, 2, 3, 4), 18)
    check("unflatten index 0", exercise.unflatten_index_3d(0, 3, 4), (0, 0, 0))
    check("unflatten index 18", exercise.unflatten_index_3d(18, 3, 4), (1, 1, 2))
    check("logical storage at FP32 width", exercise.estimate_storage_bytes(exercise.X, 4), 96)
    check("logical storage at FP16 width", exercise.estimate_storage_bytes(exercise.X, 2), 48)

    expect_value_error("empty batch is rejected", lambda: exercise.shape_3d([]))
    expect_value_error("empty token axis is rejected", lambda: exercise.shape_3d([[]]))
    expect_value_error(
        "different token counts are rejected",
        lambda: exercise.shape_3d([[[1, 2]], [[3, 4], [5, 6]]]),
    )
    expect_value_error(
        "different hidden widths are rejected",
        lambda: exercise.shape_3d([[[1, 2], [3]]]),
    )
    expect_value_error("negative flat index is rejected", lambda: exercise.unflatten_index_3d(-1, 3, 4))
    expect_value_error("nonpositive byte width is rejected", lambda: exercise.estimate_storage_bytes(exercise.X, 0))

    print("\nAll checks passed. Complete the extension and written explanation next.")


if __name__ == "__main__":
    main()

