"""Mechanical reference solution for Exercise 01.

Attempt exercise.py and use hints.md before opening this file.
"""


def shape_3d(tensor):
    if len(tensor) == 0:
        raise ValueError("batch axis cannot be empty")

    first_batch_item = tensor[0]
    if len(first_batch_item) == 0:
        raise ValueError("token axis cannot be empty")

    first_token_row = first_batch_item[0]
    if len(first_token_row) == 0:
        raise ValueError("hidden-width axis cannot be empty")

    batch_size = len(tensor)
    token_count = len(first_batch_item)
    hidden_width = len(first_token_row)

    for batch_item in tensor:
        if len(batch_item) != token_count:
            raise ValueError("batch items have different token counts")
        for token_row in batch_item:
            if len(token_row) != hidden_width:
                raise ValueError("token rows have different hidden widths")

    return (batch_size, token_count, hidden_width)


def count_values(tensor):
    total = 0
    for batch_item in tensor:
        for token_row in batch_item:
            for _value in token_row:
                total = total + 1
    return total


def read_value(tensor, batch_index, token_index, feature_index):
    selected_batch_item = tensor[batch_index]
    selected_token_row = selected_batch_item[token_index]
    selected_feature = selected_token_row[feature_index]
    return selected_feature


def flatten_index_3d(batch_index, token_index, feature_index, tokens, width):
    values_per_batch_item = tokens * width
    batch_offset = batch_index * values_per_batch_item
    token_offset = token_index * width
    feature_offset = feature_index
    flat_index = batch_offset + token_offset + feature_offset
    return flat_index


def unflatten_index_3d(flat_index, tokens, width):
    if flat_index < 0:
        raise ValueError("flat index cannot be negative")
    if tokens <= 0 or width <= 0:
        raise ValueError("tokens and width must be positive")

    values_per_batch_item = tokens * width
    batch_index = flat_index // values_per_batch_item
    position_inside_batch = flat_index % values_per_batch_item
    token_index = position_inside_batch // width
    feature_index = position_inside_batch % width
    return (batch_index, token_index, feature_index)


def estimate_storage_bytes(tensor, bytes_per_value):
    if bytes_per_value <= 0:
        raise ValueError("bytes per value must be positive")
    number_of_values = count_values(tensor)
    storage_bytes = number_of_values * bytes_per_value
    return storage_bytes

