"""Exercise 01 starter: batches and tensor shapes.

Edit the six TODO functions, then run: python3 check.py
Do not use NumPy or PyTorch in this exercise.
"""

X = [
    [
        [0.2, -0.1, 0.7, 0.4],   # batch 0, token 0: Cats
        [0.6, 0.3, -0.2, 0.1],   # batch 0, token 1: chase
        [-0.4, 0.8, 0.5, -0.3],  # batch 0, token 2: mice
    ],
    [
        [0.1, 0.5, -0.3, 0.2],   # batch 1, token 0: Dogs
        [0.7, -0.2, 0.4, 0.6],   # batch 1, token 1: guard
        [-0.5, 0.9, 0.3, -0.1],  # batch 1, token 2: homes
    ],
]


def shape_3d(tensor):
    """Return (batch_size, token_count, hidden_width).

    Raise ValueError if the tensor is empty or ragged. Ragged means that batch
    items have different token counts or token rows have different widths.
    """
    # TODO: implement with explicit loops.
    raise NotImplementedError


def count_values(tensor):
    """Count scalar values by visiting them with three explicit loops."""
    # TODO: do not calculate only B * T * D in this function.
    raise NotImplementedError


def read_value(tensor, batch_index, token_index, feature_index):
    """Return the scalar selected by the three logical indices."""
    # TODO: make the three indexing operations visible.
    raise NotImplementedError


def flatten_index_3d(batch_index, token_index, feature_index, tokens, width):
    """Map [batch, token, feature] to a row-major flat index."""
    # TODO: calculate the batch offset, token offset, and feature offset.
    raise NotImplementedError


def unflatten_index_3d(flat_index, tokens, width):
    """Map a nonnegative row-major flat index back to (batch, token, feature)."""
    # TODO: use integer division and remainder. Reject negative indices.
    raise NotImplementedError


def estimate_storage_bytes(tensor, bytes_per_value):
    """Return logical value storage, excluding Python-object overhead."""
    # TODO: reject nonpositive bytes_per_value and reuse count_values.
    raise NotImplementedError


if __name__ == "__main__":
    print("Shape:", shape_3d(X))
    print("Values:", count_values(X))
    print("Logical storage at 4 bytes/value:", estimate_storage_bytes(X, 4), "bytes")

