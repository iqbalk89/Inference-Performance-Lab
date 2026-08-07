window.PYTHON_CHALLENGES = [
  {
    id: "dot-product",
    title: "Dot Product",
    difficulty: "Foundation",
    functionName: "dot_product",
    prompt: "Return the dot product of two equal-length numeric vectors. Raise ValueError when their lengths differ. Do not use NumPy.",
    concepts: ["vectors", "loops", "multiply-accumulate", "validation"],
    starterCode: ["def dot_product(a, b):", "    # Your solution here", "    pass"].join("\n"),
    visibleTests: [
      { args: [[2, 3, 4], [5, 6, 7]], expected: 56, label: "three positive values" },
      { args: [[1, -2, 3], [4, 5, -6]], expected: -24, label: "negative values" },
      { args: [[], []], expected: 0, label: "empty vectors" },
    ],
    hiddenTests: [
      { args: [[0], [99]], expected: 0, label: "zero" },
      { args: [[0.5, 1.5], [4, 2]], expected: 5.0, label: "floating-point values" },
      { args: [[7, 8], [1]], expectedException: "ValueError", label: "different lengths" },
    ],
    hints: [
      "Check that len(a) equals len(b) before doing arithmetic.",
      "Start an accumulator at zero.",
      "For each index i, add a[i] * b[i] to the accumulator.",
      "Return the accumulator after the loop.",
    ],
  },
  {
    id: "matrix-multiplication",
    title: "Matrix Multiplication",
    difficulty: "Foundation extension",
    functionName: "matrix_multiply",
    prompt: "Multiply two rectangular matrices represented as lists of rows. Return a new list of rows. Raise ValueError when their inner dimensions differ. Do not use NumPy.",
    concepts: ["matrices", "dot products", "shape compatibility", "nested loops"],
    starterCode: ["def matrix_multiply(a, b):", "    # Your solution here", "    pass"].join("\n"),
    visibleTests: [
      { args: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], expected: [[19, 22], [43, 50]], label: "two by two matrices" },
      { args: [[[1, 2, 3]], [[4], [5], [6]]], expected: [[32]], label: "one row times one column" },
    ],
    hiddenTests: [
      { args: [[[1, 0], [0, 1]], [[9, 8, 7], [6, 5, 4]]], expected: [[9, 8, 7], [6, 5, 4]], label: "identity transformation" },
      { args: [[[1, 2, 3]], [[1, 2], [3, 4]]], expectedException: "ValueError", label: "incompatible inner dimensions" },
    ],
    hints: [
      "The number of columns in a must equal the number of rows in b.",
      "The result has len(a) rows and len(b[0]) columns.",
      "One result cell is the dot product of one a row and one b column.",
      "Build each output row, append each completed cell, then append the row.",
    ],
  },
];
