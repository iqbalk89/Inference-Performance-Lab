# Python Foundation Challenge Runner

This is a small, reusable browser-based Python grader for curriculum exercises.
It currently includes dot product and matrix multiplication.

The page assumes programming experience but no Python experience. Each
challenge includes a revealable reference solution, a C++/C#-to-Python
translation, progressive hints, and a reversible option to load the solution
into the editor.

Reference solutions deliberately avoid convenience constructs when those
constructs would hide the algorithm. Each solution includes a complete numerical
trace of loop indices, reads, products, accumulator updates, dimension checks,
and output writes.

The reference-solution viewer includes Python syntax highlighting, line
numbers, and highlighted dimension-compatibility lines. Loading the solution
moves it into the editable rich-code surface and preserves the prior draft.

The matrix challenge accepts general rectangular 2D matrices. Its tests cover
square and non-square shapes, including `3×1 @ 1×4`, `2×3 @ 3×4`, and
`3×2 @ 2×1`, plus incompatible inner dimensions.

## Run It

Open index.html directly in a browser. No local server, installation, or build
step is required.

The first page load requires internet access because the Python runtime is
loaded from the version-pinned Pyodide CDN. All application JavaScript,
challenge definitions, test cases, worker code, styles, and markup live inside
index.html. Student code runs inside a browser Web Worker, and a five-second
timeout stops likely infinite loops.

## Add a Challenge

Add one object to the challengeData JSON block in index.html with a stable ID,
displayed title and prompt, required Python function name, starter code, visible
tests, hidden tests, and progressive hints.

Each test provides args plus either expected or expectedException. Hidden means
the page does not show the input before grading; this is a learning aid, not a
security boundary, because all static site code is inspectable.
