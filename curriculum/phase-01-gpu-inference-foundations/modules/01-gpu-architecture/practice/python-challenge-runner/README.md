# Python Foundation Challenge Runner

This is a small, reusable browser-based Python grader for curriculum exercises.
It currently includes dot product and matrix multiplication.

## Run It

From the repository root:

    python3 -m http.server 8765 --directory \
      curriculum/phase-01-gpu-inference-foundations/modules/01-gpu-architecture/practice/python-challenge-runner

Then open:

    http://localhost:8765

The first page load requires internet access because the Python runtime is
loaded from the version-pinned Pyodide CDN. Student code runs inside a browser
Web Worker. A five-second timeout stops likely infinite loops.

## Add a Challenge

Add one object to challenges.js with a stable ID, displayed title and prompt,
required Python function name, starter code, visible tests, hidden tests, and
progressive hints.

Each test provides args plus either expected or expectedException. Hidden means
the page does not show the input before grading; this is a learning aid, not a
security boundary, because all static site code is inspectable.
