# Module 04 Lab — Inference Performance Lab v1 Service

## Objective

Expose the instrumented model through a minimal, testable HTTP interface without
hiding the inference mechanics learned in Module 03.

## Build Requirements

Implement under `inference-servers/v1/`:

- Application startup that loads one pinned model
- Health and readiness endpoints
- One generation endpoint
- Typed request validation
- Configurable prompt, maximum output length, and explicit sampling parameters
- Structured response containing generated text, token counts, TTFT or its
  documented approximation, end-to-end server latency, TPOT, output rate, and
  peak GPU memory
- Clear error responses
- Basic request logging without prompt contents by default

Add tests under `tests/` for:

- Health and readiness behavior
- Valid deterministic request
- Empty or invalid prompt
- Invalid generation parameters
- Over-limit request
- Stable response schema

## Architecture Note

Document:

- Process and model lifecycle
- Request flow
- Metric definitions
- Current concurrency policy
- Known limitations, including whether token streaming is implemented

## Pass Criteria

- The service starts from documented commands on a fresh environment.
- Tests do not require a paid GPU where a mock or tiny CPU fixture is adequate.
- GPU-only integration verification is documented separately.
- Timing boundaries are honest and reproducible.
