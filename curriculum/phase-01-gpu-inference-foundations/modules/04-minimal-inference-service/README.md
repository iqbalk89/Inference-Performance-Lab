# Module 04 — Minimal Inference Service

**Type:** Learn and build

**Estimated time:** 5–8 hours

## Learning Objectives

- Explain why a model process is wrapped in a service boundary.
- Design a small typed request and response contract.
- Load the model once during service startup.
- Validate generation parameters and return useful errors.
- Separate server processing time from client-observed latency.
- Test API correctness without turning the project into a framework exercise.

## Subsections

1. **Process lifecycle:** startup, model loading, readiness, request execution,
   and shutdown.
2. **API contract:** prompt, maximum output tokens, sampling parameters, and
   structured metrics.
3. **Timing boundary:** queue time, tokenization, prefill, decode, serialization,
   and network time.
4. **Reliability:** validation, timeouts, errors, deterministic test mode, and
   health/readiness checks.
5. **Concurrency introduction:** why simultaneous requests contend for a single
   model/GPU; advanced batching is deferred.

## Minimum Resources

Use FastAPI for the v1 service and read only:

1. [FastAPI lifespan events](https://fastapi.tiangolo.com/advanced/events/): read through the shared-model example; skip deprecated alternative events. **15 minutes.**
2. [FastAPI request-body models](https://fastapi.tiangolo.com/tutorial/body-nested-models/): read through nested Pydantic models and validation. **15 minutes.**
3. [FastAPI response models](https://fastapi.tiangolo.com/tutorial/response-model/): focus on response validation, serialization, and filtering. **10 minutes.**
4. [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/): read the basic `TestClient` workflow. **10 minutes.**

WebSockets, authentication frameworks, databases, Kubernetes, and production
ASGI tuning are outside this module.

## Required Work

1. Complete [exercises.md](exercises.md).
2. Complete [lab.md](lab.md).
3. Document architecture and limitations.

## Completion Gate

- One command starts the service.
- The model loads once rather than once per request.
- Valid requests generate text and invalid requests return clear errors.
- Automated tests cover health, validation, and deterministic generation.
- Response metrics have explicit definitions and units.
