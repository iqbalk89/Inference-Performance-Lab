# Module 04 Exercises

## Lifecycle

1. Why should model loading occur at startup rather than per request?
2. How should readiness differ from basic process liveness?
3. What state must be cleaned up during graceful shutdown?

## API Contract

4. Which generation parameters belong in the first API request?
5. What input limits protect the service and GPU?
6. Why should token counts and timing values be structured fields rather than
   embedded in log text?

## Timing

7. Why can client latency exceed server generation latency?
8. Define the start and end boundaries for TTFT in this service.
9. How would a non-streaming endpoint limit the accuracy of client-observed
   TTFT?
10. Which timing segments should be recorded separately?

## Reliability and Concurrency

11. What should happen when the request exceeds the supported context length?
12. Which errors are client errors and which are server errors?
13. Why is deterministic generation valuable in automated tests?
14. What could happen if two requests attempt generation concurrently on one
    GPU without an explicit policy?
15. Why are authentication, rate limiting, advanced scheduling, and dynamic
    batching deferred from the minimal service?
