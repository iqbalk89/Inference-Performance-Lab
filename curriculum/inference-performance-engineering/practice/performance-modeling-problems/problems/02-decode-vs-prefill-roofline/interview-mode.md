# Problem 02 — Interview Mode

Use this after completing the worksheet once. Ask a partner to read only this
page, or record yourself answering under time pressure.

## Format

```text
5 minutes   clarify and state assumptions
15 minutes  calculate decode and prefill models
5 minutes   compare and explain the bottleneck change
5 minutes   answer follow-ups
```

Do not begin arithmetic until you have written:

```text
shapes → FLOPs → bytes → arithmetic intensity → two time bounds → max
```

## Interviewer Prompts

Reveal these one at a time.

1. Walk me through the model boundary you chose.
2. What remains constant between decode and prefill?
3. What scales with token-row count `M`?
4. Which traffic term dominates at `M=1`?
5. How do you determine memory-bound versus compute-bound?
6. Why is `max(compute time, memory time)` used instead of their sum?
7. What assumption makes this a lower bound rather than a latency prediction?
8. What real measurements would calibrate the two peak ceilings?
9. What would batching several decode requests do to `M` and arithmetic
   intensity?
10. Could the measured kernel still be memory-bound when this model predicts
    compute-bound? Give three reasons.

## Strong-Answer Signals

A strong candidate:

- writes units throughout;
- distinguishes a model call from one token row;
- notices that weights are amortized across `M` rows;
- compares both time bounds rather than relying on intuition;
- calls the result a theoretical lower bound;
- separates latency per call, latency per row, and throughput;
- names missing traffic and imperfect peak utilization;
- proposes profiler evidence instead of claiming certainty.

## Common Failure Modes

- Multiplying weight bytes by 512 even though the model says one weight read per
  call
- Forgetting input or output traffic
- Comparing FLOPs directly with bytes without arithmetic intensity
- Adding ideal compute and memory times despite the stated roofline model
- Calling peak throughput an achieved rate
- Concluding that prefill has lower request latency because it has lower time
  per token row
- Treating the single-projection result as the complete Transformer layer

