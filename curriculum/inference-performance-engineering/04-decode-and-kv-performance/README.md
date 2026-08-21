# Module 04 — Decode and KV Performance

**Status:** Planned

This module will treat one autoregressive decode iteration as the unit of work.
It will model weight reads, KV-cache growth and reads, context length, batch
size, memory bandwidth, launch overhead, and per-token latency.

The primary artifact will predict KV capacity and decode latency, then compare
early-context and late-context profiler captures.

