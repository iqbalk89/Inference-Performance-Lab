# Expected Observations

Open this only after writing your own prediction and interpretation.

## Correctness Expectations

- One prefill forward call occurs per measured request.
- The first output ID is selected directly from prefill logits.
- An eight-token output requires seven later decode forward calls in this
  teaching convention.
- Each decode call receives one new input ID and previously returned KV state.
- KV sequence length grows by one processed position per decode call.

## Timing Expectations

- The first un-warmed run is often slower than later runs.
- CPU-clock timing without synchronization can under-report CUDA work.
- Longer output length should add decode iterations but should not materially
  alter the already-completed prefill for an otherwise identical request.
- Longer prompts should affect prefill and initial cache size. Later decode
  latency may also change because the cache history is longer.
- A very small model may be dominated by launch and framework overhead, so its
  curves should not be generalized to a frontier model.

## Profiler Expectations

- NVTX annotations provide semantic phase boundaries; CUDA kernels beneath them
  provide execution evidence.
- CPU gaps can exist between GPU kernel groups.
- The elapsed prefill or decode range includes more than a single kernel.
- Nsight Systems can reveal timing and gaps but usually cannot by itself prove
  why an individual kernel achieved a particular fraction of peak throughput.

## Common Interpretation Errors

- Calling `prefill_ms` complete client TTFT. This lab's TTFT boundary excludes
  network, queue, detokenization, and streaming.
- Dividing the post-first-token duration by total output tokens rather than the
  number of intervals.
- Treating an average as a distribution. Retain individual trials.
- Comparing CPU and GPU results without accounting for different hardware and
  software paths.
- Claiming a bottleneck from utilization percentage alone.
- Reporting “faster” without naming latency, throughput, workload, and boundary.

