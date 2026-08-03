# Learning and Engineering Backlog

This backlog preserves valuable topics without interrupting the current phase.
Items are not commitments for the present week. They should be promoted into a
phase plan only when prerequisites are in place and the work can produce a
clear, measurable deliverable.

## Current Priority

Complete Phase 1 using server-side PyTorch inference on a remote NVIDIA GPU:

- Run and understand the first decoder-only model.
- Measure latency, throughput, and GPU memory.
- Capture PyTorch Profiler and Nsight Systems traces.
- Study tokenization, prefill, decode, attention, and the KV cache.
- Build the first minimal inference service and engineering report.

## On-Device Inference Track

**Status:** Backlog — revisit after the Phase 1 server-side foundation, or
earlier only for targeted interview preparation.

### Questions to Answer

- How do device constraints change model and runtime selection?
- When should inference run locally rather than through a remote service?
- How do CPU, integrated GPU, mobile accelerator, and edge NVIDIA targets
  differ?
- How should an offline or air-gapped model be packaged, upgraded, observed,
  and rolled back?
- How can useful diagnostics be collected when engineers cannot access the
  device directly?

### Topics

- `llama.cpp`, GGML, and GGUF
- ONNX Runtime and execution providers
- Core ML at a high level
- TensorRT and NVIDIA edge platforms
- CPU vectorization, threading, and memory bandwidth
- Model conversion and operator compatibility
- Hardware capability detection and CPU fallback
- Cold-start time, binary size, model size, and resident memory
- Power, battery, and thermal limits
- Offline deployment, artifact integrity, versioning, and rollback
- Exportable logs, health information, and field diagnostics

### Candidate Project

Run the same small model locally and on the remote A10. Compare runtime, model
format, precision or quantization, load time, TTFT, output rate, memory use, and
deployment constraints. The result should explain why each environment needs a
different optimization strategy rather than declaring one universally better.

## Quantization Track

Quantization is a general inference optimization, not an on-device-only topic.
It matters on servers, edge devices, laptops, and phones because representing
weights or activations with fewer bits can:

- Reduce model storage and memory footprint.
- Reduce memory traffic and pressure on memory bandwidth.
- Allow larger models, batches, or KV caches within a fixed memory budget.
- Improve throughput or latency when the hardware and inference engine have
  efficient kernels for the selected numerical format.
- Reduce serving cost by increasing useful work per accelerator.

Quantization also has costs: possible quality loss, calibration requirements,
conversion complexity, unsupported operators, extra dequantization work, and
performance that depends on actual hardware and kernel support.

### Phase 1 Scope

- Understand FP32, FP16, BF16, INT8, and INT4 at a high level.
- Establish a reproducible FP16 or BF16 server baseline first.
- Learn the difference between weight-only and weight-plus-activation
  quantization.
- Learn post-training quantization versus quantization-aware training.
- Do not attribute an improvement to quantization without measuring quality,
  memory, latency, and throughput on the target hardware.

### Later Experiment

Compare the baseline model with at least one supported quantized version while
holding prompts, output lengths, sampling settings, warmup, and measurement
procedure constant. Record model quality limitations as well as performance.

## Other Future Topics

- Continuous and dynamic batching
- Paged attention
- Tensor and pipeline parallelism
- Speculative decoding
- Prefix caching
- Multi-GPU serving
- Kubernetes and GPU scheduling
- Production authentication, rate limiting, audit logging, and observability
