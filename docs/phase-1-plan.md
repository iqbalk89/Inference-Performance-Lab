# Phase 1 — GPU Foundation and Inference Fundamentals

## Timeline

Weeks 1–5

## Objective

Understand how a decoder-only language model generates tokens while becoming
comfortable developing, observing, and measuring inference workloads on NVIDIA
GPUs.

By the end of this phase, the repository should contain a minimal inference
service, repeatable benchmarks, profiler evidence, and an initial engineering
report.

## Hardware Strategy

The local development machine is an Intel-based MacBook Pro with Intel and AMD
graphics. It does not contain an NVIDIA GPU and therefore cannot run CUDA,
`nvidia-smi`, CUDA-enabled PyTorch, or NVIDIA profiling tools.

Work is divided between two environments:

### Local Intel Mac

- VS Code, Git, and GitHub workflows
- Python development and automated tests
- Docker and API testing
- Tokenization and transformer experiments
- CPU-based correctness checks
- Documentation and analysis
- Remote development and orchestration

### Remote Linux NVIDIA Host

- CUDA-enabled PyTorch
- Model weights and GPU inference
- `nvidia-smi`
- PyTorch Profiler
- Nsight Systems
- GPU utilization and VRAM measurements
- Performance benchmarks

CUDA and NVIDIA drivers must not be installed on the Mac. GPU claims and
measurements will only be recorded when they are produced on the remote NVIDIA
host.

The first remote environment has been accepted on a Lambda On-Demand Cloud A10
instance. CUDA, GPU Metrics, Nsight Compute counters, and elevated CPU
call-stack sampling were verified. See
[Lambda A10 Environment Acceptance](gpu-notes/lambda-a10-acceptance.md).
Current progress and the exact resumption point are recorded in the
[Phase 1 Checkpoint](phase-1-checkpoint.md).

## Part A — GPU and Environment

### Learn

- GPU architecture
- CUDA fundamentals
- CUDA runtime
- Streaming Multiprocessors
- CUDA cores and Tensor Cores
- Memory hierarchy and memory bandwidth
- VRAM
- `nvidia-smi`
- PyTorch Profiler
- Nsight Systems at an introductory level

### Build

- Obtain access to a remote Linux machine with an NVIDIA GPU.
- Connect to the host through SSH and VS Code Remote SSH.
- Create a reproducible CUDA-enabled Python environment.
- Verify GPU execution with PyTorch.
- Run the first model on the GPU.
- Observe utilization and VRAM with `nvidia-smi`.
- Capture the first PyTorch Profiler trace.
- Capture an introductory Nsight Systems trace if supported.

### Required GPU Verification

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

The remote environment record must include:

- GPU model and VRAM capacity
- Operating system
- NVIDIA driver version
- Reported CUDA version
- Python version
- PyTorch version
- Container image or dependency versions

## Part B — Inference Fundamentals

### Learn

- Transformer architecture
- Tokenization
- Decoder-only models
- Causal self-attention
- Autoregressive token generation
- KV cache
- Prefill versus decode
- Sampling
- Memory growth during inference
- Time to first token versus throughput

### Build — Inference Performance Lab v1

Create a minimal inference service with:

- One small decoder-only language model
- One HTTP generation endpoint
- Configurable prompt and output lengths
- Explicit generation parameters
- Structured request and result data
- Request timing
- Basic automated tests

Measure at least:

- Time to first token (TTFT)
- Time per output token (TPOT)
- End-to-end latency
- Input token count
- Output token count
- Output tokens per second
- Peak GPU memory usage

## Weekly Sequence

### Week 1 — GPU Concepts and Remote Environment

- Learn GPU, SM, CUDA core, Tensor Core, VRAM, and memory-bandwidth concepts.
- Select and provision a remote NVIDIA GPU.
- Connect with SSH and VS Code Remote SSH.
- Run `nvidia-smi` and verify CUDA-enabled PyTorch.
- Record the complete remote environment.

### Week 2 — First Model and Profiler Trace

- Run a small decoder-only model on the remote GPU.
- Observe utilization and memory with `nvidia-smi`.
- Capture and inspect a PyTorch Profiler trace.
- Introduce Nsight Systems and capture a trace if practical.

### Week 3 — Inference Mechanics

- Trace text through tokenization, embeddings, transformer blocks, and logits.
- Study causal attention and decoder-only generation.
- Compare prefill and decode behavior.
- Observe KV-cache creation and growth.
- Experiment with deterministic and stochastic sampling.

### Week 4 — Inference Performance Lab v1

- Build the minimal inference HTTP service.
- Add request timing and structured measurements.
- Add automated correctness tests.
- Record TTFT, TPOT, latency, throughput, and peak memory.

### Week 5 — Benchmark and Engineering Report

- Compare short and long prompts.
- Compare short and long generated outputs.
- Compare cold and warm requests.
- Test batch size where supported.
- Measure KV-cache memory growth.
- Compare prefill-heavy and decode-heavy workloads.
- Write and review the first engineering report.

## Engineering Report Standard

The Phase 1 report will contain:

1. Objective
2. Hypothesis
3. Environment and dependencies
4. Methodology
5. Results
6. Analysis
7. Limitations
8. Lessons learned

Measurements must include explicit units and enough environment information to
be reproduced. Raw measurements should remain separate from interpretation.

## Deliverables

- [x] Remote NVIDIA environment available
- [x] CUDA-enabled PyTorch verified
- [ ] First GPU model executed
- [x] GPU utilization and VRAM measurements captured
- [ ] PyTorch Profiler trace captured
- [x] Initial Nsight Systems and Nsight Compute traces captured
- [ ] Nsight Systems introduction completed
- [ ] Local or remotely accessible inference server
- [ ] Latency and throughput benchmarks
- [ ] Initial engineering report
- [ ] GPU and inference documentation

## Milestone Explanations

At completion, the following should be explainable without relying on memorized
definitions:

- What happens from input text to each generated token
- Why prefill becomes expensive as prompt length grows
- What the KV cache stores and why it improves decoding
- How KV-cache memory grows
- Why TTFT and output-token throughput measure different behavior
- How model weights, activations, and the KV cache use GPU memory
- How to verify that a workload is actually using the GPU

## Interview and Networking Status

- Interview target: no applications or interviews during this phase.
- Networking: reconnect with contacts at OpenAI and NVIDIA to learn about their
  teams and work, not to request referrals.

## Immediate Next Step

Run the first small decoder-only Hugging Face model directly through PyTorch on
the accepted Lambda A10 environment. Pin the model and dependencies, record
generation and memory metadata, observe `nvidia-smi`, and capture the first
model-level PyTorch Profiler trace. Follow the
[Phase 1 Checkpoint](phase-1-checkpoint.md) rather than repeating provider
selection or profiler-access research.
