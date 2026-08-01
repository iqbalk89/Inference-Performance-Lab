# Lambda A10 Environment Acceptance

**Date:** August 1, 2026

## Objective

Determine whether a Lambda On-Demand Cloud A10 instance can support the CUDA,
PyTorch, Nsight Systems, and Nsight Compute work required for Phase 1.

## Environment

| Component | Value |
| --- | --- |
| Provider | Lambda On-Demand Cloud |
| GPU | NVIDIA A10 |
| Usable GPU memory | 23,028 MiB |
| GPU architecture | Ampere (GA10x) |
| Operating system | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-1046-nvidia |
| NVIDIA driver | 580.105.08 |
| Driver-reported CUDA | 13.0 |
| Python | 3.10.12 |
| PyTorch | 2.7.0 |
| PyTorch-compiled CUDA | 12.8 |
| Nsight Systems | 2026.4.1.191 (official NVIDIA CLI) |
| Nsight Compute | 2025.1.1.0 |

The ephemeral public IP is intentionally not recorded.

## Tests and Results

### CUDA and PyTorch

PyTorch reported:

- CUDA available: `True`
- Device count: `1`
- Device: `NVIDIA A10`

The reusable `scripts/gpu_smoke_test.py` workload successfully executed CUDA
matrix multiplications. A 4096 by 4096 matrix multiplication took approximately
11 milliseconds in the initial smoke test. This number is an environment check,
not a controlled benchmark.

### Nsight Systems CUDA Trace

Nsight Systems collected CUDA, NVTX, and GPU workload events and generated a
viewable `.nsys-rep` report. A statistics export successfully summarized CUDA
kernels and CUDA API calls.

### Nsight Systems GPU Metrics

The profiler recognized GPU 0 as:

```text
NVIDIA A10 PCI[0000:06:00.0]
```

It selected:

```text
General Metrics for NVIDIA GA10x
```

GPU Metrics collection completed without a counter-permission error. This
confirms access to the device-level performance counters required for SM,
Tensor Core, and bandwidth metrics.

### Nsight Compute

Nsight Compute profiled a CUDA kernel over eight replay passes and generated a
valid `.ncu-rep` report. It did not produce `ERR_NVGPUCTRPERM`.

## Limitations Discovered

### CPU Sampling

`nsys status --environment` reported:

- `perf_event_paranoid=4`
- `perf_event_open` unavailable
- CPU process-tree and system-wide sampling unavailable

CUDA tracing and NVIDIA GPU counter collection still worked when CPU sampling
and context-switch tracing were disabled.

### Resolved: Nsight Systems Report Importer

Lambda's preinstalled package repository supplied Nsight Systems 2024.6.2 with
an internal `libssh` symbol version mismatch:

```text
LIBSSH_4_9_0 not found
```

That collector produced raw `.qdstrm` files but could not convert them to
`.nsys-rep`.

The issue was resolved by adding NVIDIA's official developer-tools repository
and installing the separate `nsight-systems-cli` 2026.4.1 package. The corrected
CLI produced `nsys-accepted.nsys-rep` and generated CUDA kernel and API
statistics successfully.

Future Lambda instances should run:

```bash
./scripts/bootstrap_lambda_gpu.sh
```

The script installs the verified official CLI alongside Nsight Compute and
prints the resulting GPU and profiler versions.

## Saved Artifacts

The following artifacts were copied to the local ignored `profile-results/`
directory before instance termination:

- Basic Nsight Systems CUDA trace
- Nsight Systems GPU Metrics trace
- Corrected, viewable Nsight Systems report
- Nsight Compute report

Raw profiler artifacts are kept out of Git. Reproducible scripts and summarized
results are committed.

## Decision

Lambda's A10 environment is accepted for Phase 1 GPU work because:

- CUDA-enabled PyTorch works.
- The A10 is directly visible to the workload.
- Nsight Systems can collect CUDA activity.
- Nsight Systems can access GA10x GPU Metrics.
- Nsight Compute can access per-kernel hardware counters.

The original Lambda-package importer issue has been resolved in the reusable
bootstrap workflow.
