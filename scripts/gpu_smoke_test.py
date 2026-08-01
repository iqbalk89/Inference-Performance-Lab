"""Run a small CUDA matrix-multiplication workload for environment checks."""

from __future__ import annotations

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)

    with torch.cuda.nvtx.range("warmup"):
        for _ in range(args.warmup):
            result = a @ b
    torch.cuda.synchronize()

    started = time.perf_counter()
    with torch.cuda.nvtx.range("measured_matmul"):
        for _ in range(args.iterations):
            result = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"matrix_size={args.size}")
    print(f"iterations={args.iterations}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"milliseconds_per_matmul={elapsed * 1000 / args.iterations:.3f}")
    print(f"result_sample={result[0, 0].item():.6f}")


if __name__ == "__main__":
    main()
