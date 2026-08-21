#!/usr/bin/env python3
"""Benchmark the explicitly separated inference phases and emit JSON."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from model import choose_device, run_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2", help="Hugging Face model ID or local path")
    parser.add_argument("--prompt", default="Cats chase mice")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output", type=Path, default=Path("results/measurements.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.runs < 1:
        raise SystemExit("warmups must be >= 0 and runs must be >= 1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = choose_device(torch, args.device)
    print(f"Loading {args.model!r} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    for index in range(args.warmups):
        print(f"Warm-up {index + 1}/{args.warmups}")
        run_request(model, tokenizer, torch, args.prompt, args.max_new_tokens, device)

    measurements = []
    for index in range(args.runs):
        result = run_request(model, tokenizer, torch, args.prompt, args.max_new_tokens, device)
        measurements.append(result.to_dict())
        print(
            f"Run {index + 1}/{args.runs}: prompt={result.prompt_tokens} tokens, "
            f"TTFT={result.ttft_model_boundary_ms:.2f} ms, "
            f"generation={result.generation_model_boundary_ms:.2f} ms"
        )

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "warmups": args.warmups,
            "runs": args.runs,
            "device": device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

