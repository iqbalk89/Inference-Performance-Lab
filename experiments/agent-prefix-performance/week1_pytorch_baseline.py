#!/usr/bin/env python3
"""Measure prefill, decode, and KV-cache behavior with direct PyTorch."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = Path("benchmark-results/agent-prefix-performance/week1")
PROMPT_SEED = (
    "You are a careful systems assistant. Analyze the inference request using "
    "only the supplied context, state assumptions, and return a concise answer. "
    "The stable agent context contains tool descriptions, policies, examples, "
    "and application state. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-no-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name in ("prompt_tokens", "new_tokens", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    return args


def cuda_elapsed_ms(operation: Callable[[], Any]) -> tuple[Any, float]:
    """Run an operation and measure GPU work with synchronized CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def make_exact_length_inputs(tokenizer: Any, token_count: int) -> dict[str, torch.Tensor]:
    repetitions = 1
    while True:
        encoded = tokenizer(
            PROMPT_SEED * repetitions,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] >= token_count:
            return {
                "input_ids": encoded["input_ids"][:, :token_count].contiguous(),
                "attention_mask": encoded["attention_mask"][:, :token_count].contiguous(),
            }
        repetitions *= 2


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def cache_tensors(cache: Any) -> list[torch.Tensor]:
    """Collect unique tensors from legacy tuples or Transformers Cache objects."""
    found: list[torch.Tensor] = []
    seen_objects: set[int] = set()

    def visit(value: Any) -> None:
        object_id = id(value)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if torch.is_tensor(value):
            found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        else:
            for attribute in ("key_cache", "value_cache", "layers"):
                child = getattr(value, attribute, None)
                if child is not None:
                    visit(child)

    visit(cache)
    return found


def summarize_cache(cache: Any) -> dict[str, Any]:
    tensors = cache_tensors(cache)
    return {
        "type": type(cache).__name__,
        "tensor_count": len(tensors),
        "total_bytes": sum(tensor_bytes(tensor) for tensor in tensors),
        "first_tensor_shapes": [list(tensor.shape) for tensor in tensors[:4]],
        "first_tensor_dtypes": [str(tensor.dtype) for tensor in tensors[:4]],
    }


@torch.inference_mode()
def generate_cached(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    new_tokens: int,
) -> dict[str, Any]:
    outputs, prefill_ms = cuda_elapsed_ms(
        lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    )
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    cache = outputs.past_key_values
    initial_cache = summarize_cache(cache)

    def decode() -> None:
        nonlocal cache, next_token, attention_mask
        for _ in range(new_tokens - 1):
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))),
                dim=1,
            )
            step = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = step.past_key_values
            next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

    _, decode_ms = cuda_elapsed_ms(decode)
    total_ms = prefill_ms + decode_ms
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "output_tokens": new_tokens,
        "output_tokens_per_second": new_tokens / (total_ms / 1000),
        "decode_tokens_per_second": (
            (new_tokens - 1) / (decode_ms / 1000) if new_tokens > 1 and decode_ms > 0 else None
        ),
        "logits_shape": list(outputs.logits.shape),
        "initial_kv_cache": initial_cache,
        "final_kv_cache": summarize_cache(cache),
        "generated_ids": torch.cat(generated, dim=1).cpu().tolist()[0],
    }


@torch.inference_mode()
def generate_without_cache(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    new_tokens: int,
) -> dict[str, Any]:
    sequence = input_ids
    mask = attention_mask
    generated: list[torch.Tensor] = []

    def full_recomputation() -> None:
        nonlocal sequence, mask
        for _ in range(new_tokens):
            outputs = model(input_ids=sequence, attention_mask=mask, use_cache=False)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            sequence = torch.cat((sequence, next_token), dim=1)
            mask = torch.cat((mask, mask.new_ones((mask.shape[0], 1))), dim=1)

    _, elapsed_ms = cuda_elapsed_ms(full_recomputation)
    return {
        "total_ms": elapsed_ms,
        "output_tokens": new_tokens,
        "output_tokens_per_second": new_tokens / (elapsed_ms / 1000),
        "generated_ids": torch.cat(generated, dim=1).cpu().tolist()[0],
    }


def median_metrics(runs: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = ("prefill_ms", "decode_ms", "total_ms", "output_tokens_per_second")
    summary: dict[str, float | None] = {}
    for key in keys:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        summary[f"median_{key}"] = statistics.median(values) if values else None
    return summary


def driver_version() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This experiment requires an NVIDIA GPU with CUDA-enabled PyTorch.")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
    ).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    cpu_inputs = make_exact_length_inputs(tokenizer, args.prompt_tokens)
    inputs = {name: tensor.to("cuda") for name, tensor in cpu_inputs.items()}
    parameter_bytes = sum(tensor_bytes(parameter) for parameter in model.parameters())

    for _ in range(args.warmups):
        generate_cached(model, **inputs, new_tokens=min(args.new_tokens, 4))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cached_runs = [
        generate_cached(model, **inputs, new_tokens=args.new_tokens)
        for _ in range(args.repeats)
    ]
    cached_peak = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }

    uncached_runs: list[dict[str, Any]] = []
    uncached_peak: dict[str, int] | None = None
    if not args.skip_no_cache:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        uncached_runs = [
            generate_without_cache(model, **inputs, new_tokens=args.new_tokens)
            for _ in range(args.repeats)
        ]
        uncached_peak = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    resolved_revision = getattr(model.config, "_commit_hash", None)
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "driver": driver_version(),
        },
        "model": {
            "id": args.model,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "dtype": args.dtype,
            "load_seconds": load_seconds,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_bytes": parameter_bytes,
        },
        "workload": {
            "batch_size": int(inputs["input_ids"].shape[0]),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "new_tokens": args.new_tokens,
            "input_ids_shape": list(inputs["input_ids"].shape),
            "attention_mask_shape": list(inputs["attention_mask"].shape),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "decoding": "greedy argmax",
        },
        "cached": {
            "summary": median_metrics(cached_runs),
            "peak_cuda_memory": cached_peak,
            "runs": cached_runs,
        },
        "without_cache": None if args.skip_no_cache else {
            "summary": median_metrics(uncached_runs),
            "peak_cuda_memory": uncached_peak,
            "runs": uncached_runs,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / (
        f"pytorch-p{args.prompt_tokens}-o{args.new_tokens}-{stamp}.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(output_path),
        "resolved_revision": resolved_revision,
        "cached_summary": result["cached"]["summary"],
        "without_cache_summary": (
            None if result["without_cache"] is None else result["without_cache"]["summary"]
        ),
        "initial_kv_cache": cached_runs[0]["initial_kv_cache"],
        "cached_peak_cuda_memory": cached_peak,
    }, indent=2))


if __name__ == "__main__":
    main()
