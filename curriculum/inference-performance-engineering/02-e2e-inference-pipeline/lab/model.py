#!/usr/bin/env python3
"""Run one explicitly separated prefill/decode request.

This is teaching code, not a production serving implementation. Its purpose is
to make phase boundaries visible in measurements and profiler timelines.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class RequestMeasurement:
    prompt_tokens: int
    output_tokens: int
    tokenize_ms: float
    prefill_ms: float
    first_sample_ms: float
    decode_forward_ms: list[float] = field(default_factory=list)
    decode_sample_ms: list[float] = field(default_factory=list)
    ttft_model_boundary_ms: float = 0.0
    generation_model_boundary_ms: float = 0.0
    output_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(torch, device: str) -> None:
    """Wait for asynchronous device work before reading the CPU clock."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@contextmanager
def annotated_range(torch, device: str, name: str) -> Iterator[None]:
    """Create PyTorch and, on CUDA, NVTX profiler annotations."""
    with torch.profiler.record_function(name):
        if device.startswith("cuda"):
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if device.startswith("cuda"):
                torch.cuda.nvtx.range_pop()


def timed_call(torch, device: str, name: str, function):
    synchronize(torch, device)
    start = perf_counter()
    with annotated_range(torch, device, name):
        result = function()
    synchronize(torch, device)
    return result, (perf_counter() - start) * 1_000.0


def run_request(model, tokenizer, torch, prompt: str, max_new_tokens: int, device: str) -> RequestMeasurement:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    encoded, tokenize_ms = timed_call(
        torch,
        device,
        "phase/tokenize",
        lambda: tokenizer(prompt, return_tensors="pt"),
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.inference_mode():
        prefill_output, prefill_ms = timed_call(
            torch,
            device,
            "phase/prefill_forward",
            lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True),
        )

        next_token, first_sample_ms = timed_call(
            torch,
            device,
            "phase/first_token_selection",
            lambda: torch.argmax(prefill_output.logits[:, -1, :], dim=-1, keepdim=True),
        )

        generated = [next_token]
        past_key_values = prefill_output.past_key_values
        decode_forward_ms: list[float] = []
        decode_sample_ms: list[float] = []

        # The first output ID was selected from prefill logits. Each later ID
        # requires one decode forward pass that processes the preceding ID.
        for step in range(1, max_new_tokens):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            decode_output, forward_ms = timed_call(
                torch,
                device,
                f"phase/decode_forward_{step}",
                lambda: model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                ),
            )
            next_token, sample_ms = timed_call(
                torch,
                device,
                f"phase/decode_selection_{step}",
                lambda: torch.argmax(decode_output.logits[:, -1, :], dim=-1, keepdim=True),
            )
            past_key_values = decode_output.past_key_values
            generated.append(next_token)
            decode_forward_ms.append(forward_ms)
            decode_sample_ms.append(sample_ms)

    generated_ids = torch.cat(generated, dim=1)[0].detach().cpu().tolist()
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    ttft = tokenize_ms + prefill_ms + first_sample_ms
    generation = ttft + sum(decode_forward_ms) + sum(decode_sample_ms)
    return RequestMeasurement(
        prompt_tokens=int(input_ids.shape[1]),
        output_tokens=len(generated_ids),
        tokenize_ms=tokenize_ms,
        prefill_ms=prefill_ms,
        first_sample_ms=first_sample_ms,
        decode_forward_ms=decode_forward_ms,
        decode_sample_ms=decode_sample_ms,
        ttft_model_boundary_ms=ttft,
        generation_model_boundary_ms=generation,
        output_text=output_text,
    )

