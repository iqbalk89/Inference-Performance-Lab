#!/usr/bin/env python3
"""Summarize benchmark JSON without external analysis dependencies."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(name: str, values: list[float]) -> None:
    print(
        f"{name:30s} median={statistics.median(values):9.3f} ms  "
        f"p90={percentile(values, 0.90):9.3f} ms  min={min(values):9.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("results/measurements.json"))
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    rows = payload["measurements"]
    if not rows:
        raise SystemExit("No measurements found")

    describe("tokenize", [row["tokenize_ms"] for row in rows])
    describe("prefill", [row["prefill_ms"] for row in rows])
    describe("first token selection", [row["first_sample_ms"] for row in rows])
    describe("TTFT (model boundary)", [row["ttft_model_boundary_ms"] for row in rows])
    describe("generation (model boundary)", [row["generation_model_boundary_ms"] for row in rows])

    decode_steps = [value for row in rows for value in row["decode_forward_ms"]]
    if decode_steps:
        describe("decode forward step", decode_steps)


if __name__ == "__main__":
    main()

