#!/usr/bin/env python3
"""Export an injected workbench scenario for the static visual client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_workbench import build_slice_zero_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", choices=("hierarchical", "flat"), default="hierarchical")
    parser.add_argument("--output", type=Path, default=Path("workbench-ui/public/scenario.json"))
    args = parser.parse_args()

    scenario = build_slice_zero_scenario(memory_variant=args.memory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.memory} scenario to {args.output}")


if __name__ == "__main__":
    main()
