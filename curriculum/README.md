# Inference Performance Curriculum

This directory is the canonical learning path for the repository. Work through
phases in order. Each phase is divided into numbered modules, and each module
uses the same structure:

1. **Learn** — concepts and a deliberately limited resource list
2. **Check** — questions and short reasoning exercises
3. **Build** — a lab with required evidence and output locations
4. **Reflect** — conclusions recorded in the learning journal

The curriculum describes what to learn and build. Production-style artifacts
live in the repository's engineering directories:

| Curriculum activity | Engineering output |
| --- | --- |
| Focused investigation | `experiments/` |
| Profiling configuration and summaries | `profiling/` |
| Repeatable performance measurement | `benchmarks/` |
| Model-serving implementation | `inference-servers/` |
| Reusable code | `src/` |
| Automation | `scripts/` |
| Automated verification | `tests/` |
| Engineering reports and durable notes | `docs/` |

## Phases

| Phase | Topic | Status |
| --- | --- | --- |
| [Phase 0](phase-00-development-environment/README.md) | Development environment and workstation setup | Complete |
| [Phase 1](phase-01-gpu-inference-foundations/README.md) | GPU foundations and inference fundamentals | In progress |

## Job-Oriented Performance Track

The [Inference Performance Engineering track](inference-performance-engineering/README.md)
reorganizes the material around the workflow used in modeling and profiling
roles: predict, measure, profile, explain, optimize, and remeasure. It preserves
the phase curriculum as a reference library while providing a clearer primary
path through end-to-end inference, prefill, decode, capacity, and optimization.

Future phases will be added only when their scope is defined. Deferred topics
remain in the [backlog](../docs/backlog.md) rather than appearing as unfinished
curriculum.

## How to Resume

1. Open the active phase README.
2. Find the first module not marked complete.
3. Read its lesson before opening the lab.
4. Complete the exercises without notes where requested.
5. Perform the lab and save evidence in the specified output paths.
6. Update the module checklist and `docs/learning-journal.md`.

Do not launch paid GPU infrastructure for concept-only modules.
