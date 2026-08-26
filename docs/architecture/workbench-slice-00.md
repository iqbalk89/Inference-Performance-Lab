# Workbench Slice 0 — Contracts and Interactive Architecture

## Outcome

Slice 0 establishes a working vertical path from composable Python objects to
an interactive system diagram:

```text
injected implementations
        ↓
stable Python contracts
        ↓
serialized scenario
        ↓
React diagram and inspector
```

It deliberately uses fixed Problem 02 facts. Slice 1 will replace the fixed
facts with executable performance calculations.

## Run It

From the repository root:

```bash
make workbench-install
make workbench-dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`.

Interactions:

- click a block or edge to inspect it;
- double-click a block labeled `Push in` to drill down;
- use breadcrumbs to return to an earlier level;
- pan, zoom, and fit the diagram with the canvas controls.

## Current Drill-Down

```text
Inference System
└── Educational GPU
    ├── Prefill
    │   └── Problem 02 [512 × 4096] projection
    └── Decode
        └── Problem 02 [1 × 4096] projection
```

The GPU view separates logical process blocks from physical hardware blocks.
Dashed mapping lines say that a process demands a hardware resource; solid
arrows represent physical or logical data paths.

## Dependency-Injection Design

The composition root is
[`scenario.py`](../../src/inference_workbench/scenario.py). It is the only place
that chooses default implementations. Consumers depend on abstract contracts:

```text
SystemModel
AcceleratorModel
MemoryModel
ComputeModel
```

The default GPU is assembled through composition:

```text
ComposableGPU
├── injected MemoryModel
└── injected ComputeModel
```

The default host system is also composed from injected client, server, CPU,
host-memory, host-link, and accelerator components.

This design follows four rules:

1. Implementations inherit from a narrow abstract contract.
2. Parent models receive child models through their constructors.
3. Variant selection occurs at the composition root or registry.
4. The UI consumes a stable scenario contract and does not import a specific
   GPU, memory, or workload implementation.

## Swapping a Memory Model

Two implementations prove the boundary:

- `HierarchicalMemoryModel`: HBM, controllers, L2, and SM-local memory;
- `FlatMemoryModel`: one intentionally coarse device-memory block.

Run the alternate topology:

```bash
make workbench-export-flat
npm --prefix workbench-ui run dev
```

The inference phases and visual client remain unchanged. Restore the default:

```bash
make workbench-export
```

Programmatic constructor injection is also supported:

```python
scenario = build_slice_zero_scenario(memory_model=my_memory_model)
```

## Swapping Other Components

The composition root accepts:

```python
build_slice_zero_scenario(
    memory_model=my_memory,
    compute_model=my_compute,
    accelerator_model=my_accelerator,
    system_model=my_system,
)
```

This enables later variants such as:

- NVIDIA A10, A100, H100, or future GPU specifications;
- simplified, cache-aware, or profiler-calibrated memory models;
- Tensor Core, general arithmetic, or mixed compute models;
- PCIe, NVLink, or unified-memory host systems;
- single-GPU, multi-GPU, or multi-node topologies.

Named builders can be registered in `VariantRegistry` for configuration-driven
selection. Direct constructor injection remains available for tests and
experiments.

## Stable Visual Contract

The frontend receives:

```text
WorkbenchScenario
├── metadata
└── diagrams
    ├── components
    │   ├── identity and kind
    │   ├── position and lane
    │   ├── metrics and evidence
    │   └── optional drill-down target
    └── connections
        ├── source and target
        ├── direction and category
        └── metrics and evidence
```

Metrics carry an evidence tag:

```text
theoretical | assumed | calibrated | measured
```

Consequently, later model implementations can add calculated values without
changing the UI schema.

## Why This Is Object-Oriented Without Overusing Inheritance

Inheritance defines replaceable capabilities. Composition constructs actual
systems. A GPU does not inherit from HBM or an SM array; it contains injected
memory and compute models. This mirrors the physical system and prevents a
large class hierarchy from coupling unrelated behavior.

Immutable dataclasses represent values such as metrics, components, diagrams,
and connections. Abstract base classes represent behavior and variation.

## Verification

Run:

```bash
make test
make workbench-build
npm --prefix workbench-ui audit --audit-level=high
```

The Slice 0 tests verify:

- the full drill-down graph exists;
- hierarchical and flat memory models can be exchanged;
- new registry variants can be added;
- an entirely different accelerator can be injected;
- system connections adapt to the injected accelerator identity.

## Slice 0 Boundary

Included:

- contracts and evidence types;
- component registries and constructor injection;
- replaceable system, accelerator, compute, and memory models;
- Problem 02 prefill/decode phase models;
- scenario export;
- interactive system, GPU, and phase diagrams;
- component/edge inspector and breadcrumbs;
- automated substitution tests.

Deferred to Slice 1:

- dynamically calculated FLOPs and bytes;
- roofline and ridge-point calculations;
- computed rates and bottleneck status;
- user-editable scenario inputs;
- live model recomputation through an API.
