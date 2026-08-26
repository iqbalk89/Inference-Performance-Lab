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

It uses the Problem 02 scenario as its first executable analytical model. The
calculation engine, rather than the UI, produces the displayed FLOPs, traffic,
arithmetic intensity, time bounds, path rates, and bottleneck classification.

## Run It

From the repository root:

```bash
make workbench-install
make workbench-dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`.

Interactions:

- click a block or edge to inspect it;
- click any inspector metric labeled `Open full calculation` for its complete
  beginner-oriented derivation;
- double-click a block labeled `Push in` to drill down;
- use breadcrumbs to return to an earlier level;
- pan, zoom, and fit the diagram with the canvas controls.

## Current Drill-Down

```text
Inference System
└── Educational GPU
    ├── Prefill
    │   └── Level 1: Problem 02 operator math
    │       └── Level 2: HBM-boundary ledger
    │           └── Level 3: physical GPU execution path
    └── Decode
        └── Level 1: Problem 02 operator math
            └── Level 2: HBM-boundary ledger
                └── Level 3: physical GPU execution path
```

The GPU view separates logical process blocks from physical hardware blocks.
Dashed mapping lines say that a process demands a hardware resource; solid
arrows represent physical or logical data paths.

The prefill and decode projections use progressive views instead of combining
multiple abstraction levels on one canvas.

Level 1 answers, “What mathematics occurs?”

```text
X tensor ──┐
           ├──▶ matrix multiplication ──▶ Y tensor
W tensor ──┘
```

Level 2 answers, “How does the operator execute, and which bytes does the
simplified model count at the HBM boundary?” The matrix multiplication remains
in the center; the HBM reads and output write are attached to it rather than
replacing it.

```text
             ┌──▶ read X ──▶┐
HBM boundary ├──▶ read W ──▶│ matrix multiplication ──▶ write Y ──▶ HBM
             └──────────────┘             │
                                  ─ ─ ─ ▶ roofline accounting
```

Level 3 answers, “Through which physical resources can data travel?”

```text
HBM ◀──▶ shared L2 ◀──▶ SM-local storage ◀──▶ Tensor Core pipelines
```

The `Y → HBM` direction is explicit in Level 2 because output bytes contribute
to total traffic, arithmetic intensity, and the memory-time bound. Level 3
introduces L2 only when discussing the physical route. Its hit rate and
internal traffic are displayed as `Unknown—measure or calibrate`; the workbench
does not invent cache behavior merely to fill the diagram.

This is intentionally a boundary-level model. It does not claim that data
physically jumps directly from HBM into a mathematical tensor node. Actual
transactions travel through memory controllers and caches; those details are
shown in Level 3 and will be quantified when cache behavior is
modeled or measured.

Path badges are intentionally compact:

```text
bytes · bandwidth · ideal transfer bound
```

Names such as `HBM channels` and `memory fabric` live in the clickable path
inspector instead of competing with adjacent blocks on the schematic.

The HBM-boundary view also includes an interactive roofline sensitivity chart.
It sweeps token-row count `M` and recomputes arithmetic intensity, ideal
compute time, ideal memory time, and the roofline lower bound. The controls let
you change `M`, assumed HBM bandwidth, and assumed peak FP16 compute. This makes
the model useful for sensitivity analysis instead of only displaying one
hard-coded prefill or decode point. The chart remains an ideal model and does
not claim measured kernel latency.

## Host and GPU Ownership in the Full Pipeline

The system-level diagram owns the host-side stages and the boundary transfer:

```text
CPU: request text → tokenizer → host token IDs
                                      │
                              PCIe / host link
                                      ▼
GPU HBM: embedding table → GPU gather → hidden states
                                      ▼
                         Transformer layers → logits → selection
```

Embedding lookup normally runs on the GPU. The embedding table is a model
weight stored in GPU memory, and the GPU executes a gather kernel that selects
the row for each token ID. The CPU performs tokenization and orchestration. A
deployment may offload embeddings or sampling, but those are explicit variant
choices rather than assumptions hidden inside the base diagram.

The GPU phase diagrams begin at `Device token IDs`, after the boundary transfer.
Prefill receives the prompt IDs produced by the system-level host path. Decode
uses the common device-resident loop in which the newly selected token ID is
fed directly into the next embedding lookup; it does not incorrectly tokenize
the generated ID again on the CPU.

## Calculation Drill-Down

Every derived projection metric carries structured calculation provenance. Its
detail view includes:

1. what is being calculated and why it matters;
2. the general formula;
3. every symbol, value, meaning, and source;
4. numbered substitution and arithmetic steps;
5. explicit unit cancellation;
6. a plain-language interpretation;
7. assumptions and limitations.

Current complete walkthroughs cover work/FLOPs, input/weight/output bytes, total
HBM traffic, arithmetic intensity, ridge point, compute and memory time bounds,
roofline lower bound, bottleneck classification, and individual path-transfer
bounds.

## Dependency-Injection Design

The composition root is
[`scenario.py`](../../src/inference_workbench/scenario.py). It is the only place
that chooses default implementations. Consumers depend on abstract contracts:

```text
SystemModel
AcceleratorModel
MemoryModel
ComputeModel
PhaseModel
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

Projection diagrams are also assembled from reusable objects in
[`blocks.py`](../../src/inference_workbench/blocks.py):

- `TensorBlock` represents a logical tensor such as `X`, `W`, or `Y`;
- `OperationBlock` represents mathematical work such as matrix multiplication;
- `AnalysisBlock` represents a performance model or diagnostic such as roofline
  accounting; it is deliberately not an inference-pipeline operation;
- `ResourceBlock` represents physical hardware such as HBM, L2, or an SM;
- `Path` constructs consistently styled logical, transfer, physical, and
  operation-to-hardware mapping connections.

The prefill and decode views instantiate the same block classes. Future
operators, GPU variants, and memory models can therefore replace their data and
composition without duplicating frontend markup or diagram-specific classes.

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
    prefill_model=my_prefill,
    decode_model=my_decode,
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
- projection diagrams use reusable semantic block types;
- projection output has an explicit, calculated `Y → HBM` write path;
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

- user-editable scenario inputs;
- live model recomputation through an API.
- complete operator coverage beyond the first projection;
- GPU-profile selection from configuration.
