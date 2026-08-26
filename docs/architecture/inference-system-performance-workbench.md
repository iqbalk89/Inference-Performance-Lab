# Inference System Performance Workbench

## Status

This document is the authoritative product, modeling, and implementation plan
for the repository's central portfolio project.

The workbench will model, measure, visualize, and explain LLM inference across
the workload, GPU, host system, serving system, and eventually distributed
fleet. It is an analytical and empirically calibrated performance estimator—not
a cycle-accurate GPU simulator.

## Career Objective

The project is designed to build evidence for inference performance modeling,
profiling, workload-characterization, and system-dynamics roles such as:

- NVIDIA DGX Cloud AI efficiency and performance engineering;
- Anthropic inference system dynamics and cross-layer performance analysis;
- OpenAI AI-system performance modeling and architecture analysis.

The desired professional capability is:

```text
model → measure → profile → explain → recommend
```

A theoretical model without measurement is a hypothesis. A profiler trace
without a prior prediction is an observation. The workbench must connect both.

## Real-World Questions the Workbench Must Answer

- Will a model and its expected KV cache fit on the target GPU configuration?
- What are the predicted prefill latency, decode latency, TTFT, and TPOT?
- Which operator and hardware resource establish each lower bound?
- Is an operation limited by compute, HBM, cache traffic, communication, or
  overhead?
- How do batch size, prompt length, context length, precision, and concurrency
  change latency, throughput, memory, utilization, and cost?
- Why does measured performance differ from the analytical prediction?
- Which proposed optimization has the largest likely value?
- How does tensor parallelism trade reduced per-GPU compute against collective
  communication?
- How many GPUs are required for a workload and service-level objective?
- What is the latency, capacity, correctness, and cost impact of quantization?

## Non-Goals

The first versions will not attempt to:

- simulate individual GPU clock cycles;
- reproduce undocumented NVIDIA scheduling or cache behavior exactly;
- replace Nsight Systems, Nsight Compute, PyTorch Profiler, or an inference
  server;
- model every Transformer architecture or GPU generation;
- reproduce the internal scheduling implementation of vLLM;
- claim precision unsupported by measurements.

## Guiding Principles

1. **Keep the model independent from the UI.** The same engine must power
   diagrams, reports, tests, and command-line use.
2. **Separate hardware from process.** Prefill and decode are execution phases,
   not physical GPU components.
3. **Join them with explicit execution mappings.** Every operation states what
   hardware it uses, what data moves, and what time bound results.
4. **Separate prefill and decode.** They have different shapes, reuse patterns,
   traffic, and bottlenecks.
5. **Show derivations and units.** Every displayed answer must be explainable.
6. **Label evidence quality.** Values are theoretical, assumed, calibrated, or
   measured.
7. **Prefer useful abstraction to false detail.** Increase fidelity only when
   measurements justify it.
8. **Make recommendations, not only charts.** The final output must connect a
   bottleneck to a quantified decision.

## Conceptual Architecture

The workbench contains three related models:

```text
Logical workload graph
    Transformer phases and operators
                    │
                    │ execution mappings
                    ▼
Physical resource graph
    CPU, memory, buses, GPU resources and interconnects
                    │
                    ▼
Timing and capacity model
    latency, throughput, utilization, capacity and cost
```

### Logical workload graph

Represents what inference performs:

```text
request → tokenization → prefill → repeated decode → response
```

Within a Transformer layer:

```text
normalization
    ↓
Q/K/V projections
    ↓
attention score calculation
    ↓
softmax
    ↓
attention-value mixing
    ↓
output projection
    ↓
normalization
    ↓
MLP projections and activation
```

### Physical resource graph

Represents where work executes and data moves:

```text
host CPU and DRAM
        ↕ PCIe or NVLink
GPU HBM ↔ memory controllers ↔ L2 ↔ on-chip fabric ↔ SM array
                                                        │
                                                        ├── registers
                                                        ├── shared memory/L1
                                                        ├── load/store paths
                                                        ├── Tensor Core paths
                                                        ├── arithmetic paths
                                                        └── special-function paths
```

### Execution mapping

Joins one logical operation to resource demand:

```text
operation
├── tensor shapes and datatypes
├── mathematical work
├── transfers by memory boundary
├── compute-pipeline demand
├── communication demand
├── dependency and overlap assumptions
├── predicted time bounds
└── measured evidence
```

This is the central data structure of the application.

## Hierarchical Visual Experience

The UI uses breadcrumb navigation and progressive drill-down:

```text
System
  → GPU
    → Prefill or Decode
      → Transformer operator
        → Kernel and SM behavior
```

The breadcrumb should remain visible:

```text
System > GPU 0 > Decode > Layer 7 > Q Projection
```

### Level 0 — Inference system

The first screen shows the complete system:

```text
Client → API/server → scheduler → CPU/tokenizer → host DRAM
                                                 │
                                           PCIe/NVLink
                                                 │
                                                 ▼
                                               GPU 0
                                                 │
                                               NVLink
                                                 │
                                               GPU N
```

Initial components:

- request source;
- API/model server;
- scheduler and batcher;
- tokenizer and CPU runtime;
- host memory;
- model storage;
- CPU-to-GPU interconnect;
- one GPU.

Later components:

- multiple GPUs;
- NVLink and NVSwitch;
- network interface;
- InfiniBand or Ethernet;
- remote storage;
- multiple serving replicas.

System-level metrics include request rate, batch size, prompt and output-token
distributions, queueing, tokenization, transfer bytes, transfer time, GPU time,
TTFT, TPOT, end-to-end latency, throughput, concurrency, and cost.

### Level 1 — GPU with process and hardware lanes

Clicking the GPU opens two coordinated lanes:

```text
INFERENCE PROCESS
    Prefill → Decode iteration → sampling → next decode iteration
       │           │
       └──── resource demands ────┐
                                  ▼
GPU HARDWARE
    HBM ↔ memory controllers ↔ L2 ↔ fabric ↔ SM array
```

The upper lane is logical. The lower lane is physical. Selecting a phase
highlights the hardware resources and data paths it uses.

GPU-level metrics include weight residency, KV-cache occupancy, activation and
workspace capacity, HBM traffic, cache traffic, compute demand, arithmetic
intensity, effective rates, utilization, latency contributions, and predicted
bottleneck.

### Level 2 — Prefill

The prefill view follows all prompt rows through every layer and ends when the
first output-token logits are available.

It reports:

- batch size and prompt-token rows;
- tensor shapes by operator;
- FLOPs by operator and layer;
- weight, activation, and KV-write traffic;
- arithmetic intensity;
- Tensor Core eligibility and effective compute rate;
- HBM and cache demand;
- operator and phase latency;
- contribution to TTFT.

### Level 2 — Decode

The decode view represents one token-generation iteration and its repetition.
It includes a context-length control so the user can observe KV-cache capacity,
KV reads, attention traffic, latency, and maximum concurrency change as context
grows.

It reports:

- active sequences and decode batch size;
- current context length;
- FLOPs per generated token;
- weight traffic;
- KV-cache reads and writes;
- arithmetic intensity;
- HBM, cache, and compute demand;
- launch and scheduling overhead when measured;
- TPOT and tokens per second.

### Level 3 — Operator mapped to hardware

Selecting an operator displays its tensors and physical dataflow. For example:

```text
Q projection: [M × K] × [K × N] → [M × N]

weights in HBM ─┐
                ├→ L2 → SM-local storage → Tensor Cores → output → HBM
input in HBM ───┘
```

The detail panel shows:

- tensor shapes and datatypes;
- full formula substitutions;
- FLOPs and MACs;
- bytes crossing each modeled boundary;
- arithmetic intensity at the named boundary;
- compute and memory time bounds;
- roofline position;
- predicted bottleneck;
- percent of phase latency;
- assumptions, confidence, and provenance.

### Level 4 — Kernel and SM

The deepest initial level is driven primarily by profiler evidence:

- grid and thread-block shape;
- block and warp scheduling;
- occupancy;
- registers per thread;
- shared-memory allocation;
- L1/L2/HBM traffic;
- cache hit rates;
- achieved bandwidth;
- Tensor Core or arithmetic-pipeline instructions;
- launch and kernel duration.

The UI compares predicted and measured values side by side rather than claiming
that the analytical model predicts all microarchitectural behavior.

## Visual Semantics

### Nodes

Each block displays a small default metric set. Selecting it opens capacity,
demand, predicted behavior, measured behavior, derivation, assumptions, and
sensitivity analysis.

Suggested colors:

- green: substantial remaining headroom;
- yellow: approaching a modeled limit;
- red: predicted or measured bottleneck;
- blue: selected;
- gray: inactive in the selected phase.

### Edges

Every arrow represents a real data or communication path:

- thickness: bytes transferred;
- animation speed: transfer rate;
- color: utilization or pressure;
- solid: measured;
- dashed: modeled or assumed.

Hover details include direction, data category, bytes, peak rate, effective
rate, utilization, transfer-time contribution, and evidence quality.

### Analysis overlays

The same diagram supports:

```text
Capacity | Traffic | Bandwidth | Latency | Utilization
Bottleneck | Prediction Error | Cost
```

### Operating modes

- **Learn:** component explanations, formulas, units, and animated dataflow.
- **Model:** configurable analytical estimates and bottlenecks.
- **Compare:** GPUs, datatypes, batches, contexts, topologies, or model designs.
- **Profile:** imported measurements, prediction errors, and calibration.

## Modeling Domains

### Model configuration

- layer count;
- hidden and intermediate widths;
- attention and KV-head counts;
- head dimension;
- vocabulary size;
- architecture type;
- weight and KV datatypes;
- quantization metadata assumptions.

### Workload configuration

- prefill or decode;
- batch size;
- prompt length;
- output length;
- current context length;
- concurrency;
- length distributions;
- service-level objectives.

### Hardware configuration

- HBM capacity and bandwidth;
- L2 capacity and modeled/measured bandwidth;
- SM count and clock information;
- register and shared/L1 capacity per SM;
- compute ceilings by datatype and pipeline;
- PCIe/NVLink/network bandwidth;
- topology;
- power and hourly cost when relevant.

### Serving configuration

- static or continuous batching;
- maximum batched tokens;
- chunked prefill;
- scheduling policy;
- tensor/pipeline parallelism;
- replicas and autoscaling target;
- KV-cache allocation policy;
- prefix caching.

## Core Calculations

At minimum, the engine will calculate:

```text
tensor values and bytes
parameter count and weight capacity
operator FLOPs
traffic by named memory boundary
arithmetic intensity
hardware ridge points by datatype
compute-time bound
memory-time bound
communication-time bound
KV-cache capacity and traffic
prefill phase time
decode time per token
TTFT and TPOT
throughput and concurrency
cost per request and token
```

For resource `r`:

```text
T_compute       = FLOPs / effective compute rate
T_memory,r      = bytes crossing r / effective bandwidth of r
T_communication = communication bytes / effective link bandwidth
```

An operation cannot blindly add all memory-level times because transfers can be
nested or overlapped. It also cannot blindly take only the maximum because some
dependencies and overheads serialize. Each model must state its overlap and
dependency assumptions.

## Memory-Hierarchy Fidelity

Memory modeling will advance in stages.

### Stage 1 — Capacity and ideal bandwidth bounds

Model host DRAM, PCIe/NVLink, HBM, L2, shared/L1, and registers using capacity,
required bytes, peak/effective bandwidth, and ideal transfer time.

### Stage 2 — Configurable reuse

Add explicit assumed L2/L1 hit rates, tiling/reuse factors, KV locality, and
traffic propagation between levels.

### Stage 3 — Profiler-calibrated hierarchy

Replace assumptions with measured traffic, hit rates, and achieved bandwidth
where the profiler and provider permit collection.

The UI must never present a guessed cache quantity as measured fact.

## Evidence and Confidence Model

Every important value is tagged:

| Evidence class | Meaning |
| --- | --- |
| Theoretical | Derived from published peak specifications or exact shapes |
| Assumed | User- or model-supplied approximation |
| Calibrated | Adjusted using a controlled microbenchmark |
| Measured | Directly collected for the selected workload |

Each result retains its formula, source inputs, units, measurement source,
timestamp, hardware/software configuration, and uncertainty or limitations.

## Calibration and Validation

The workbench is professionally useful only when it closes this loop:

```text
prediction → benchmark → profiler evidence → error → explanation → revised model
```

Calibration sources include:

- memory-bandwidth microbenchmarks;
- GEMM microbenchmarks by shape and datatype;
- PyTorch Profiler;
- Nsight Systems;
- Nsight Compute;
- inference-server request traces;
- system and GPU telemetry.

The validation view compares predicted and measured HBM traffic, achieved
bandwidth, FLOP/s, kernel time, phase time, TTFT, TPOT, throughput, memory use,
and cost.

## Correctness and Reliability

Performance improvements must not silently damage output quality or service
behavior. Later validation includes:

- numerical comparisons across datatype and hardware changes;
- logits and token-output differences;
- task-level evaluation thresholds;
- performance and correctness regression gates;
- error rates, timeouts, and dropped requests;
- p50, p95, and p99 latency;
- capacity headroom and autoscaling behavior.

## Sensitivity and Decision Support

Each result should answer “what should change?” by sweeping:

- batch size;
- prompt and context length;
- datatype and quantization;
- effective compute and bandwidth;
- cache assumptions;
- tensor-parallel degree;
- scheduling and utilization targets.

Recommendations must quantify benefit, cost, risk, confidence, and the metric
that improves or regresses.

## Technology Architecture

```text
YAML/JSON configurations
          │
          ▼
Python modeling library
          │
          ├── CLI and tests
          ├── report generation
          └── typed FastAPI interface
                         │
                         ▼
React + TypeScript application
          ├── React Flow hierarchical diagrams
          ├── Plotly charts and Sankey flows
          └── SVG/CSS dataflow animation
```

Suggested repository organization:

```text
src/inference_workbench/
├── schemas/
├── models/
│   ├── operators/
│   ├── phases/
│   ├── hardware/
│   ├── memory/
│   ├── serving/
│   └── distributed/
├── calibration/
├── reporting/
└── api/

workbench-ui/
├── src/components/
├── src/diagrams/
├── src/inspectors/
├── src/charts/
└── src/scenarios/

configs/
├── models/
├── hardware/
└── workloads/

tests/
├── unit/
├── analytical/
├── regression/
└── integration/
```

This is a target structure; directories should be created only when their first
working content is implemented.

## Delivery Plan

### Slice 0 — Contracts and static prototype

- Define model, workload, hardware, execution-mapping, and result schemas.
- Define units and evidence tags.
- Create the static System → GPU → operation drill-down.
- Use Problem 02 as a fixed scenario.
- Confirm the visual distinction between process and hardware lanes.

Exit criterion: a user can navigate the hierarchy and explain what every block
and arrow represents.

### Slice 1 — Executable projection model

- Implement the Problem 02 projection calculator.
- Support decode `M=1` and prefill `M=512`.
- Display shapes, FLOPs, bytes, arithmetic intensity, ridge point, compute and
  memory bounds, and bottleneck.
- Show formula provenance in the inspector.
- Add exact analytical tests matching the answer sheet.

Exit criterion: the UI contains no hard-coded calculated answers; changing an
input recomputes every displayed result.

### Slice 2 — One Transformer layer

- Add normalization, Q/K/V, attention, output projection, and MLP.
- Model prefill and decode independently.
- Add KV-cache read/write accounting.
- Attribute phase time and traffic by operator.

Exit criterion: the workbench explains which operators dominate one layer in
each phase and why.

### Slice 3 — Complete single-GPU model

- Scale across all layers.
- Add weight, activation, workspace, and KV capacity.
- Estimate TTFT, TPOT, throughput, and maximum concurrency.
- Add model/GPU/workload configuration files.

Exit criterion: one model/workload/GPU combination produces an auditable
single-GPU report.

### Slice 4 — Measurement and calibration

- Add reproducible GEMM and bandwidth microbenchmarks.
- Import benchmark, Nsight, and framework measurements.
- Compare predicted and measured quantities.
- Calculate error and fit explicit efficiency factors.
- Write the first root-cause report.

Exit criterion: at least one prediction error is explained with profiler
evidence and the calibrated model improves held-out predictions.

### Slice 5 — Serving dynamics

- Add continuous batching, concurrency, queueing, and context distributions.
- Model capacity, TTFT/TPOT distributions, utilization, and cost.
- Add sensitivity studies for batch and utilization targets.

Exit criterion: the tool recommends a serving configuration under an explicit
latency/cost objective.

### Slice 6 — Distributed inference

- Add tensor-parallel compute partitioning.
- Add collective communication and overlap assumptions.
- Model NVLink/NVSwitch and later node networking.
- Compare scale-up and scale-out configurations.

Exit criterion: the model explains when another GPU helps, when communication
dominates, and which topology is preferable under stated assumptions.

### Slice 7 — Regression, correctness, and reliability

- Add baseline comparisons and automated regression thresholds.
- Add numerical and output-correctness evaluation.
- Add tail latency, errors, and capacity-headroom views.
- Produce release-style pass/fail evidence.

Exit criterion: an optimization cannot be accepted based only on mean speed.

## Portfolio Studies

1. Decode versus prefill roofline and data-reuse analysis.
2. Batch-size effects on weight reuse, throughput, TPOT, and capacity.
3. Context-length effects on KV traffic, memory, and decode latency.
4. Precision effects on capacity, ridge point, speed, and correctness.
5. Predicted versus measured hierarchical roofline study.
6. Tensor-parallel scaling and communication crossover.
7. End-to-end latency-gap investigation from request to kernel.
8. Latency-cost-capacity recommendation for a serving target.

Every study follows the repository experiment standard and ends with an
evidence-backed recommendation.

## Role Alignment

### NVIDIA-style performance engineering

Emphasize controlled benchmarks, workload characterization, GPU/system
profiling, regressions, automation, compute/network/storage bottlenecks, and
clear optimization reports.

### Anthropic-style inference system dynamics

Emphasize theoretical-frontier gaps, cross-layer tracing, FLOPs funnels,
serving and batching, cost, observability, tail latency, reliability,
correctness, and opportunity ranking.

### OpenAI-style performance modeling

Emphasize reusable framework design, multiple abstraction levels, compute and
memory balance, topology and communication, sensitivity studies, architecture
comparisons, model validation, and actionable system-design recommendations.

## Preparation Allocation

The project should guide study effort approximately as follows:

```text
30% analytical modeling
30% benchmarking and GPU profiling
20% inference serving and distributed systems
10% data analysis and observability
10% technical reports and recommendations
```

The visual interface is important for comprehension and communication, but it
must not displace model correctness, reproducibility, measurement, or analysis.

## Definition of Professional Quality

The workbench is portfolio-ready when it can demonstrate:

1. an explicit prediction made before measurement;
2. reproducible benchmark and environment configuration;
3. profiler evidence tied to a specific claim;
4. an explanation of prediction error;
5. a controlled optimization experiment;
6. latency, throughput, memory, capacity, cost, and correctness tradeoffs;
7. a quantified recommendation with assumptions and confidence;
8. concise written communication understandable across teams.

## Immediate Next Step

Implement Slice 0 before expanding the curriculum or adding full-model
formulas. The first scenario will be the existing decode-versus-prefill
projection problem, which already provides verified equations and expected
values. This produces a narrow, testable vertical slice through the complete
architecture without prematurely modeling undocumented hardware behavior.
