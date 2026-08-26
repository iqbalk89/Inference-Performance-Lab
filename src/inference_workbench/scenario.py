"""Composition root: dependencies are selected and assembled only here."""

from __future__ import annotations

from .contracts import AcceleratorModel, Component, ComponentKind, ComputeModel, MemoryModel, Metric, Position, SystemModel, VariantRegistry, WorkbenchScenario
from .hardware import ComposableGPU, FlatMemoryModel, HierarchicalMemoryModel, SMArrayComputeModel
from .system import SingleAcceleratorSystem
from .workloads import ProjectionPhaseModel


memory_variants: VariantRegistry = VariantRegistry()
memory_variants.register("hierarchical", HierarchicalMemoryModel)
memory_variants.register("flat", FlatMemoryModel)

compute_variants: VariantRegistry = VariantRegistry()
compute_variants.register("sm-array", SMArrayComputeModel)


def build_slice_zero_scenario(
    *,
    memory_variant: str = "hierarchical",
    memory_model: MemoryModel | None = None,
    compute_model: ComputeModel | None = None,
    accelerator_model: AcceleratorModel | None = None,
    system_model: SystemModel | None = None,
) -> WorkbenchScenario:
    """Composition root with optional injection at every major system boundary."""

    if memory_model is not None:
        memory = memory_model
    elif memory_variant == "hierarchical":
        memory = memory_variants.create(
            memory_variant,
            hbm_capacity_gb=24,
            hbm_bandwidth_gbps=600,
            l2_capacity_mb=6,
        )
    elif memory_variant == "flat":
        memory = memory_variants.create(
            memory_variant, capacity_gb=24, bandwidth_gbps=600
        )
    else:
        memory = memory_variants.create(memory_variant)

    compute = compute_model or compute_variants.create("sm-array", sm_count=72, fp16_tflops=120)
    gpu = accelerator_model or ComposableGPU("gpu-0", "Educational GPU", memory, compute)
    system = system_model or SingleAcceleratorSystem(
            Component("client", "Client", ComponentKind.SYSTEM, "Creates an inference request.", Position(70, 260)),
            Component("server", "Inference Server", ComponentKind.SERVICE, "Queues, batches, and dispatches requests.", Position(300, 260)),
            Component("cpu", "CPU Runtime", ComponentKind.PROCESSOR, "Tokenizes input and orchestrates accelerator work.", Position(540, 170)),
            Component("host-memory", "Host DRAM", ComponentKind.MEMORY, "Stores host-side request and runtime data.", Position(540, 390)),
            Component(
                "host-link", "PCIe / Host Link", ComponentKind.INTERCONNECT,
                "Transfers commands and tensors between host and GPU.", Position(790, 260),
                (Metric("Detailed transfer model", "Later slice"),),
            ),
    )

    prefill = ProjectionPhaseModel(
        "prefill", "Prefill", 512,
        "Processes 512 prompt rows together against a shared projection matrix.",
    )
    decode = ProjectionPhaseModel(
        "decode", "Decode", 1,
        "Processes one newly generated token row against the same projection matrix.",
    )

    return WorkbenchScenario(
        "slice-0-problem-02",
        "Problem 02: System-to-GPU Explorer",
        "system",
        (system.diagram(gpu), gpu.diagram(), prefill.diagram(), decode.diagram()),
        {
            "slice": 0,
            "memory_variant": memory_variant,
            "available_memory_variants": list(memory_variants.variants),
            "calculation_status": "fixed topology; executable estimates arrive in Slice 1",
        },
    )
