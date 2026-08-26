"""Composition root: dependencies are selected and assembled only here."""

from __future__ import annotations

from .contracts import AcceleratorModel, CalculationDetail, CalculationInput, CalculationStep, Component, ComponentKind, ComputeModel, EvidenceKind, MemoryModel, Metric, PhaseModel, Position, SystemModel, VariantRegistry, WorkbenchScenario
from .estimates import decimal_bytes, decimal_flops, problem_02_estimate, projection_calculations, transfer_calculation
from .hardware import ComposableGPU, FlatMemoryModel, HierarchicalMemoryModel, SMArrayComputeModel
from .estimates import problem_02_estimate
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
    prefill_model: PhaseModel | None = None,
    decode_model: PhaseModel | None = None,
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
    prefill_estimate = problem_02_estimate(512)
    decode_estimate = problem_02_estimate(1)
    prefill_calculations = projection_calculations(prefill_estimate)
    decode_calculations = projection_calculations(decode_estimate)
    phase_metrics = {
        "prefill": (
            Metric("Lower bound", round(prefill_estimate.lower_bound_us, 4), "µs", calculation=prefill_calculations["lower_bound"]),
            Metric("Bottleneck", prefill_estimate.bottleneck, calculation=prefill_calculations["bottleneck"]),
            Metric("Arithmetic intensity", prefill_estimate.arithmetic_intensity, "FLOPs/byte", calculation=prefill_calculations["arithmetic_intensity"]),
            Metric("Work", decimal_flops(prefill_estimate.flops), calculation=prefill_calculations["work"]),
            Metric("HBM traffic", decimal_bytes(prefill_estimate.total_hbm_bytes), calculation=prefill_calculations["total_bytes"]),
            Metric("Rows", 512, "rows", EvidenceKind.ASSUMED),
        ),
        "decode": (
            Metric("Lower bound", round(decode_estimate.lower_bound_us, 4), "µs", calculation=decode_calculations["lower_bound"]),
            Metric("Bottleneck", decode_estimate.bottleneck, calculation=decode_calculations["bottleneck"]),
            Metric("Arithmetic intensity", round(decode_estimate.arithmetic_intensity, 4), "FLOPs/byte", calculation=decode_calculations["arithmetic_intensity"]),
            Metric("Work", decimal_flops(decode_estimate.flops), calculation=decode_calculations["work"]),
            Metric("HBM traffic", decimal_bytes(decode_estimate.total_hbm_bytes), calculation=decode_calculations["total_bytes"]),
            Metric("Rows", 1, "row", EvidenceKind.ASSUMED),
        ),
    }
    gpu = accelerator_model or ComposableGPU("gpu-0", "Educational GPU", memory, compute, phase_metrics)
    system = system_model or SingleAcceleratorSystem(
            Component("client", "Client", ComponentKind.SYSTEM, "Creates an inference request.", Position(30, 260)),
            Component("server", "Inference Server", ComponentKind.SERVICE, "Queues, batches, and dispatches requests.", Position(280, 260)),
            Component("cpu", "CPU Runtime", ComponentKind.PROCESSOR, "Tokenizes input and orchestrates accelerator work.", Position(530, 130)),
            Component("host-memory", "Host DRAM", ComponentKind.MEMORY, "Stores host-side request and runtime data.", Position(530, 430)),
            Component(
                "host-link", "PCIe / Host Link", ComponentKind.INTERCONNECT,
                "Transfers token IDs and commands between host and GPU.", Position(780, 260),
                (
                    Metric("Assumed peak rate", 32, "GB/s", EvidenceKind.ASSUMED),
                    Metric(
                        "Prefill token-ID transfer", 2048, "bytes", EvidenceKind.THEORETICAL,
                        derivation="512 IDs × 4 bytes/ID",
                        calculation=CalculationDetail(
                            "Prompt token-ID payload",
                            "Before GPU prefill begins, the host provides the numerical token IDs representing the prompt. This educational scenario assumes one 32-bit integer for each of 512 prompt positions.",
                            "payload bytes = prompt tokens × bytes per token ID",
                            (
                                CalculationInput("T", "512 token IDs", "Number of prompt-token positions", "Problem 02 prefill row count"),
                                CalculationInput("s", "4 bytes/ID", "Storage for one assumed 32-bit token ID", "Educational host-transfer assumption"),
                            ),
                            (
                                CalculationStep("Count IDs", "512 prompt positions = 512 token IDs", "Each prompt position is represented by one integer ID."),
                                CalculationStep("Convert to bytes", "512 IDs × 4 bytes/ID = 2,048 bytes", "The ID units cancel."),
                                CalculationStep("Readable scale", "2,048 bytes = 2.048 decimal KB", "This is far smaller than the projection's weight traffic."),
                            ),
                            "IDs × bytes/ID = bytes.",
                            "The modeled prompt-ID payload is 2,048 bytes before protocol and runtime overhead.",
                            ("Assumes 32-bit token IDs and excludes request metadata, embeddings, and control traffic.",),
                        ),
                    ),
                    Metric("Ideal prefill transfer", 0.064, "µs", EvidenceKind.THEORETICAL, derivation="2,048 bytes / 32 GB/s", calculation=transfer_calculation(name="Host-to-GPU token ID", byte_count=2048, bandwidth_gbps=32, source="Prompt token-ID payload calculation")),
                ),
            ),
    )

    prefill = prefill_model or ProjectionPhaseModel(
        "prefill", "Prefill", 512,
        "Processes 512 prompt rows together against a shared projection matrix.",
        prefill_estimate,
    )
    decode = decode_model or ProjectionPhaseModel(
        "decode", "Decode", 1,
        "Processes one newly generated token row against the same projection matrix.",
        decode_estimate,
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
            "calculation_status": "executable Problem 02 projection estimates",
        },
    )
