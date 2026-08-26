"""Composable Slice 0 hardware implementations."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    AcceleratorModel,
    Component,
    ComponentKind,
    ComputeModel,
    Connection,
    Diagram,
    EvidenceKind,
    MemoryModel,
    Metric,
    Position,
)


@dataclass(frozen=True)
class HierarchicalMemoryModel(MemoryModel):
    """HBM → L2 → SM-local memory model with injectable specifications."""

    hbm_capacity_gb: float
    hbm_bandwidth_gbps: float
    l2_capacity_mb: float
    local_memory_label: str = "Registers + Shared/L1"

    def components(self) -> tuple[Component, ...]:
        return (
            Component(
                "hbm", "HBM", ComponentKind.MEMORY,
                "GPU-resident weights, activations, and KV-cache storage.",
                Position(90, 390),
                (
                    Metric("Capacity", self.hbm_capacity_gb, "GB"),
                    Metric("Peak bandwidth", self.hbm_bandwidth_gbps, "GB/s"),
                ),
                lane="hardware",
            ),
            Component(
                "memory-controllers", "Memory Controllers", ComponentKind.INTERCONNECT,
                "Channels requests between HBM and the GPU on-chip memory system.",
                Position(340, 390), lane="hardware",
            ),
            Component(
                "l2", "Shared L2 Cache", ComponentKind.MEMORY,
                "GPU-wide cache shared by the SM array.", Position(590, 390),
                (Metric("Capacity", self.l2_capacity_mb, "MB"),), lane="hardware",
            ),
            Component(
                "sm-local-memory", self.local_memory_label, ComponentKind.MEMORY,
                "Per-SM storage closest to active threads and execution pipelines.",
                Position(1090, 390),
                (Metric("Detailed traffic", "Profiler-backed in Slice 4", evidence=EvidenceKind.ASSUMED),),
                lane="hardware",
            ),
        )

    def connections(self) -> tuple[Connection, ...]:
        return (
            Connection("hbm-to-mc", "hbm", "memory-controllers", "HBM channels", "both"),
            Connection("mc-to-l2", "memory-controllers", "l2", "memory fabric", "both"),
            Connection("l2-to-local", "l2", "sm-local-memory", "cache-line traffic", "both"),
        )


@dataclass(frozen=True)
class FlatMemoryModel(MemoryModel):
    """Coarse HBM-only alternative proving the hierarchy is replaceable."""

    capacity_gb: float
    bandwidth_gbps: float

    def components(self) -> tuple[Component, ...]:
        return (
            Component(
                "device-memory", "Flat Device Memory", ComponentKind.MEMORY,
                "A deliberately coarse model that treats device memory as one resource.",
                Position(300, 390),
                (
                    Metric("Capacity", self.capacity_gb, "GB"),
                    Metric("Peak bandwidth", self.bandwidth_gbps, "GB/s"),
                ), lane="hardware",
            ),
        )

    def connections(self) -> tuple[Connection, ...]:
        return ()


@dataclass(frozen=True)
class SMArrayComputeModel(ComputeModel):
    sm_count: int
    fp16_tflops: float

    def components(self) -> tuple[Component, ...]:
        return (
            Component(
                "sm-array", "SM Array", ComponentKind.COMPUTE,
                "Schedules thread blocks across streaming multiprocessors.",
                Position(840, 390),
                (
                    Metric("SM count", self.sm_count),
                    Metric("Peak FP16 compute", self.fp16_tflops, "TFLOP/s"),
                ), lane="hardware",
            ),
        )


@dataclass(frozen=True)
class ComposableGPU(AcceleratorModel):
    """GPU assembled from injected memory and compute implementations."""

    gpu_id: str
    display_name: str
    memory: MemoryModel
    compute: ComputeModel

    def system_component(self) -> Component:
        return Component(
            self.gpu_id,
            self.display_name,
            ComponentKind.ACCELERATOR,
            "Click to inspect inference phases mapped onto GPU hardware.",
            Position(1010, 260),
            (Metric("Active scenario", "Problem 02", evidence=EvidenceKind.ASSUMED),),
            drilldown_graph_id=f"{self.gpu_id}-detail",
        )

    def diagram(self) -> Diagram:
        memory_components = self.memory.components()
        compute_components = self.compute.components()
        phase_components = (
            Component(
                "prefill", "Prefill", ComponentKind.PHASE,
                "Processes all prompt-token rows and creates their KV-cache entries.",
                Position(300, 90),
                (Metric("Problem 02 rows", 512, "rows", EvidenceKind.ASSUMED),),
                drilldown_graph_id="prefill-detail", lane="process",
            ),
            Component(
                "decode", "Decode", ComponentKind.PHASE,
                "Processes one new row per active sequence and advances generation.",
                Position(750, 90),
                (Metric("Problem 02 rows", 1, "row", EvidenceKind.ASSUMED),),
                drilldown_graph_id="decode-detail", lane="process",
            ),
        )
        hardware_ids = [component.component_id for component in memory_components + compute_components]
        mappings = tuple(
            Connection(
                f"{phase.component_id}-uses-{resource_id}",
                phase.component_id,
                resource_id,
                "resource demand",
                category="mapping",
            )
            for phase in phase_components
            for resource_id in hardware_ids
        )
        return Diagram(
            f"{self.gpu_id}-detail",
            self.display_name,
            "Logical inference phases above; injected physical resources below.",
            "system",
            phase_components + memory_components + compute_components,
            self.memory.connections() + mappings,
            (
                "Slice 0 shows topology and fixed Problem 02 facts; calculations arrive in Slice 1.",
                "Dashed mapping edges mean a process uses a resource; they are not physical buses.",
            ),
        )
