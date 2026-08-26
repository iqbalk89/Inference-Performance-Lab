"""Composable Slice 0 hardware implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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
                Position(50, 410),
                (
                    Metric("Capacity", self.hbm_capacity_gb, "GB"),
                    Metric("Peak bandwidth", self.hbm_bandwidth_gbps, "GB/s"),
                ),
                lane="hardware",
            ),
            Component(
                "memory-controllers", "Memory Controllers", ComponentKind.INTERCONNECT,
                "Channels requests between HBM and the GPU on-chip memory system.",
                Position(300, 410), lane="hardware",
            ),
            Component(
                "l2", "Shared L2 Cache", ComponentKind.MEMORY,
                "GPU-wide cache shared by the SM array.", Position(550, 410),
                (Metric("Capacity", self.l2_capacity_mb, "MB"),), lane="hardware",
            ),
            Component(
                "sm-local-memory", self.local_memory_label, ComponentKind.MEMORY,
                "Per-SM storage closest to active threads and execution pipelines.",
                Position(1050, 410),
                (Metric("Detailed traffic", "Profiler-backed in Slice 4", evidence=EvidenceKind.ASSUMED),),
                lane="hardware",
            ),
        )

    def connections(self) -> tuple[Connection, ...]:
        return (
            Connection(
                "hbm-to-mc", "hbm", "memory-controllers", "HBM channels", "both",
                metrics=(Metric("Peak path rate", self.hbm_bandwidth_gbps, "GB/s"),),
                badge=f"{self.hbm_bandwidth_gbps:g} GB/s",
            ),
            Connection("mc-to-l2", "memory-controllers", "l2", "memory fabric", "both"),
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
                Position(300, 410),
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
                Position(800, 410),
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
    phase_metrics: Mapping[str, tuple[Metric, ...]] = field(default_factory=dict)

    def system_component(self) -> Component:
        prefill_metrics = self.phase_metrics.get("prefill", ())
        decode_metrics = self.phase_metrics.get("decode", ())
        summary_metrics = (Metric("Active scenario", "Problem 02", evidence=EvidenceKind.ASSUMED),)
        if prefill_metrics and decode_metrics:
            summary_metrics = (
                Metric("Prefill lower bound", prefill_metrics[0].value, prefill_metrics[0].unit, calculation=prefill_metrics[0].calculation),
                Metric("Decode lower bound", decode_metrics[0].value, decode_metrics[0].unit, calculation=decode_metrics[0].calculation),
                Metric("Hardware", "120 TFLOP/s · 600 GB/s"),
            )
        return Component(
            self.gpu_id,
            self.display_name,
            ComponentKind.ACCELERATOR,
            "Click to inspect inference phases mapped onto GPU hardware.",
            Position(1010, 260),
            summary_metrics,
            drilldown_graph_id=f"{self.gpu_id}-detail",
        )

    def diagram(self) -> Diagram:
        memory_components = self.memory.components()
        compute_components = self.compute.components()
        phase_components = (
            Component(
                "inference-overview", "Full inference pipeline", ComponentKind.OPERATION,
                "Combined Prefill/Decode swim-lane view of QKV, attention, KV-cache, output projection, and feed-forward stages.",
                Position(525, 170), drilldown_graph_id="inference-overview", lane="process",
            ),
            Component(
                "prefill", "Prefill", ComponentKind.PHASE,
                "Processes all prompt-token rows and creates their KV-cache entries.",
                Position(300, 80),
                self.phase_metrics.get("prefill", (Metric("Problem 02 rows", 512, "rows", EvidenceKind.ASSUMED),)),
                drilldown_graph_id="prefill-detail", lane="process",
            ),
            Component(
                "decode", "Decode", ComponentKind.PHASE,
                "Processes one new row per active sequence and advances generation.",
                Position(750, 80),
                self.phase_metrics.get("decode", (Metric("Problem 02 rows", 1, "row", EvidenceKind.ASSUMED),)),
                drilldown_graph_id="decode-detail", lane="process",
            ),
        )
        memory_target = "hbm" if any(item.component_id == "hbm" for item in memory_components) else memory_components[0].component_id
        compute_target = compute_components[0].component_id
        mappings = tuple(
            Connection(
                f"{phase.component_id}-uses-{resource_id}",
                phase.component_id,
                resource_id,
                "resource demand",
                category="mapping",
            )
            for phase in phase_components
            for resource_id in (memory_target, compute_target)
        )
        physical_connections: tuple[Connection, ...] = ()
        if any(item.component_id == "l2" for item in memory_components):
            physical_connections = (
                Connection("l2-to-sm", "l2", compute_target, "operand traffic", "both"),
                Connection("sm-to-local", compute_target, "sm-local-memory", "register/shared/L1 traffic", "both"),
            )
        elif memory_components:
            physical_connections = (
                Connection("memory-to-sm", memory_target, compute_target, "operand traffic", "both"),
            )
        return Diagram(
            f"{self.gpu_id}-detail",
            self.display_name,
            "Logical inference phases above; injected physical resources below.",
            "system",
            phase_components + memory_components + compute_components,
            self.memory.connections() + physical_connections + mappings,
            (
                "Problem 02 values are executable idealized estimates, not measured performance.",
                "Dashed mapping edges mean a process uses a resource; they are not physical buses.",
            ),
        )
