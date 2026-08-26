"""Injectable workload and phase diagrams for Slice 0."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Component, ComponentKind, Connection, Diagram, EvidenceKind, Metric, PhaseModel, Position
from .estimates import ProjectionEstimate, decimal_bytes, decimal_flops


@dataclass(frozen=True)
class ProjectionPhaseModel(PhaseModel):
    phase_id: str
    phase_name: str
    rows: int
    explanation: str
    estimate: ProjectionEstimate | None = None

    def diagram(self) -> Diagram:
        if self.estimate is None:
            raise ValueError("ProjectionPhaseModel requires an estimate before visualization")
        estimate = self.estimate
        return Diagram(
            f"{self.phase_id}-detail",
            f"{self.phase_name}: Problem 02 projection",
            self.explanation,
            "gpu-0-detail",
            (
                Component(
                    f"{self.phase_id}-input", f"X [{self.rows} × 4096]",
                    ComponentKind.OPERATION, "FP16 input activation rows.",
                    Position(90, 210),
                    (
                        Metric("Shape", f"[{self.rows} × 4096]"),
                        Metric("HBM read", decimal_bytes(estimate.input_bytes), derivation=f"{self.rows} × 4096 × 2 bytes"),
                        Metric("Transfer bound", round(estimate.transfer_time_us(estimate.input_bytes), 4), "µs", derivation=f"{estimate.input_bytes:,} bytes / 600 GB/s"),
                    ),
                    lane="process",
                ),
                Component(
                    f"{self.phase_id}-weight", "W [4096 × 4096]",
                    ComponentKind.MEMORY, "Shared FP16 projection weights.",
                    Position(90, 455),
                    (
                        Metric("HBM read", decimal_bytes(estimate.weight_bytes), derivation="4096 × 4096 × 2 bytes"),
                        Metric("Element size", 2, "bytes", EvidenceKind.ASSUMED),
                        Metric("Transfer bound", round(estimate.transfer_time_us(estimate.weight_bytes), 4), "µs", derivation=f"{estimate.weight_bytes:,} bytes / 600 GB/s"),
                    ),
                    lane="hardware",
                ),
                Component(
                    f"{self.phase_id}-matmul", "Matrix multiplication",
                    ComponentKind.OPERATION, "XW using GPU compute and memory resources.",
                    Position(520, 300),
                    (
                        Metric("Work", decimal_flops(estimate.flops), derivation=f"2 × {self.rows} × 4096 × 4096"),
                        Metric("Total HBM traffic", decimal_bytes(estimate.total_hbm_bytes), derivation="weight read + input read + output write"),
                        Metric("Arithmetic intensity", round(estimate.arithmetic_intensity, 4), "FLOPs/byte", derivation=f"{estimate.flops:,} FLOPs / {estimate.total_hbm_bytes:,} bytes"),
                        Metric("Compute bound", round(estimate.compute_time_us, 4), "µs", derivation=f"{estimate.flops:,} FLOPs / 120 TFLOP/s"),
                        Metric("Memory bound", round(estimate.memory_time_us, 4), "µs", derivation=f"{estimate.total_hbm_bytes:,} bytes / 600 GB/s"),
                        Metric("Roofline lower bound", round(estimate.lower_bound_us, 4), "µs", derivation="max(compute bound, memory bound)"),
                        Metric("Predicted bottleneck", estimate.bottleneck),
                    ),
                    lane="process",
                ),
                Component(
                    f"{self.phase_id}-output", f"Y [{self.rows} × 4096]",
                    ComponentKind.OPERATION, "FP16 output activation rows.",
                    Position(970, 300),
                    (
                        Metric("Shape", f"[{self.rows} × 4096]"),
                        Metric("HBM write", decimal_bytes(estimate.output_bytes), derivation=f"{self.rows} × 4096 × 2 bytes"),
                        Metric("Transfer bound", round(estimate.transfer_time_us(estimate.output_bytes), 4), "µs", derivation=f"{estimate.output_bytes:,} bytes / 600 GB/s"),
                    ), lane="process",
                ),
            ),
            (
                Connection(
                    f"{self.phase_id}-x-flow", f"{self.phase_id}-input", f"{self.phase_id}-matmul",
                    f"{decimal_bytes(estimate.input_bytes)} @ 600 GB/s → {estimate.transfer_time_us(estimate.input_bytes):.4f} µs",
                    metrics=(
                        Metric("Bytes", estimate.input_bytes, "bytes"),
                        Metric("Peak path rate", 600, "GB/s"),
                        Metric("Transfer bound", round(estimate.transfer_time_us(estimate.input_bytes), 4), "µs"),
                    ),
                ),
                Connection(
                    f"{self.phase_id}-w-flow", f"{self.phase_id}-weight", f"{self.phase_id}-matmul",
                    f"{decimal_bytes(estimate.weight_bytes)} @ 600 GB/s → {estimate.transfer_time_us(estimate.weight_bytes):.4f} µs",
                    metrics=(
                        Metric("Bytes", estimate.weight_bytes, "bytes"),
                        Metric("Peak path rate", 600, "GB/s"),
                        Metric("Transfer bound", round(estimate.transfer_time_us(estimate.weight_bytes), 4), "µs"),
                    ),
                ),
                Connection(
                    f"{self.phase_id}-y-flow", f"{self.phase_id}-matmul", f"{self.phase_id}-output",
                    f"{decimal_bytes(estimate.output_bytes)} @ 600 GB/s → {estimate.transfer_time_us(estimate.output_bytes):.4f} µs",
                    metrics=(
                        Metric("Bytes", estimate.output_bytes, "bytes"),
                        Metric("Peak path rate", 600, "GB/s"),
                        Metric("Transfer bound", round(estimate.transfer_time_us(estimate.output_bytes), 4), "µs"),
                    ),
                ),
            ),
            (
                "FP16 weights and activations use two bytes per value.",
                "The simplified scenario reads W and X once and writes Y once.",
            ),
        )
