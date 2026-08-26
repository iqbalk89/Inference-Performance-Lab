"""Injectable workload and phase diagrams for Slice 0."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Component, ComponentKind, Connection, Diagram, EvidenceKind, Metric, PhaseModel, Position


@dataclass(frozen=True)
class ProjectionPhaseModel(PhaseModel):
    phase_id: str
    phase_name: str
    rows: int
    explanation: str

    def diagram(self) -> Diagram:
        return Diagram(
            f"{self.phase_id}-detail",
            f"{self.phase_name}: Problem 02 projection",
            self.explanation,
            "gpu-0-detail",
            (
                Component(
                    f"{self.phase_id}-input", f"X [{self.rows} × 4096]",
                    ComponentKind.OPERATION, "FP16 input activation rows.",
                    Position(130, 235),
                    (Metric("Rows", self.rows, "rows", EvidenceKind.ASSUMED),),
                    lane="process",
                ),
                Component(
                    f"{self.phase_id}-weight", "W [4096 × 4096]",
                    ComponentKind.MEMORY, "Shared FP16 projection weights.",
                    Position(130, 440),
                    (Metric("Element size", 2, "bytes", EvidenceKind.ASSUMED),),
                    lane="hardware",
                ),
                Component(
                    f"{self.phase_id}-matmul", "Matrix multiplication",
                    ComponentKind.OPERATION, "XW using GPU compute and memory resources.",
                    Position(560, 300),
                    (
                        Metric("Formula", "2MKN FLOPs"),
                        Metric("Calculation status", "Slice 1"),
                    ),
                    lane="process",
                ),
                Component(
                    f"{self.phase_id}-output", f"Y [{self.rows} × 4096]",
                    ComponentKind.OPERATION, "FP16 output activation rows.",
                    Position(1010, 300), lane="process",
                ),
            ),
            (
                Connection(f"{self.phase_id}-x-flow", f"{self.phase_id}-input", f"{self.phase_id}-matmul", "input rows"),
                Connection(f"{self.phase_id}-w-flow", f"{self.phase_id}-weight", f"{self.phase_id}-matmul", "weight read"),
                Connection(f"{self.phase_id}-y-flow", f"{self.phase_id}-matmul", f"{self.phase_id}-output", "output rows"),
            ),
            (
                "FP16 weights and activations use two bytes per value.",
                "The simplified scenario reads W and X once and writes Y once.",
            ),
        )
