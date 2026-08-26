"""Injectable workload and phase diagrams."""

from __future__ import annotations

from dataclasses import dataclass

from .blocks import OperationBlock, Path, ResourceBlock, TensorBlock
from .contracts import ComponentKind, Diagram, EvidenceKind, Metric, PhaseModel, Position
from .estimates import (
    ProjectionEstimate,
    decimal_bytes,
    decimal_flops,
    projection_calculations,
    transfer_calculation,
)


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
        calculations = projection_calculations(estimate)
        prefix = self.phase_id

        input_id = f"{prefix}-input"
        weight_id = f"{prefix}-weight"
        matmul_id = f"{prefix}-matmul"
        output_id = f"{prefix}-output"
        hbm_id = f"{prefix}-hbm"
        l2_id = f"{prefix}-l2"
        compute_id = f"{prefix}-sm-compute"

        input_transfer = transfer_calculation(
            name="Input read", byte_count=estimate.input_bytes,
            bandwidth_gbps=600, source="Input-byte calculation",
        )
        weight_transfer = transfer_calculation(
            name="Weight read", byte_count=estimate.weight_bytes,
            bandwidth_gbps=600, source="Weight-byte calculation",
        )
        output_transfer = transfer_calculation(
            name="Output write", byte_count=estimate.output_bytes,
            bandwidth_gbps=600, source="Output-byte calculation",
        )

        components = (
            TensorBlock(
                input_id, f"X [{self.rows} × 4096]",
                "Logical FP16 input tensor. Its rows represent token positions processed together.",
                Position(300, 75),
                (
                    Metric("Shape", f"[{self.rows} × 4096]"),
                    Metric("Values", self.rows * 4096, "values"),
                    Metric("HBM read", decimal_bytes(estimate.input_bytes), calculation=calculations["input_bytes"]),
                ),
            ).component(),
            TensorBlock(
                weight_id, "W [4096 × 4096]",
                "Logical FP16 learned-weight tensor shared by every input row in this call.",
                Position(300, 275),
                (
                    Metric("Shape", "[4096 × 4096]"),
                    Metric("Values", 4096 * 4096, "values"),
                    Metric("HBM read", decimal_bytes(estimate.weight_bytes), calculation=calculations["weight_bytes"]),
                ),
            ).component(),
            OperationBlock(
                matmul_id, "Matrix multiplication",
                "Logical XW projection. The work maps to the GPU SM and Tensor Core system.",
                Position(650, 175),
                (
                    Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
                    Metric("Total HBM traffic", decimal_bytes(estimate.total_hbm_bytes), calculation=calculations["total_bytes"]),
                    Metric("Arithmetic intensity", round(estimate.arithmetic_intensity, 4), "FLOPs/byte", calculation=calculations["arithmetic_intensity"]),
                    Metric("Hardware ridge point", estimate.hardware.ridge_point_flops_per_byte, "FLOPs/byte", calculation=calculations["ridge_point"]),
                    Metric("Compute bound", round(estimate.compute_time_us, 4), "µs", calculation=calculations["compute_time"]),
                    Metric("Memory bound", round(estimate.memory_time_us, 4), "µs", calculation=calculations["memory_time"]),
                    Metric("Roofline lower bound", round(estimate.lower_bound_us, 4), "µs", calculation=calculations["lower_bound"]),
                    Metric("Predicted bottleneck", estimate.bottleneck, calculation=calculations["bottleneck"]),
                ),
            ).component(),
            TensorBlock(
                output_id, f"Y [{self.rows} × 4096]",
                "Logical FP16 output tensor. Under this model it is written back across the HBM boundary.",
                Position(1000, 175),
                (
                    Metric("Shape", f"[{self.rows} × 4096]"),
                    Metric("Values", self.rows * 4096, "values"),
                    Metric("HBM write", decimal_bytes(estimate.output_bytes), calculation=calculations["output_bytes"]),
                ),
            ).component(),
            ResourceBlock(
                hbm_id, "HBM", ComponentKind.MEMORY,
                "Physical off-chip GPU memory containing X, W, and the stored Y result.",
                Position(30, 505),
                (
                    Metric("Peak bandwidth", 600, "GB/s", EvidenceKind.ASSUMED),
                    Metric("Total projection traffic", decimal_bytes(estimate.total_hbm_bytes), calculation=calculations["total_bytes"]),
                    Metric("Ideal memory bound", round(estimate.memory_time_us, 4), "µs", calculation=calculations["memory_time"]),
                ),
            ).component(),
            ResourceBlock(
                l2_id, "Shared L2 Cache", ComponentKind.MEMORY,
                "Physical GPU-wide cache between HBM and the SM array. Slice 0 does not invent an L2 hit rate.",
                Position(450, 505),
                (
                    Metric("Traffic status", "Requires profiler/calibrated cache model", evidence=EvidenceKind.ASSUMED),
                ),
            ).component(),
            ResourceBlock(
                compute_id, "SM + Tensor Cores", ComponentKind.COMPUTE,
                "Physical execution resources that perform the tiled matrix multiplication.",
                Position(800, 505),
                (
                    Metric("Peak FP16 compute", 120, "TFLOP/s", EvidenceKind.ASSUMED),
                    Metric("Projection work", decimal_flops(estimate.flops), calculation=calculations["work"]),
                    Metric("Ideal compute bound", round(estimate.compute_time_us, 4), "µs", calculation=calculations["compute_time"]),
                ),
            ).component(),
        )

        input_badge = f"{decimal_bytes(estimate.input_bytes)} · 600 GB/s · {estimate.transfer_time_us(estimate.input_bytes):.4f} µs"
        weight_badge = f"{decimal_bytes(estimate.weight_bytes)} · 600 GB/s · {estimate.transfer_time_us(estimate.weight_bytes):.4f} µs"
        output_badge = f"{decimal_bytes(estimate.output_bytes)} · 600 GB/s · {estimate.transfer_time_us(estimate.output_bytes):.4f} µs"

        connections = (
            Path.logical(f"{prefix}-x-logical", input_id, matmul_id, "X participates in XW"),
            Path.logical(f"{prefix}-w-logical", weight_id, matmul_id, "W participates in XW"),
            Path.logical(f"{prefix}-y-logical", matmul_id, output_id, "XW produces Y"),
            Path.transfer(
                f"{prefix}-input-read", hbm_id, input_id, "Read input X from HBM",
                badge=input_badge,
                metrics=(
                    Metric("Bytes read", estimate.input_bytes, "bytes", calculation=calculations["input_bytes"]),
                    Metric("Peak HBM rate", 600, "GB/s", EvidenceKind.ASSUMED),
                    Metric("Transfer bound", round(estimate.transfer_time_us(estimate.input_bytes), 4), "µs", calculation=input_transfer),
                ),
            ),
            Path.transfer(
                f"{prefix}-weight-read", hbm_id, weight_id, "Read weight W from HBM",
                badge=weight_badge,
                metrics=(
                    Metric("Bytes read", estimate.weight_bytes, "bytes", calculation=calculations["weight_bytes"]),
                    Metric("Peak HBM rate", 600, "GB/s", EvidenceKind.ASSUMED),
                    Metric("Transfer bound", round(estimate.transfer_time_us(estimate.weight_bytes), 4), "µs", calculation=weight_transfer),
                ),
            ),
            Path.transfer(
                f"{prefix}-output-write", output_id, hbm_id, "Write output Y back to HBM",
                badge=output_badge,
                metrics=(
                    Metric("Bytes written", estimate.output_bytes, "bytes", calculation=calculations["output_bytes"]),
                    Metric("Peak HBM rate", 600, "GB/s", EvidenceKind.ASSUMED),
                    Metric("Transfer bound", round(estimate.transfer_time_us(estimate.output_bytes), 4), "µs", calculation=output_transfer),
                ),
            ),
            Path.physical(
                f"{prefix}-hbm-l2", hbm_id, l2_id, "HBM-to-L2 physical path",
                badge="600 GB/s HBM boundary",
                metrics=(Metric("Peak HBM boundary rate", 600, "GB/s", EvidenceKind.ASSUMED),),
            ),
            Path.physical(f"{prefix}-l2-sm", l2_id, compute_id, "L2-to-SM on-chip path"),
            Path.mapping(f"{prefix}-matmul-mapping", matmul_id, compute_id, "Matrix multiplication executes across SMs and Tensor Cores"),
        )

        return Diagram(
            f"{prefix}-detail",
            f"{self.phase_name}: Problem 02 projection",
            f"{self.explanation} Solid blue arrows show logical tensor flow; cyan transfer arrows cross the HBM boundary; dashed amber arrows map work to hardware.",
            "gpu-0-detail",
            components,
            connections,
            (
                "FP16 weights and activations use two bytes per value.",
                "The simplified scenario reads W and X once and writes Y once.",
                "L2 traffic and hit rates remain unknown until a cache model or profiler evidence is supplied.",
            ),
        )
