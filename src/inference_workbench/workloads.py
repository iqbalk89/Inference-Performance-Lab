"""Injectable workload models with progressive educational drill-downs."""

from __future__ import annotations

from dataclasses import dataclass

from .blocks import AnalysisBlock, OperationBlock, Path, ResourceBlock, TensorBlock
from .contracts import ChartData, ComponentKind, Diagram, EvidenceKind, Metric, PhaseModel, Position
from .estimates import ProjectionEstimate, decimal_bytes, decimal_flops, projection_calculations, projection_roofline_points, transfer_calculation


@dataclass(frozen=True)
class ProjectionPhaseModel(PhaseModel):
    """One XW projection shown at three progressively deeper abstractions."""

    phase_id: str
    phase_name: str
    rows: int
    explanation: str
    estimate: ProjectionEstimate | None = None

    @property
    def boundary_graph_id(self) -> str:
        return f"{self.phase_id}-hbm-boundary"

    @property
    def execution_graph_id(self) -> str:
        return f"{self.phase_id}-execution-path"

    def _required_estimate(self) -> ProjectionEstimate:
        if self.estimate is None:
            raise ValueError("ProjectionPhaseModel requires an estimate before visualization")
        return self.estimate

    def diagram(self) -> Diagram:
        """Level 1: the mathematical operator and tensor shapes only."""
        estimate = self._required_estimate()
        calculations = projection_calculations(estimate)
        prefix = self.phase_id
        input_id, weight_id = f"{prefix}-input", f"{prefix}-weight"
        matmul_id, output_id = f"{prefix}-matmul", f"{prefix}-output"

        components = (
            TensorBlock(input_id, f"X [{self.rows} × 4096]", "Input activation tensor. Each row is one token position presented to this projection.", Position(100, 135), (
                Metric("Shape", f"[{self.rows} × 4096]"),
                Metric("Values", self.rows * 4096, "values"),
                Metric("FP16 size", decimal_bytes(estimate.input_bytes), calculation=calculations["input_bytes"]),
            )).component(),
            TensorBlock(weight_id, "W [4096 × 4096]", "Learned projection weights. The same matrix is applied to every row of X.", Position(100, 395), (
                Metric("Shape", "[4096 × 4096]"),
                Metric("Values", 4096 * 4096, "values"),
                Metric("FP16 size", decimal_bytes(estimate.weight_bytes), calculation=calculations["weight_bytes"]),
            )).component(),
            OperationBlock(matmul_id, "Matrix multiplication", "Computes Y = XW. Push in to see which bytes cross the HBM boundary.", Position(500, 265), (
                Metric("Equation", "Y = XW"),
                Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
            ), self.boundary_graph_id).component(),
            TensorBlock(output_id, f"Y [{self.rows} × 4096]", "Output activation tensor produced by this projection.", Position(900, 265), (
                Metric("Shape", f"[{self.rows} × 4096]"),
                Metric("Values", self.rows * 4096, "values"),
                Metric("FP16 size", decimal_bytes(estimate.output_bytes), calculation=calculations["output_bytes"]),
            )).component(),
        )
        connections = (
            Path.logical(f"{prefix}-x-logical", input_id, matmul_id, "X supplies activation rows"),
            Path.logical(f"{prefix}-w-logical", weight_id, matmul_id, "W supplies projection weights"),
            Path.logical(f"{prefix}-y-logical", matmul_id, output_id, "XW produces Y"),
        )
        return Diagram(f"{prefix}-detail", f"{self.phase_name}: Problem 02 projection", f"{self.explanation} This first level shows mathematics only—not the GPU memory route.", "gpu-0-detail", components, connections, ("FP16 weights and activations use two bytes per value.",))

    def _transfer_metrics(self, name: str, byte_count: int, calculation_key: str) -> tuple[Metric, ...]:
        estimate = self._required_estimate()
        calculations = projection_calculations(estimate)
        transfer = transfer_calculation(name=name, byte_count=byte_count, bandwidth_gbps=600, source=f"{name} byte calculation")
        return (
            Metric("Traffic", decimal_bytes(byte_count), calculation=calculations[calculation_key]),
            Metric("Assumed HBM rate", 600, "GB/s", EvidenceKind.ASSUMED),
            Metric("Ideal transfer bound", round(estimate.transfer_time_us(byte_count), 4), "µs", calculation=transfer),
        )

    def boundary_diagram(self) -> Diagram:
        """Level 2: account only for traffic crossing the HBM boundary."""
        estimate = self._required_estimate()
        calculations = projection_calculations(estimate)
        prefix = self.phase_id
        hbm_id = f"{prefix}-boundary-hbm"
        input_id, weight_id = f"{prefix}-input-read", f"{prefix}-weight-read"
        output_id, matmul_id = f"{prefix}-output-write", f"{prefix}-boundary-matmul"
        accounting_id = f"{prefix}-roofline-accounting"
        input_metrics = self._transfer_metrics("Input read", estimate.input_bytes, "input_bytes")
        weight_metrics = self._transfer_metrics("Weight read", estimate.weight_bytes, "weight_bytes")
        output_metrics = self._transfer_metrics("Output write", estimate.output_bytes, "output_bytes")

        components = (
            ResourceBlock(hbm_id, "HBM boundary", ComponentKind.MEMORY, "Traffic is counted when data is read from or written to off-chip GPU memory.", Position(70, 275), (
                Metric("Assumed peak bandwidth", 600, "GB/s", EvidenceKind.ASSUMED),
                Metric("Total counted traffic", decimal_bytes(estimate.total_hbm_bytes), calculation=calculations["total_bytes"]),
            )).component(),
            OperationBlock(input_id, "Read X", "One modeled read of the input activation tensor from HBM.", Position(390, 65), input_metrics).component(),
            OperationBlock(weight_id, "Read W", "One modeled read of the learned projection matrix from HBM.", Position(390, 275), weight_metrics).component(),
            OperationBlock(matmul_id, "Matrix multiplication", "The operator remains visible here: read tiles of X and W are multiplied and accumulated to produce Y.", Position(650, 275), (
                Metric("Equation", "Y = XW"),
                Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
            )).component(),
            OperationBlock(output_id, "Write Y", "The output activation is explicitly written back across the HBM boundary.", Position(920, 485), output_metrics).component(),
            AnalysisBlock(accounting_id, "Roofline performance model", "Analyzes the inference operator using FLOPs, HBM traffic, arithmetic intensity, and hardware limits. This block is not part of runtime execution.", Position(960, 65), (
                Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
                Metric("HBM traffic", decimal_bytes(estimate.total_hbm_bytes), calculation=calculations["total_bytes"]),
                Metric("Arithmetic intensity", round(estimate.arithmetic_intensity, 4), "FLOPs/byte", calculation=calculations["arithmetic_intensity"]),
                Metric("Roofline lower bound", round(estimate.lower_bound_us, 4), "µs", calculation=calculations["lower_bound"]),
                Metric("Predicted bottleneck", estimate.bottleneck, calculation=calculations["bottleneck"]),
            ), self.execution_graph_id).component(),
        )

        def badge(byte_count: int) -> str:
            return f"{decimal_bytes(byte_count)} · 600 GB/s · {estimate.transfer_time_us(byte_count):.4f} µs"

        connections = (
            Path.transfer(f"{prefix}-x-boundary", hbm_id, input_id, "Read X from HBM", badge=badge(estimate.input_bytes), metrics=input_metrics),
            Path.transfer(f"{prefix}-w-boundary", hbm_id, weight_id, "Read W from HBM", badge=badge(estimate.weight_bytes), metrics=weight_metrics),
            Path.transfer(f"{prefix}-y-boundary", output_id, hbm_id, "Write Y to HBM", badge=badge(estimate.output_bytes), metrics=output_metrics),
            Path.logical(f"{prefix}-x-to-matmul", input_id, matmul_id, "Read X tiles into the operator"),
            Path.logical(f"{prefix}-w-to-matmul", weight_id, matmul_id, "Read W tiles into the operator"),
            Path.logical(f"{prefix}-matmul-to-y", matmul_id, output_id, "Matrix multiplication produces Y"),
            Path.mapping(f"{prefix}-matmul-accounted", matmul_id, accounting_id, "Operator feeds the performance model"),
        )
        return Diagram(self.boundary_graph_id, f"{self.phase_name}: projection execution and HBM accounting", "The matrix multiplication stays visible while its HBM reads, output write, and roofline consequences are added around it.", f"{prefix}-detail", components, connections, (
            "The simplified model reads X once, reads W once, and writes Y once.",
            "The 600 GB/s HBM rate is an assumed educational hardware parameter.",
            "Cache effects, rereads, write allocation, alignment, and workspace are excluded.",
        ), (
            ChartData(
                chart_id=f"{prefix}-roofline-study",
                kind="roofline-sensitivity",
                title="Projection sensitivity: token rows vs. ideal latency",
                description="Change M, HBM bandwidth, or peak compute to see how the ideal memory and compute bounds move. This is a model, not measured kernel latency.",
                x_label="Token rows processed together (M)",
                y_label="Ideal lower-bound time (µs)",
                points=projection_roofline_points((1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)),
                parameters={"input_width": 4096, "output_width": 4096, "bytes_per_value": 2, "compute_tflops": 120, "hbm_bandwidth_gbps": 600, "selected_rows": self.rows},
            ),
        ))

    def execution_diagram(self) -> Diagram:
        """Level 3: show physical hierarchy without inventing cache traffic."""
        estimate = self._required_estimate()
        calculations = projection_calculations(estimate)
        prefix = self.phase_id
        hbm_id, l2_id = f"{prefix}-physical-hbm", f"{prefix}-physical-l2"
        local_id, compute_id = f"{prefix}-sm-local", f"{prefix}-tensor-cores"
        matmul_id = f"{prefix}-physical-matmul"
        unknown = "Unknown—measure or calibrate"
        components = (
            OperationBlock(matmul_id, "Matrix multiplication", "The same XW operator is now mapped onto the physical memory and compute path below.", Position(540, 70), (
                Metric("Equation", "Y = XW"),
                Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
            )).component(),
            ResourceBlock(hbm_id, "HBM", ComponentKind.MEMORY, "Off-chip GPU memory. Boundary accounting counts X and W reads plus the Y write here.", Position(60, 280), (
                Metric("Assumed peak bandwidth", 600, "GB/s", EvidenceKind.ASSUMED),
                Metric("Boundary traffic", decimal_bytes(estimate.total_hbm_bytes), calculation=calculations["total_bytes"]),
            )).component(),
            ResourceBlock(l2_id, "Shared L2 cache", ComponentKind.MEMORY, "GPU-wide cache between HBM and the SMs. It may reduce HBM rereads, but this model has no measured hit rate.", Position(360, 280), (
                Metric("Hit rate", unknown, evidence=EvidenceKind.ASSUMED),
                Metric("Internal traffic", unknown, evidence=EvidenceKind.ASSUMED),
            )).component(),
            ResourceBlock(local_id, "SM-local storage", ComponentKind.MEMORY, "Software-managed shared memory and registers hold tiles and partial results close to execution.", Position(660, 280), (
                Metric("Shared-memory traffic", unknown, evidence=EvidenceKind.ASSUMED),
                Metric("Register reuse", unknown, evidence=EvidenceKind.ASSUMED),
            )).component(),
            ResourceBlock(compute_id, "Tensor Core pipelines", ComponentKind.COMPUTE, "Physical matrix pipelines execute tiled multiply-accumulate work across the SM array.", Position(960, 280), (
                Metric("Projection work", decimal_flops(estimate.flops), calculation=calculations["work"]),
                Metric("Assumed peak FP16", 120, "TFLOP/s", EvidenceKind.ASSUMED),
                Metric("Achieved utilization", unknown, evidence=EvidenceKind.ASSUMED),
            )).component(),
        )
        connections = (
            Path.mapping(f"{prefix}-physical-matmul-map", matmul_id, compute_id, "Matrix multiplication executes on Tensor Core pipelines"),
            Path.physical(f"{prefix}-physical-hbm-l2", hbm_id, l2_id, "Physical HBM and L2 path", badge="HBM boundary: assumed 600 GB/s"),
            Path.physical(f"{prefix}-physical-l2-local", l2_id, local_id, "On-chip L2 and SM path", badge="Rate requires hardware profile"),
            Path.physical(f"{prefix}-physical-local-compute", local_id, compute_id, "Tile delivery and result path", badge="Traffic requires profiling"),
        )
        return Diagram(self.execution_graph_id, f"{self.phase_name}: physical GPU execution path", "This level shows where data can travel. It separates known HBM-boundary accounting from cache and on-chip quantities that remain unknown.", self.boundary_graph_id, components, connections, (
            "Arrows show possible bidirectional physical movement, not a measured transaction trace.",
            "L2, shared-memory, register, and achieved-compute quantities require profiling or a calibrated model.",
        ))

    def diagrams(self) -> tuple[Diagram, ...]:
        return (self.diagram(), self.boundary_diagram(), self.execution_diagram())
