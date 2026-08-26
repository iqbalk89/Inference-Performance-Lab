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
    input_width: int = 4096
    output_width: int = 4096
    weight_label: str = "W"
    output_label: str = "Y"
    equation: str = "Y = XW"

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
        if self.output_width == 12288 and self.output_label == "[Q K V]":
            return self._qkv_operator_diagram(estimate)
        calculations = projection_calculations(estimate)
        prefix = self.phase_id
        input_id, weight_id = f"{prefix}-input", f"{prefix}-weight"
        matmul_id, output_id = f"{prefix}-matmul", f"{prefix}-output"

        components = (
            TensorBlock(input_id, f"X [{self.rows} × {self.input_width}]", "Input activation tensor. Each row is one token position presented to this projection.", Position(100, 135), (
                Metric("Shape", f"[{self.rows} × {self.input_width}]"),
                Metric("Values", self.rows * self.input_width, "values"),
                Metric("FP16 size", decimal_bytes(estimate.input_bytes), calculation=calculations["input_bytes"]),
            )).component(),
            TensorBlock(weight_id, f"{self.weight_label} [{self.input_width} × {self.output_width}]", "Learned projection weights. The same matrix is applied to every row of X.", Position(100, 395), (
                Metric("Shape", f"[{self.input_width} × {self.output_width}]"),
                Metric("Values", self.input_width * self.output_width, "values"),
                Metric("FP16 size", decimal_bytes(estimate.weight_bytes), calculation=calculations["weight_bytes"]),
            )).component(),
            OperationBlock(matmul_id, "Matrix multiplication", "Computes Y = XW. Push in to see which bytes cross the HBM boundary.", Position(500, 265), (
                Metric("Equation", self.equation),
                Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
            ), self.boundary_graph_id).component(),
            TensorBlock(output_id, f"{self.output_label} [{self.rows} × {self.output_width}]", "Output activation tensor produced by this projection.", Position(900, 265), (
                Metric("Shape", f"[{self.rows} × {self.output_width}]"),
                Metric("Values", self.rows * self.output_width, "values"),
                Metric("FP16 size", decimal_bytes(estimate.output_bytes), calculation=calculations["output_bytes"]),
            )).component(),
        )
        connections = (
            Path.logical(f"{prefix}-x-logical", input_id, matmul_id, "X supplies activation rows"),
            Path.logical(f"{prefix}-w-logical", weight_id, matmul_id, "W supplies projection weights"),
            Path.logical(f"{prefix}-y-logical", matmul_id, output_id, "XW produces Y"),
        )
        return Diagram(f"{prefix}-detail", f"{self.phase_name}: Problem 02 projection", f"{self.explanation} This first level shows mathematics only—not the GPU memory route.", "gpu-0-detail", components, connections, ("FP16 weights and activations use two bytes per value.",))

    def _qkv_operator_diagram(self, estimate: ProjectionEstimate) -> Diagram:
        """Show the fused kernel's three logical output branches explicitly."""
        calculations = projection_calculations(estimate)
        prefix = self.phase_id
        input_id, weight_id, fused_id = f"{prefix}-input", f"{prefix}-weight", f"{prefix}-matmul"
        q_id, k_id, v_id = f"{prefix}-q", f"{prefix}-k", f"{prefix}-v"
        components = (
            TensorBlock(input_id, f"X [{self.rows} × {self.input_width}]", "The same input rows feed all three logical projections.", Position(80, 250), (
                Metric("Shape", f"[{self.rows} × {self.input_width}]"),
                Metric("Values", self.rows * self.input_width, "values"),
                Metric("FP16 size", decimal_bytes(estimate.input_bytes), calculation=calculations["input_bytes"]),
            )).component(),
            TensorBlock(weight_id, f"W_QKV [{self.input_width} × {self.output_width}]", "One fused weight matrix containing the learned Q, K, and V projection weights side by side.", Position(80, 500), (
                Metric("Shape", f"[{self.input_width} × {self.output_width}]"),
                Metric("Values", self.input_width * self.output_width, "values"),
                Metric("FP16 size", decimal_bytes(estimate.weight_bytes), calculation=calculations["weight_bytes"]),
            )).component(),
            OperationBlock(fused_id, "Fused QKV projection", "One matrix multiplication computes three logical results. The output is then split into Q, K, and V views.", Position(455, 350), (
                Metric("Equation", "[Q K V] = XW_QKV"),
                Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
                Metric("Output width", 12288, "features"),
            ), self.boundary_graph_id).component(),
            TensorBlock(q_id, f"Q [{self.rows} × 4096]", "Query vectors. Consumed by the QKᵀ score operation; not stored as a growing KV cache.", Position(850, 120), (
                Metric("Shape", f"[{self.rows} × 4096]"),
                Metric("Role", "forms attention queries"),
            )).component(),
            TensorBlock(k_id, f"K [{self.rows} × 4096]", "Key vectors. Compared with queries and retained in the KV cache for later decode steps.", Position(850, 330), (
                Metric("Shape", f"[{self.rows} × 4096]"),
                Metric("Role", "forms attention keys + cache entries"),
            )).component(),
            TensorBlock(v_id, f"V [{self.rows} × 4096]", "Value vectors. Mixed according to attention scores and retained in the KV cache.", Position(850, 540), (
                Metric("Shape", f"[{self.rows} × 4096]"),
                Metric("Role", "forms attention values + cache entries"),
            )).component(),
        )
        connections = (
            Path.logical(f"{prefix}-x-to-fused", input_id, fused_id, "X is reused by the fused projection"),
            Path.logical(f"{prefix}-w-to-fused", weight_id, fused_id, "W_QKV supplies Q/K/V weights"),
            Path.logical(f"{prefix}-fused-to-q", fused_id, q_id, "Split Q branch"),
            Path.logical(f"{prefix}-fused-to-k", fused_id, k_id, "Split K branch"),
            Path.logical(f"{prefix}-fused-to-v", fused_id, v_id, "Split V branch"),
        )
        return Diagram(
            f"{prefix}-detail", f"{self.phase_name}: QKV projection branches",
            "The fused matrix multiplication is one runtime operation, but its result has three distinct logical consumers: queries, keys, and values.",
            "gpu-0-detail", components, connections, (
                "Q, K, and V are logical views of the fused output; they are not three independent runtime kernels in this fused representation.",
                "The QKᵀ score, softmax, value mixing, and KV-cache operations are the next attention-stage models.",
            ),
        )

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
                Metric("Equation", self.equation),
                Metric("Work", decimal_flops(estimate.flops), calculation=calculations["work"]),
            )).component(),
            OperationBlock(output_id, f"Write {self.output_label}", "The output activation is explicitly written back across the HBM boundary.", Position(920, 485), output_metrics).component(),
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
            Path.transfer(f"{prefix}-y-boundary", output_id, hbm_id, f"Write {self.output_label} to HBM", badge=badge(estimate.output_bytes), metrics=output_metrics),
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
                parameters={"input_width": self.input_width, "output_width": self.output_width, "bytes_per_value": 2, "compute_tflops": 120, "hbm_bandwidth_gbps": 600, "selected_rows": self.rows},
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
                Metric("Equation", self.equation),
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


@dataclass(frozen=True)
class InferencePipelinePhaseModel(PhaseModel):
    """End-to-end phase view: the operator sequence around each matmul."""

    phase_id: str
    phase_name: str
    rows: int
    estimate: ProjectionEstimate
    qkv_graph_id: str = "qkv-detail"

    def diagram(self) -> Diagram:
        prefix = self.phase_id
        ids = {
            name: f"{prefix}-{name}"
            for name in (
                "tokens", "embedding-table", "embedding", "layers", "qkv", "attention", "cache",
                "output-projection", "mlp", "final-norm", "lm-head", "logits",
                "sampling", "next-token",
            )
        }
        calculations = projection_calculations(self.estimate)
        cache_summary = (
            "Reads prior K/V for every decode step and appends the new K/V rows."
            if self.phase_id == "decode" else
            "Receives the prompt K/V rows so later decode steps can reuse them."
        )
        input_components = (
            TensorBlock(ids["tokens"], "Device token IDs", "The GPU phase begins after tokenization and host-to-device transfer. These IDs are now available in device memory for embedding lookup.", Position(30, 220), (
                Metric("Rows", self.rows, "rows", EvidenceKind.ASSUMED),
                Metric("Storage", "device memory"),
                Metric("Ingress", "System-level host link"),
            )).component(),
        )
        tail_components = (
            OperationBlock(ids["sampling"], "Sampling / selection", "Applies the configured decoding policy—greedy, temperature, top-k, or top-p—to choose the next token ID.", Position(2340, 190)).component(),
            TensorBlock(ids["next-token"], "Next token ID", "The selected ID is fed back into the next decode iteration.", Position(2540, 190)).component(),
        ) if self.phase_id == "decode" else (
            AnalysisBlock(f"{prefix}-prefill-boundary", "Prefill complete → Decode begins", "The final prompt-position logits are ready and the KV cache has been populated. Sampling the first generated token occurs at this boundary, outside the Prefill forward pass.", Position(2340, 190), (
                Metric("Boundary", "Final prompt logits → first-token sampling"),
                Metric("TTFT relevance", "Included in end-to-end TTFT", evidence=EvidenceKind.ASSUMED),
            )).component(),
        )
        components = input_components + (
            ResourceBlock(ids["embedding-table"], "Embedding table in HBM", ComponentKind.MEMORY, "The learned vocabulary embedding table normally resides in GPU HBM and is read by a GPU gather kernel.", Position(650, 480), (
                Metric("Location", "GPU HBM"),
                Metric("Access", "GPU gather"),
            )).component(),
            OperationBlock(ids["embedding"], "GPU embedding lookup", "Runs on the GPU as a gather: each token ID selects one row from the embedding table and produces an initial hidden vector.", Position(650, 220)).component(),
            OperationBlock(ids["layers"], "Transformer layer stack", "Repeats the same high-level layer pattern for every layer: norm, QKV, attention, output projection, norm, and MLP.", Position(870, 220), (
                Metric("Rows per phase", self.rows, "rows", EvidenceKind.ASSUMED),
            )).component(),
            OperationBlock(ids["qkv"], "QKV projection", "Creates Q, K, and V. This is one of the detailed operator models available in the workbench.", Position(1080, 90), (
                Metric("Fused work", decimal_flops(self.estimate.flops), calculation=calculations["work"]),
            ), self.qkv_graph_id).component(),
            OperationBlock(ids["attention"], "Attention", "Forms attention scores from Q and K, applies the causal rule, normalizes scores, and mixes V.", Position(1280, 90)).component(),
            OperationBlock(ids["cache"], "KV cache", cache_summary, Position(1280, 360), (
                Metric("Role", "persistent K/V state"),
                Metric("Capacity model", "Next modeling slice", evidence=EvidenceKind.ASSUMED),
            )).component(),
            OperationBlock(ids["output-projection"], "Attention output projection", "Projects the attention result back to the model width, then participates in a residual connection.", Position(1510, 90)).component(),
            OperationBlock(ids["mlp"], "MLP / feed-forward block", "Expands the hidden width, applies the nonlinear transformation, and projects back down.", Position(1510, 300)).component(),
            OperationBlock(ids["final-norm"], "Final normalization", "Normalizes the final hidden state before vocabulary prediction.", Position(1740, 190)).component(),
            OperationBlock(ids["lm-head"], "LM head", "Projects the final hidden vector into one score per vocabulary item.", Position(1940, 190)).component(),
            TensorBlock(ids["logits"], "Vocabulary logits", "One score per possible next token. The highest-scoring entries are candidates, not yet a selected token.", Position(2140, 190), (
                Metric("Output", "vocabulary scores"),
            )).component(),
        ) + tail_components
        input_connections = (
            Path.logical(f"{prefix}-tokens-embedding", ids["tokens"], ids["embedding"], "Device IDs select embedding rows"),
        )
        tail_connections = (
            (
                Path.logical(f"{prefix}-logits-sampling", ids["logits"], ids["sampling"], "Scores enter decoding policy"),
                Path.logical(f"{prefix}-sampling-next-token", ids["sampling"], ids["next-token"], "Selected token ID"),
            ) if self.phase_id == "decode" else (
                Path.mapping(f"{prefix}-logits-prefill-boundary", ids["logits"], f"{prefix}-prefill-boundary", "Final prompt logits cross the phase boundary"),
            )
        )
        connections = input_connections + (
            Path.mapping(f"{prefix}-embedding-table-lookup", ids["embedding-table"], ids["embedding"], "GPU gather reads embedding rows"),
            Path.logical(f"{prefix}-embedding-layers", ids["embedding"], ids["layers"], "Hidden states enter every layer"),
            Path.logical(f"{prefix}-layers-qkv", ids["layers"], ids["qkv"], "Layer hidden state enters QKV"),
            Path.logical(f"{prefix}-qkv-attention", ids["qkv"], ids["attention"], "Q/K/V feed attention"),
            Path.state(f"{prefix}-attention-cache", ids["attention"], ids["cache"], "Read or write K/V state"),
            Path.logical(f"{prefix}-attention-output", ids["attention"], ids["output-projection"], "Attention context enters output projection"),
            Path.logical(f"{prefix}-output-mlp", ids["output-projection"], ids["mlp"], "Residual stream enters feed-forward block"),
            Path.logical(f"{prefix}-mlp-final-norm", ids["mlp"], ids["final-norm"], "Final hidden state"),
            Path.logical(f"{prefix}-norm-lm-head", ids["final-norm"], ids["lm-head"], "Normalized hidden state"),
            Path.logical(f"{prefix}-lm-head-logits", ids["lm-head"], ids["logits"], "Vocabulary scores"),
        ) + tail_connections
        loop_connections = (
            Path.state(f"{prefix}-next-token-loop", ids["next-token"], ids["tokens"], "Decode loop feeds the next token"),
        ) if self.phase_id == "decode" else ()
        return Diagram(
            f"{prefix}-detail", f"{self.phase_name}: end-to-end inference pipeline",
            "The full phase is a sequence of operators. QKV is one operation inside the Transformer layer; it is not a separate inference phase.",
            "gpu-0-detail", components, connections + loop_connections,
            (
                f"{self.phase_name} processes {self.rows} token row{'s' if self.rows != 1 else ''} in this simplified view.",
                "The layer stack is shown once but executes repeatedly for every Transformer layer.",
                "The QKV block links to its detailed fused-projection branch view.",
                "Prefill ends when final prompt-position logits are ready; only Decode samples and feeds a selected token back into another iteration.",
            ),
        )
