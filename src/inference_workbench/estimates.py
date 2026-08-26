"""Auditable analytical estimates used by the first executable workbench slice."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import CalculationDetail, CalculationInput, CalculationStep


@dataclass(frozen=True)
class ProjectionInputs:
    rows: int
    input_width: int
    output_width: int
    bytes_per_weight: int
    bytes_per_activation: int


@dataclass(frozen=True)
class HardwareRates:
    compute_tflops: float
    hbm_bandwidth_gbps: float

    @property
    def compute_flops_per_second(self) -> float:
        return self.compute_tflops * 1e12

    @property
    def hbm_bytes_per_second(self) -> float:
        return self.hbm_bandwidth_gbps * 1e9

    @property
    def ridge_point_flops_per_byte(self) -> float:
        return self.compute_flops_per_second / self.hbm_bytes_per_second


@dataclass(frozen=True)
class ProjectionEstimate:
    inputs: ProjectionInputs
    hardware: HardwareRates

    @property
    def flops(self) -> int:
        value = self.inputs
        return 2 * value.rows * value.input_width * value.output_width

    @property
    def weight_bytes(self) -> int:
        value = self.inputs
        return value.input_width * value.output_width * value.bytes_per_weight

    @property
    def input_bytes(self) -> int:
        value = self.inputs
        return value.rows * value.input_width * value.bytes_per_activation

    @property
    def output_bytes(self) -> int:
        value = self.inputs
        return value.rows * value.output_width * value.bytes_per_activation

    @property
    def total_hbm_bytes(self) -> int:
        return self.weight_bytes + self.input_bytes + self.output_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_hbm_bytes

    @property
    def compute_time_us(self) -> float:
        return self.flops / self.hardware.compute_flops_per_second * 1e6

    @property
    def memory_time_us(self) -> float:
        return self.total_hbm_bytes / self.hardware.hbm_bytes_per_second * 1e6

    @property
    def lower_bound_us(self) -> float:
        return max(self.compute_time_us, self.memory_time_us)

    @property
    def bottleneck(self) -> str:
        return "HBM bandwidth" if self.memory_time_us > self.compute_time_us else "FP16 compute"

    def transfer_time_us(self, byte_count: int) -> float:
        return byte_count / self.hardware.hbm_bytes_per_second * 1e6


def problem_02_estimate(rows: int) -> ProjectionEstimate:
    return ProjectionEstimate(
        ProjectionInputs(
            rows=rows,
            input_width=4096,
            output_width=4096,
            bytes_per_weight=2,
            bytes_per_activation=2,
        ),
        HardwareRates(compute_tflops=120, hbm_bandwidth_gbps=600),
    )


def decimal_bytes(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} GB"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MB"
    if value >= 1_000:
        return f"{value / 1_000:.2f} KB"
    return f"{value} B"


def decimal_flops(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} GFLOPs"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MFLOPs"
    return f"{value} FLOPs"


def projection_calculations(estimate: ProjectionEstimate) -> dict[str, CalculationDetail]:
    """Beginner-oriented provenance for every derived projection quantity."""

    value = estimate.inputs
    hardware = estimate.hardware
    common_inputs = (
        CalculationInput("M", f"{value.rows}", "Number of token rows processed together", "First dimension of X"),
        CalculationInput("K", f"{value.input_width}", "Number of input features consumed by each output dot product", "Second dimension of X and first dimension of W"),
        CalculationInput("N", f"{value.output_width}", "Number of output features produced for each row", "Second dimension of W and Y"),
    )
    byte_inputs = common_inputs + (
        CalculationInput("s_w", f"{value.bytes_per_weight} bytes/value", "Storage occupied by one FP16 weight", "Problem datatype assumption"),
        CalculationInput("s_a", f"{value.bytes_per_activation} bytes/value", "Storage occupied by one FP16 activation", "Problem datatype assumption"),
    )
    work = CalculationDetail(
        "Matrix-multiplication work",
        "The projection creates an M-by-N output matrix. Every output value is a dot product containing K multiplications. Under the standard performance-modeling convention, each contribution counts as one multiply plus one add, or approximately two floating-point operations.",
        "FLOPs = 2 × M × K × N",
        common_inputs + (CalculationInput("2", "2 FLOPs/contribution", "One multiplication plus one addition", "FLOP-counting convention"),),
        (
            CalculationStep("Count output values", f"M × N = {value.rows} × {value.output_width} = {value.rows * value.output_width:,}", "There is one output value for every row and output feature."),
            CalculationStep("Count contributions", f"{value.rows * value.output_width:,} outputs × {value.input_width} contributions/output = {value.rows * value.output_width * value.input_width:,} contributions", "Each output is a length-K dot product."),
            CalculationStep("Convert contributions to FLOPs", f"2 × {value.rows} × {value.input_width} × {value.output_width} = {estimate.flops:,} FLOPs", "Each contribution is modeled as one multiplication and one addition."),
            CalculationStep("Readable scale", f"{estimate.flops:,} FLOPs = {decimal_flops(estimate.flops)}", "Decimal prefixes are used for performance quantities."),
        ),
        "The row, feature, and contribution counts are dimensionless. Multiplying them by FLOPs/contribution leaves FLOPs.",
        f"This single projection performs {decimal_flops(estimate.flops)} of arithmetic for {value.rows} token row{'s' if value.rows != 1 else ''}.",
        ("Uses the conventional approximate 2MKN count rather than exactly K multiplies and K−1 adds per dot product.",),
    )

    def tensor_bytes_detail(name: str, dimensions: str, count: int, size: int, traffic: str) -> CalculationDetail:
        return CalculationDetail(
            f"{name} {traffic}",
            f"A tensor's byte size equals its number of scalar values multiplied by the storage occupied by each value. The simplified traffic model counts this tensor crossing the HBM boundary once.",
            "bytes = number of values × bytes per value",
            byte_inputs,
            (
                CalculationStep("Read the tensor shape", dimensions, "Multiply all shape dimensions to count scalar values."),
                CalculationStep("Count values", f"{count // size:,} values", "This is the number of FP16 scalars in the tensor."),
                CalculationStep("Convert values to bytes", f"{count // size:,} values × {size} bytes/value = {count:,} bytes", "FP16 occupies two bytes for each stored value."),
                CalculationStep("Readable scale", f"{count:,} bytes = {decimal_bytes(count)}", "The diagram uses decimal KB and MB so it matches GB/s hardware-rate units."),
            ),
            "values × bytes/value = bytes; the value unit cancels.",
            f"The model counts {decimal_bytes(count)} of {name.lower()} traffic at the HBM boundary.",
            ("Counts one transfer and excludes cache effects, rereads, alignment, metadata, and workspace.",),
        )

    weight = tensor_bytes_detail("Weight", f"W shape = K × N = {value.input_width} × {value.output_width}", estimate.weight_bytes, value.bytes_per_weight, "read")
    input_bytes = tensor_bytes_detail("Input", f"X shape = M × K = {value.rows} × {value.input_width}", estimate.input_bytes, value.bytes_per_activation, "read")
    output_bytes = tensor_bytes_detail("Output", f"Y shape = M × N = {value.rows} × {value.output_width}", estimate.output_bytes, value.bytes_per_activation, "write")

    total = CalculationDetail(
        "Total modeled HBM traffic",
        "The operation requires weights and input activations before it can compute and writes output activations afterward. The simplified model adds these three non-overlapping byte categories.",
        "total HBM bytes = weight read + input read + output write",
        byte_inputs,
        (
            CalculationStep("Weight read", f"{estimate.weight_bytes:,} bytes", "The shared K-by-N FP16 weight matrix is read once per modeled call."),
            CalculationStep("Input read", f"{estimate.input_bytes:,} bytes", "The M-by-K FP16 input matrix is read once."),
            CalculationStep("Output write", f"{estimate.output_bytes:,} bytes", "The M-by-N FP16 result is written once."),
            CalculationStep("Add traffic", f"{estimate.weight_bytes:,} + {estimate.input_bytes:,} + {estimate.output_bytes:,} = {estimate.total_hbm_bytes:,} bytes", "Traffic categories add because all three cross the named HBM boundary."),
            CalculationStep("Readable scale", f"{estimate.total_hbm_bytes:,} bytes = {decimal_bytes(estimate.total_hbm_bytes)}", "This is a traffic quantity, not GPU memory capacity."),
        ),
        "bytes + bytes + bytes = bytes.",
        f"The projection is modeled as moving {decimal_bytes(estimate.total_hbm_bytes)} across the HBM boundary.",
        ("W is counted once per model call.", "No cache hits, extra rereads, workspace, fusion, or allocator traffic are included."),
    )

    ai = CalculationDetail(
        "Arithmetic intensity",
        "Arithmetic intensity measures how much useful arithmetic is obtained from every byte that crosses a named memory boundary. Here the boundary is HBM-to-chip. It is a workload ratio, not a speed or elapsed time.",
        "arithmetic intensity = FLOPs ÷ HBM bytes",
        common_inputs,
        (
            CalculationStep("Use calculated work", f"work = {estimate.flops:,} FLOPs", "This comes from 2MKN."),
            CalculationStep("Use calculated traffic", f"traffic = {estimate.total_hbm_bytes:,} bytes", "This is weight read plus input read plus output write."),
            CalculationStep("Divide", f"{estimate.flops:,} FLOPs ÷ {estimate.total_hbm_bytes:,} bytes = {estimate.arithmetic_intensity:.6f} FLOPs/byte", "The ratio states how much work each modeled HBM byte supports."),
            CalculationStep("Display rounding", f"{estimate.arithmetic_intensity:.6f} → {estimate.arithmetic_intensity:.4f} FLOPs/byte", "The UI rounds only the displayed value; calculations retain full precision."),
        ),
        "FLOPs ÷ bytes = FLOPs/byte.",
        f"At {estimate.arithmetic_intensity:.4f} FLOPs/byte, this operation is {'below' if estimate.arithmetic_intensity < hardware.ridge_point_flops_per_byte else 'above'} the {hardware.ridge_point_flops_per_byte:.1f} FLOPs/byte hardware ridge point.",
        ("Arithmetic intensity changes if actual HBM traffic differs from this simplified model.",),
    )

    compute = CalculationDetail(
        "Ideal compute-time bound",
        "This asks how long the arithmetic would take if the GPU sustained its full stated FP16 compute rate. It ignores memory stalls and all overhead, so it is an optimistic lower bound.",
        "compute time = FLOPs ÷ peak compute rate",
        common_inputs + (CalculationInput("P", f"{hardware.compute_tflops} TFLOP/s", "Hypothetical peak FP16 compute rate", "Problem 02 hardware assumption"),),
        (
            CalculationStep("Convert peak rate", f"{hardware.compute_tflops} TFLOP/s × 10¹² = {hardware.compute_flops_per_second:,.0f} FLOPs/s", "Tera is the decimal prefix for 10¹²."),
            CalculationStep("Divide work by rate", f"{estimate.flops:,} FLOPs ÷ {hardware.compute_flops_per_second:,.0f} FLOPs/s = {estimate.compute_time_us / 1e6:.12f} s", "FLOPs cancel and leave seconds."),
            CalculationStep("Convert seconds", f"{estimate.compute_time_us / 1e6:.12f} s × 10⁶ µs/s = {estimate.compute_time_us:.6f} µs", "One second contains one million microseconds."),
            CalculationStep("Display rounding", f"{estimate.compute_time_us:.6f} → {estimate.compute_time_us:.4f} µs", "The UI shows four digits after the decimal point."),
        ),
        "FLOPs ÷ (FLOPs/s) = seconds; seconds × µs/second = µs.",
        f"Even perfect compute utilization could not finish the arithmetic in less than approximately {estimate.compute_time_us:.4f} µs.",
        ("Assumes the matrix shape can use the stated FP16 peak.", "Excludes launch, synchronization, memory, and non-matmul work."),
    )

    memory = CalculationDetail(
        "Ideal HBM-time bound",
        "This asks how long the modeled bytes would take to cross the HBM boundary if the GPU sustained full peak HBM bandwidth. It is an optimistic lower bound, not measured latency.",
        "memory time = HBM bytes ÷ peak HBM bandwidth",
        byte_inputs + (CalculationInput("BW", f"{hardware.hbm_bandwidth_gbps} GB/s", "Hypothetical peak HBM bandwidth", "Problem 02 hardware assumption"),),
        (
            CalculationStep("Convert bandwidth", f"{hardware.hbm_bandwidth_gbps} GB/s × 10⁹ = {hardware.hbm_bytes_per_second:,.0f} bytes/s", "Performance specifications use decimal GB."),
            CalculationStep("Divide traffic by rate", f"{estimate.total_hbm_bytes:,} bytes ÷ {hardware.hbm_bytes_per_second:,.0f} bytes/s = {estimate.memory_time_us / 1e6:.12f} s", "Bytes cancel and leave seconds."),
            CalculationStep("Convert seconds", f"{estimate.memory_time_us / 1e6:.12f} s × 10⁶ µs/s = {estimate.memory_time_us:.6f} µs", "Convert the result to the latency unit used in the workbench."),
            CalculationStep("Display rounding", f"{estimate.memory_time_us:.6f} → {estimate.memory_time_us:.4f} µs", "The model retains the unrounded value internally."),
        ),
        "bytes ÷ (bytes/s) = seconds; seconds × µs/second = µs.",
        f"Even perfect bandwidth utilization could not move the modeled traffic in less than approximately {estimate.memory_time_us:.4f} µs.",
        ("Assumes peak bandwidth is attainable.", "Excludes cache effects and traffic omitted by the simplified model."),
    )

    lower = CalculationDetail(
        "Roofline lower-bound time",
        "The operation needs both its arithmetic and its data movement. In the ideal roofline model they overlap perfectly, so the slower resource determines the earliest possible completion time. Taking the maximum avoids double-counting perfectly overlapped work.",
        "roofline lower bound = max(compute-time bound, memory-time bound)",
        (
            CalculationInput("T_compute", f"{estimate.compute_time_us:.6f} µs", "Ideal arithmetic time", "Compute-bound calculation"),
            CalculationInput("T_memory", f"{estimate.memory_time_us:.6f} µs", "Ideal HBM traffic time", "Memory-bound calculation"),
        ),
        (
            CalculationStep("Compare bounds", f"max({estimate.compute_time_us:.6f}, {estimate.memory_time_us:.6f}) µs", "Select the larger ideal resource time."),
            CalculationStep("Choose limiting time", f"= {estimate.lower_bound_us:.6f} µs", f"The {estimate.bottleneck.lower()} bound is larger."),
            CalculationStep("Display rounding", f"{estimate.lower_bound_us:.6f} → {estimate.lower_bound_us:.4f} µs", "This remains a theoretical floor, not a prediction of measured kernel duration."),
        ),
        "Both operands already use microseconds, so the maximum also has units of microseconds.",
        f"The idealized operation cannot complete faster than {estimate.lower_bound_us:.4f} µs; real execution should be equal or slower.",
        ("Assumes ideal overlap between compute and HBM demand.", "Does not include kernel launch or framework overhead."),
    )

    ridge = CalculationDetail(
        "Hardware ridge point",
        "The ridge point describes the GPU's balance: how many FLOPs each delivered HBM byte must enable for peak bandwidth to supply enough data for peak compute.",
        "ridge point = peak compute rate ÷ peak HBM bandwidth",
        (
            CalculationInput("P", f"{hardware.compute_tflops} TFLOP/s", "Peak FP16 compute", "Problem 02 hardware assumption"),
            CalculationInput("BW", f"{hardware.hbm_bandwidth_gbps} GB/s", "Peak HBM bandwidth", "Problem 02 hardware assumption"),
        ),
        (
            CalculationStep("Expand prefixes", f"({hardware.compute_tflops} × 10¹² FLOPs/s) ÷ ({hardware.hbm_bandwidth_gbps} × 10⁹ bytes/s)", "Convert tera and giga to base units."),
            CalculationStep("Cancel time", "(FLOPs/s) ÷ (bytes/s) = FLOPs/byte", "The per-second terms cancel."),
            CalculationStep("Calculate", f"{hardware.compute_flops_per_second:,.0f} ÷ {hardware.hbm_bytes_per_second:,.0f} = {hardware.ridge_point_flops_per_byte:.1f} FLOPs/byte", "This is the workload intensity required to reach both ideal ceilings together."),
        ),
        "(FLOPs/s) / (bytes/s) = FLOPs/byte.",
        f"Operations below {hardware.ridge_point_flops_per_byte:.1f} FLOPs/byte are on the memory side of this ideal roofline; operations above it are on the compute side.",
        ("The ridge changes with datatype because the compute ceiling changes.",),
    )

    bottleneck = CalculationDetail(
        "Predicted limiting resource",
        "A roofline classification can be obtained either by comparing the two time bounds or by comparing workload arithmetic intensity with the hardware ridge point. Both tests are equivalent under the same assumptions.",
        "bottleneck = resource with the larger ideal time bound",
        (
            CalculationInput("AI", f"{estimate.arithmetic_intensity:.4f} FLOPs/byte", "Workload arithmetic intensity", "Arithmetic-intensity calculation"),
            CalculationInput("ridge", f"{hardware.ridge_point_flops_per_byte:.1f} FLOPs/byte", "Hardware balance point", "Ridge-point calculation"),
            CalculationInput("T_compute", f"{estimate.compute_time_us:.4f} µs", "Compute-time bound", "Compute-time calculation"),
            CalculationInput("T_memory", f"{estimate.memory_time_us:.4f} µs", "Memory-time bound", "Memory-time calculation"),
        ),
        (
            CalculationStep("Compare intensity", f"{estimate.arithmetic_intensity:.4f} {'<' if estimate.arithmetic_intensity < hardware.ridge_point_flops_per_byte else '>'} {hardware.ridge_point_flops_per_byte:.1f} FLOPs/byte", "This places the workload on one side of the ridge."),
            CalculationStep("Compare time bounds", f"T_memory = {estimate.memory_time_us:.4f} µs; T_compute = {estimate.compute_time_us:.4f} µs", "The larger resource time sets the roofline floor."),
            CalculationStep("Classify", f"Predicted bottleneck = {estimate.bottleneck}", "This is an idealized hypothesis to test with profiling."),
        ),
        "The comparison uses matching units on each side.",
        f"The model predicts that improving {estimate.bottleneck.lower()} is the relevant first direction, but measurement must confirm it.",
        ("A roofline classification does not prove the cause of measured latency.",),
    )

    return {
        "work": work,
        "weight_bytes": weight,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total,
        "arithmetic_intensity": ai,
        "compute_time": compute,
        "memory_time": memory,
        "lower_bound": lower,
        "ridge_point": ridge,
        "bottleneck": bottleneck,
    }


def transfer_calculation(
    *, name: str, byte_count: int, bandwidth_gbps: float, source: str
) -> CalculationDetail:
    time_us = byte_count / (bandwidth_gbps * 1e9) * 1e6
    return CalculationDetail(
        f"{name} transfer-time bound",
        "A transfer-time bound divides the amount of data crossing one physical boundary by that boundary's maximum delivery rate. It is the fastest ideal transfer, not a measured latency.",
        "transfer time = transferred bytes ÷ bandwidth",
        (
            CalculationInput("Q", f"{byte_count:,} bytes", "Data crossing this path", source),
            CalculationInput("BW", f"{bandwidth_gbps} GB/s", "Peak or assumed path bandwidth", "Selected hardware model"),
        ),
        (
            CalculationStep("Convert bandwidth", f"{bandwidth_gbps} GB/s × 10⁹ = {bandwidth_gbps * 1e9:,.0f} bytes/s", "Giga is the decimal prefix for one billion."),
            CalculationStep("Divide bytes by rate", f"{byte_count:,} bytes ÷ {bandwidth_gbps * 1e9:,.0f} bytes/s = {time_us / 1e6:.12f} s", "The byte units cancel, leaving seconds."),
            CalculationStep("Convert seconds", f"{time_us / 1e6:.12f} s × 10⁶ µs/s = {time_us:.6f} µs", "The workbench presents short hardware times in microseconds."),
            CalculationStep("Display rounding", f"{time_us:.6f} → {time_us:.4f} µs", "The internal result retains more precision."),
        ),
        "bytes ÷ (bytes/s) = seconds; seconds × µs/second = µs.",
        f"At an ideal {bandwidth_gbps} GB/s, moving {decimal_bytes(byte_count)} cannot take less than approximately {time_us:.4f} µs.",
        ("Assumes the path sustains its full stated rate.", "Excludes protocol, queueing, setup, and synchronization overhead."),
    )
