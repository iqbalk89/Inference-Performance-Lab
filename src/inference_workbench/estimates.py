"""Auditable analytical estimates used by the first executable workbench slice."""

from __future__ import annotations

from dataclasses import dataclass


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
