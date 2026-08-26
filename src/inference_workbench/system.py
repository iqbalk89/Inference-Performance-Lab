"""Injectable system-topology implementations."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import AcceleratorModel, Component, Connection, Diagram, EvidenceKind, Metric, SystemModel


@dataclass(frozen=True)
class SingleAcceleratorSystem(SystemModel):
    """A host system composed entirely from injected component records."""

    client: Component
    server: Component
    cpu: Component
    host_memory: Component
    host_link: Component

    def diagram(self, accelerator: AcceleratorModel) -> Diagram:
        gpu_component = accelerator.system_component()
        return Diagram(
            "system",
            "Inference System",
            "Begin with the complete host and accelerator data path; click the GPU to push in.",
            None,
            (self.client, self.server, self.cpu, self.host_memory, self.host_link, gpu_component),
            (
                Connection("client-server", self.client.component_id, self.server.component_id, "request"),
                Connection("server-cpu", self.server.component_id, self.cpu.component_id, "scheduled work"),
                Connection("cpu-dram", self.cpu.component_id, self.host_memory.component_id, "host data", "both"),
                Connection("cpu-link", self.cpu.component_id, self.host_link.component_id, "2.05 KB token IDs", "both"),
                Connection(
                    "link-gpu", self.host_link.component_id, gpu_component.component_id,
                    "2.05 KB @ 32 GB/s → 0.064 µs", "both",
                    metrics=(
                        Metric("Prefill payload", 2048, "bytes", EvidenceKind.THEORETICAL, derivation="512 token IDs × 4 bytes"),
                        Metric("Assumed peak rate", 32, "GB/s", EvidenceKind.ASSUMED),
                        Metric("Ideal transfer bound", 0.064, "µs", EvidenceKind.THEORETICAL, derivation="2,048 / 32 × 10⁹ seconds"),
                    ),
                ),
            ),
            ("This Slice 0 topology is educational and intentionally single-GPU.",),
        )
