"""Stable contracts shared by model implementations and the visual client."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable, Generic, TypeVar


class EvidenceKind(StrEnum):
    THEORETICAL = "theoretical"
    ASSUMED = "assumed"
    CALIBRATED = "calibrated"
    MEASURED = "measured"


class ComponentKind(StrEnum):
    SYSTEM = "system"
    SERVICE = "service"
    PROCESSOR = "processor"
    MEMORY = "memory"
    INTERCONNECT = "interconnect"
    ACCELERATOR = "accelerator"
    COMPUTE = "compute"
    PHASE = "phase"
    OPERATION = "operation"


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | int | str
    unit: str = ""
    evidence: EvidenceKind = EvidenceKind.THEORETICAL
    description: str = ""
    derivation: str = ""


@dataclass(frozen=True)
class Position:
    x: float
    y: float


@dataclass(frozen=True)
class Component:
    component_id: str
    label: str
    kind: ComponentKind
    summary: str
    position: Position
    metrics: tuple[Metric, ...] = ()
    drilldown_graph_id: str | None = None
    lane: str = "hardware"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        for metric in result["metrics"]:
            metric["evidence"] = metric["evidence"].value
        return result


@dataclass(frozen=True)
class Connection:
    connection_id: str
    source_id: str
    target_id: str
    label: str
    direction: str = "forward"
    category: str = "data"
    metrics: tuple[Metric, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for metric in result["metrics"]:
            metric["evidence"] = metric["evidence"].value
        return result


@dataclass(frozen=True)
class Diagram:
    graph_id: str
    title: str
    subtitle: str
    parent_graph_id: str | None
    components: tuple[Component, ...]
    connections: tuple[Connection, ...]
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "parent_graph_id": self.parent_graph_id,
            "components": [component.to_dict() for component in self.components],
            "connections": [connection.to_dict() for connection in self.connections],
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class WorkbenchScenario:
    scenario_id: str
    title: str
    initial_graph_id: str
    diagrams: tuple[Diagram, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "initial_graph_id": self.initial_graph_id,
            "diagrams": {
                diagram.graph_id: diagram.to_dict() for diagram in self.diagrams
            },
            "metadata": self.metadata,
        }


class MemoryModel(ABC):
    """Injectable description of an accelerator memory hierarchy."""

    @abstractmethod
    def components(self) -> tuple[Component, ...]:
        raise NotImplementedError

    @abstractmethod
    def connections(self) -> tuple[Connection, ...]:
        raise NotImplementedError


class ComputeModel(ABC):
    """Injectable description of accelerator compute resources."""

    @abstractmethod
    def components(self) -> tuple[Component, ...]:
        raise NotImplementedError


class AcceleratorModel(ABC):
    """Interface implemented by any GPU or future accelerator model."""

    @abstractmethod
    def system_component(self) -> Component:
        raise NotImplementedError

    @abstractmethod
    def diagram(self) -> Diagram:
        raise NotImplementedError


class SystemModel(ABC):
    """Injectable host, service, interconnect, and accelerator topology."""

    @abstractmethod
    def diagram(self, accelerator: AcceleratorModel) -> Diagram:
        raise NotImplementedError


BuilderT = TypeVar("BuilderT", bound=Callable[..., Any])


class VariantRegistry(Generic[BuilderT]):
    """Small dependency-injection registry for named component variants."""

    def __init__(self) -> None:
        self._builders: dict[str, BuilderT] = {}

    def register(self, name: str, builder: BuilderT) -> None:
        if name in self._builders:
            raise ValueError(f"Variant already registered: {name}")
        self._builders[name] = builder

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            builder = self._builders[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._builders)) or "none"
            raise KeyError(f"Unknown variant '{name}'. Available: {choices}") from exc
        return builder(**kwargs)

    @property
    def variants(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))
